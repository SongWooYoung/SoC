#pragma once

// Qwen3.5 Text Model — C++ inference implementation
// Phase 3: bottom-up construction matching modeling_qwen3_5.py
//
// All computation is f32 on CPU (Accelerate/BLAS).
// Weights stay mmap'd as bf16, converted on-the-fly per operation.

#include "models/qwen3_5/config.h"
#include "utils/ops.h"
#include "utils/safetensors.h"

#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

// ═══════════════════════════════════════════════════════════════════════════
// Forward declarations & cache types
// ═══════════════════════════════════════════════════════════════════════════

struct KVCache {
    std::vector<float> key;     // [cur_len, num_kv_heads, head_dim]
    std::vector<float> value;   // [cur_len, num_kv_heads, head_dim]
    int length = 0;
    int num_kv_heads = 0;
    int head_dim = 0;

    void init(int nkv, int hd) {
        num_kv_heads = nkv;
        head_dim = hd;
        length = 0;
        key.clear();
        value.clear();
    }

    void append(const float* new_k, const float* new_v, int new_len) {
        size_t row = static_cast<size_t>(num_kv_heads) * head_dim;
        size_t add = static_cast<size_t>(new_len) * row;
        key.insert(key.end(), new_k, new_k + add);
        value.insert(value.end(), new_v, new_v + add);
        length += new_len;
    }
};

struct GDNCache {
    std::vector<float> conv_state;       // [conv_dim, kernel_size-1]
    std::vector<float> recurrent_state;  // [num_v_heads, k_head_dim, v_head_dim]
    std::vector<float> conv_weight_f32;  // pre-converted conv weight
    bool has_state = false;

    void init(int conv_dim, int kernel_size,
              int num_v_heads, int k_head_dim, int v_head_dim,
              const uint16_t* conv_w_bf16) {
        int ks_minus1 = kernel_size - 1;
        conv_state.assign(static_cast<size_t>(conv_dim) * ks_minus1, 0.0f);
        recurrent_state.assign(
            static_cast<size_t>(num_v_heads) * k_head_dim * v_head_dim, 0.0f);
        has_state = false;

        // Pre-convert conv weight
        int wn = conv_dim * kernel_size;
        conv_weight_f32.resize(static_cast<size_t>(wn));
        bf16_to_f32_array(conv_weight_f32.data(), conv_w_bf16,
                          static_cast<size_t>(wn));
    }
};

struct ModelCache {
    std::vector<KVCache> kv_caches;          // one per full-attention layer
    std::vector<GDNCache> gdn_caches;        // one per linear-attention layer
    int seq_offset = 0;                      // total tokens processed so far
};

// ═══════════════════════════════════════════════════════════════════════════
// 3a. RMSNorm
// ═══════════════════════════════════════════════════════════════════════════

struct Qwen3_5RMSNorm {
    std::vector<float> weight;   // loaded from safetensors (trained values)
    float eps = 1e-6f;
    int dim = 0;

    void load(const SafetensorsBundle& bundle, const std::string& prefix) {
        weight = bundle.load_f32(prefix + ".weight");
        dim = static_cast<int>(weight.size());
    }

    // out = x * rsqrt(mean(x²) + eps) * (1 + weight)
    void forward(float* out, const float* x, int tokens) const {
        for (int t = 0; t < tokens; t++) {
            const float* xi = x + t * dim;
            float* oi = out + t * dim;
            rmsnorm(oi, xi, weight.data(), dim, eps);
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// 3a'. RMSNormGated (for GatedDeltaNet)
// ═══════════════════════════════════════════════════════════════════════════

struct Qwen3_5RMSNormGated {
    std::vector<float> weight;   // initialized to ones in Python
    float eps = 1e-6f;
    int dim = 0;

    void load(const SafetensorsBundle& bundle, const std::string& prefix) {
        weight = bundle.load_f32(prefix + ".weight");
        dim = static_cast<int>(weight.size());
    }

    // out = weight * (x * rsqrt(var+eps)) * silu(gate)
    void forward(float* out, const float* x, const float* gate) const {
        rmsnorm_gated(out, x, gate, weight.data(), dim, eps);
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// 3b. Rotary Embedding (RoPE + MRoPE)
// ═══════════════════════════════════════════════════════════════════════════

struct Qwen3_5RotaryEmbedding {
    std::vector<float> inv_freq;     // [rotary_dim / 2]
    std::vector<int> mrope_section;  // e.g. [11, 11, 10]
    float attention_scaling = 1.0f;
    int rotary_dim = 0;

    void init(const Qwen3_5TextConfig& cfg) {
        rotary_dim = cfg.rotary_dim();
        int half = rotary_dim / 2;
        double base = cfg.rope_parameters.rope_theta;
        mrope_section = cfg.rope_parameters.mrope_section;

        inv_freq.resize(half);
        for (int i = 0; i < half; i++) {
            double exp = static_cast<double>(2 * i) / rotary_dim;
            inv_freq[i] = static_cast<float>(1.0 / std::pow(base, exp));
        }
    }

    // Compute cos/sin tables for given position_ids
    // position_ids: [3, seq_len]  (temporal, height, width)
    // Output cos/sin: [seq_len, rotary_dim]
    void forward(float* cos_out, float* sin_out,
                 const int* position_ids, int seq_len) const {
        int half = rotary_dim / 2;

        // Step 1: freqs[3, seq_len, half] = position_ids[dim, :, None] * inv_freq[None, :]
        std::vector<float> freqs(3 * seq_len * half);
        for (int d = 0; d < 3; d++) {
            for (int s = 0; s < seq_len; s++) {
                float pos = static_cast<float>(position_ids[d * seq_len + s]);
                for (int f = 0; f < half; f++) {
                    freqs[(d * seq_len + s) * half + f] = pos * inv_freq[f];
                }
            }
        }

        // Step 2: apply interleaved MRoPE
        // Start with temporal freqs, then interleave H, W
        // mrope_section = [s0, s1, s2] where s0+s1+s2 ≈ half
        // For each freq index, assign it to the appropriate dimension
        // Interleaved pattern: indices 0,3,6,... → temporal
        //                      indices 1,4,7,... → height
        //                      indices 2,5,8,... → width
        std::vector<float> merged(seq_len * half);

        // Copy temporal (dim=0) as base
        for (int s = 0; s < seq_len; s++)
            std::memcpy(merged.data() + s * half,
                        freqs.data() + s * half,
                        half * sizeof(float));

        // Overwrite height (dim=1) at stride-3 positions starting at offset 1
        if (mrope_section.size() > 1) {
            int length = mrope_section[1] * 3;
            for (int s = 0; s < seq_len; s++) {
                for (int i = 1; i < length && i < half; i += 3) {
                    merged[s * half + i] =
                        freqs[(1 * seq_len + s) * half + i];
                }
            }
        }

        // Overwrite width (dim=2) at stride-3 positions starting at offset 2
        if (mrope_section.size() > 2) {
            int length = mrope_section[2] * 3;
            for (int s = 0; s < seq_len; s++) {
                for (int i = 2; i < length && i < half; i += 3) {
                    merged[s * half + i] =
                        freqs[(2 * seq_len + s) * half + i];
                }
            }
        }

        // Step 3: emb = cat(merged, merged) → [seq_len, rotary_dim]
        // cos = cos(emb) * attention_scaling
        // sin = sin(emb) * attention_scaling
        for (int s = 0; s < seq_len; s++) {
            for (int i = 0; i < half; i++) {
                float f = merged[s * half + i];
                cos_out[s * rotary_dim + i]        = std::cos(f) * attention_scaling;
                cos_out[s * rotary_dim + half + i]  = std::cos(f) * attention_scaling;
                sin_out[s * rotary_dim + i]        = std::sin(f) * attention_scaling;
                sin_out[s * rotary_dim + half + i]  = std::sin(f) * attention_scaling;
            }
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// 3c. MLP
// ═══════════════════════════════════════════════════════════════════════════

struct Qwen3_5MLP {
    const uint16_t* gate_proj = nullptr;   // [intermediate, hidden]
    const uint16_t* up_proj   = nullptr;   // [intermediate, hidden]
    const uint16_t* down_proj = nullptr;   // [hidden, intermediate]
    int hidden_size = 0;
    int intermediate_size = 0;

    void load(const SafetensorsBundle& bundle, const std::string& prefix,
              const Qwen3_5TextConfig& cfg) {
        hidden_size = cfg.hidden_size;
        intermediate_size = cfg.intermediate_size;
        gate_proj = bundle.bf16(prefix + ".gate_proj.weight");
        up_proj   = bundle.bf16(prefix + ".up_proj.weight");
        down_proj = bundle.bf16(prefix + ".down_proj.weight");
    }

    // out[tokens, hidden] = down(silu(gate(x)) * up(x))
    void forward(float* out, const float* x, int tokens,
                 Scratch& scratch) const {
        int M = tokens, K = hidden_size, N = intermediate_size;

        // gate = gate_proj(x)  [M, N]
        std::vector<float> gate(static_cast<size_t>(M) * N);
        linear_bf16(gate.data(), x, gate_proj, M, K, N, scratch);

        // up = up_proj(x)  [M, N]
        std::vector<float> up(static_cast<size_t>(M) * N);
        linear_bf16(up.data(), x, up_proj, M, K, N, scratch);

        // gate = silu(gate) * up
        silu_inplace(gate.data(), M * N);
        vec_mul(gate.data(), gate.data(), up.data(), M * N);

        // out = down_proj(gate)  [M, hidden]
        linear_bf16(out, gate.data(), down_proj, M, N, K, scratch);
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// 3d. Attention (full, with gating)
// ═══════════════════════════════════════════════════════════════════════════

struct Qwen3_5Attention {
    const uint16_t* q_proj = nullptr;  // [num_heads*head_dim*2, hidden]
    const uint16_t* k_proj = nullptr;  // [num_kv_heads*head_dim, hidden]
    const uint16_t* v_proj = nullptr;  // [num_kv_heads*head_dim, hidden]
    const uint16_t* o_proj = nullptr;  // [hidden, num_heads*head_dim]

    Qwen3_5RMSNorm q_norm;
    Qwen3_5RMSNorm k_norm;

    int hidden_size = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int rotary_dim = 0;
    float scale = 0.0f;
    int layer_idx = 0;

    void load(const SafetensorsBundle& bundle, const std::string& prefix,
              const Qwen3_5TextConfig& cfg, int idx) {
        hidden_size = cfg.hidden_size;
        num_heads = cfg.num_attention_heads;
        num_kv_heads = cfg.num_key_value_heads;
        head_dim = cfg.head_dim;
        rotary_dim = cfg.rotary_dim();
        scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        layer_idx = idx;

        q_proj = bundle.bf16(prefix + ".q_proj.weight");
        k_proj = bundle.bf16(prefix + ".k_proj.weight");
        v_proj = bundle.bf16(prefix + ".v_proj.weight");
        o_proj = bundle.bf16(prefix + ".o_proj.weight");

        q_norm.load(bundle, prefix + ".q_norm");
        k_norm.load(bundle, prefix + ".k_norm");
    }

    // hidden_states: [seq_len, hidden_size]
    // cos/sin: [seq_len, rotary_dim]
    // mask: [seq_len, kv_len]  (additive causal mask or nullptr)
    // out: [seq_len, hidden_size]
    void forward(float* out, const float* hidden_states,
                 const float* cos_vals, const float* sin_vals,
                 const float* mask,
                 int seq_len, KVCache& cache, Scratch& scratch) const {
        int M = seq_len;
        int qkv_head_dim = head_dim;

        // ── Q projection: [M, hidden] → [M, num_heads * head_dim * 2]
        int q_out_dim = num_heads * qkv_head_dim * 2;
        std::vector<float> q_raw(static_cast<size_t>(M) * q_out_dim);
        linear_bf16(q_raw.data(), hidden_states, q_proj,
                    M, hidden_size, q_out_dim, scratch);

        // Split into query and gate: each [M, num_heads, head_dim]
        int q_dim = num_heads * qkv_head_dim;
        std::vector<float> query(static_cast<size_t>(M) * q_dim);
        std::vector<float> gate(static_cast<size_t>(M) * q_dim);

        for (int t = 0; t < M; t++) {
            // q_raw layout: [M, num_heads, head_dim*2]
            // Split each head's head_dim*2 into query and gate
            for (int h = 0; h < num_heads; h++) {
                const float* src = q_raw.data() + t * q_out_dim
                                   + h * qkv_head_dim * 2;
                float* qd = query.data() + t * q_dim + h * qkv_head_dim;
                float* gd = gate.data() + t * q_dim + h * qkv_head_dim;
                std::memcpy(qd, src, qkv_head_dim * sizeof(float));
                std::memcpy(gd, src + qkv_head_dim,
                            qkv_head_dim * sizeof(float));
            }
        }

        // Reshape gate: [M, num_heads * head_dim] → will be used after attn
        // gate is already in this shape

        // ── K projection: [M, hidden] → [M, num_kv_heads * head_dim]
        int k_dim = num_kv_heads * qkv_head_dim;
        std::vector<float> key(static_cast<size_t>(M) * k_dim);
        linear_bf16(key.data(), hidden_states, k_proj,
                    M, hidden_size, k_dim, scratch);

        // ── V projection: [M, hidden] → [M, num_kv_heads * head_dim]
        std::vector<float> value(static_cast<size_t>(M) * k_dim);
        linear_bf16(value.data(), hidden_states, v_proj,
                    M, hidden_size, k_dim, scratch);

        // ── Apply Q/K norms (per head)
        // query: [M, num_heads, head_dim] → normalize each head
        for (int t = 0; t < M; t++) {
            for (int h = 0; h < num_heads; h++) {
                float* qh = query.data() + (t * num_heads + h) * qkv_head_dim;
                q_norm.forward(qh, qh, 1);
            }
        }
        // key: [M, num_kv_heads, head_dim]
        for (int t = 0; t < M; t++) {
            for (int h = 0; h < num_kv_heads; h++) {
                float* kh = key.data() + (t * num_kv_heads + h) * qkv_head_dim;
                k_norm.forward(kh, kh, 1);
            }
        }

        // ── Apply RoPE
        apply_rope(query.data(), key.data(), cos_vals, sin_vals,
                   M, num_heads, num_kv_heads, qkv_head_dim, rotary_dim);

        // ── Update KV cache
        cache.append(key.data(), value.data(), M);

        // ── Attention
        int kv_len = cache.length;
        std::vector<float> attn_out(static_cast<size_t>(M) * q_dim);

        // Build full causal mask if needed
        std::vector<float> full_mask;
        const float* mask_ptr = nullptr;
        if (mask) {
            mask_ptr = mask;
        } else if (kv_len > M) {
            // Decode: no mask needed for single token
            mask_ptr = nullptr;
        }

        attention(attn_out.data(),
                  query.data(), cache.key.data(), cache.value.data(),
                  mask_ptr,
                  M, kv_len, num_heads, num_kv_heads, qkv_head_dim,
                  scale, scratch);

        // ── Apply gate: output *= sigmoid(gate)
        sigmoid_inplace(gate.data(), M * q_dim);
        vec_mul(attn_out.data(), attn_out.data(), gate.data(), M * q_dim);

        // ── O projection: [M, num_heads*head_dim] → [M, hidden]
        linear_bf16(out, attn_out.data(), o_proj,
                    M, q_dim, hidden_size, scratch);
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// 3e. GatedDeltaNet (linear attention)
// ═══════════════════════════════════════════════════════════════════════════

struct Qwen3_5GatedDeltaNet {
    const uint16_t* in_proj_qkv = nullptr; // [key_dim*2+value_dim, hidden]
    const uint16_t* in_proj_z   = nullptr; // [value_dim, hidden]
    const uint16_t* in_proj_b   = nullptr; // [num_v_heads, hidden]
    const uint16_t* in_proj_a   = nullptr; // [num_v_heads, hidden]
    const uint16_t* conv1d_w    = nullptr; // [conv_dim, 1, kernel_size]
    const uint16_t* out_proj_w  = nullptr; // [hidden, value_dim]

    std::vector<float> dt_bias;     // [num_v_heads]
    std::vector<float> A_log;       // [num_v_heads]

    Qwen3_5RMSNormGated norm;

    int hidden_size = 0;
    int num_v_heads = 0;
    int num_k_heads = 0;
    int head_k_dim = 0;
    int head_v_dim = 0;
    int key_dim = 0;
    int value_dim = 0;
    int conv_dim = 0;
    int conv_kernel_size = 0;
    int layer_idx = 0;

    void load(const SafetensorsBundle& bundle, const std::string& prefix,
              const Qwen3_5TextConfig& cfg, int idx) {
        hidden_size = cfg.hidden_size;
        num_v_heads = cfg.linear_num_value_heads;
        num_k_heads = cfg.linear_num_key_heads;
        head_k_dim  = cfg.linear_key_head_dim;
        head_v_dim  = cfg.linear_value_head_dim;
        key_dim     = head_k_dim * num_k_heads;
        value_dim   = head_v_dim * num_v_heads;
        conv_dim    = key_dim * 2 + value_dim;
        conv_kernel_size = cfg.linear_conv_kernel_dim;
        layer_idx = idx;

        in_proj_qkv = bundle.bf16(prefix + ".in_proj_qkv.weight");
        in_proj_z   = bundle.bf16(prefix + ".in_proj_z.weight");
        in_proj_b   = bundle.bf16(prefix + ".in_proj_b.weight");
        in_proj_a   = bundle.bf16(prefix + ".in_proj_a.weight");
        conv1d_w    = bundle.bf16(prefix + ".conv1d.weight");
        out_proj_w  = bundle.bf16(prefix + ".out_proj.weight");

        dt_bias = bundle.load_f32(prefix + ".dt_bias");
        A_log   = bundle.load_f32(prefix + ".A_log");

        norm.load(bundle, prefix + ".norm");
    }

    // ── Recurrent mode (single step, decode) ────────────────────────────
    void forward_recurrent(float* out, const float* hidden_states,
                           GDNCache& cache, Scratch& scratch) const {
        int M = 1;

        // Project QKV
        std::vector<float> qkv(conv_dim);
        linear_bf16(qkv.data(), hidden_states, in_proj_qkv,
                    M, hidden_size, conv_dim, scratch);

        // Project z (gate)
        std::vector<float> z(value_dim);
        linear_bf16(z.data(), hidden_states, in_proj_z,
                    M, hidden_size, value_dim, scratch);

        // Project b (beta) and a (alpha)
        std::vector<float> b(num_v_heads);
        std::vector<float> a(num_v_heads);
        linear_bf16(b.data(), hidden_states, in_proj_b,
                    M, hidden_size, num_v_heads, scratch);
        linear_bf16(a.data(), hidden_states, in_proj_a,
                    M, hidden_size, num_v_heads, scratch);

        // Conv1d step update
        std::vector<float> conv_out(conv_dim);
        causal_conv1d_step(conv_out.data(), cache.conv_state.data(),
                           qkv.data(), cache.conv_weight_f32.data(),
                           conv_dim, conv_kernel_size);

        // SiLU activation on conv output
        silu_inplace(conv_out.data(), conv_dim);

        // Split QKV
        float* q_ptr = conv_out.data();
        float* k_ptr = conv_out.data() + key_dim;
        float* v_ptr = conv_out.data() + key_dim * 2;

        // Reshape to heads
        // q: [num_k_heads, head_k_dim]
        // k: [num_k_heads, head_k_dim]
        // v: [num_v_heads, head_v_dim]

        // L2 normalize q and k
        for (int h = 0; h < num_k_heads; h++) {
            l2norm(q_ptr + h * head_k_dim, q_ptr + h * head_k_dim,
                   head_k_dim);
            l2norm(k_ptr + h * head_k_dim, k_ptr + h * head_k_dim,
                   head_k_dim);
        }

        // Expand q/k if num_v_heads > num_k_heads (repeat interleave)
        int groups = num_v_heads / num_k_heads;
        std::vector<float> q_expanded(num_v_heads * head_k_dim);
        std::vector<float> k_expanded(num_v_heads * head_k_dim);
        for (int vh = 0; vh < num_v_heads; vh++) {
            int kh = vh / groups;
            std::memcpy(q_expanded.data() + vh * head_k_dim,
                        q_ptr + kh * head_k_dim,
                        head_k_dim * sizeof(float));
            std::memcpy(k_expanded.data() + vh * head_k_dim,
                        k_ptr + kh * head_k_dim,
                        head_k_dim * sizeof(float));
        }

        // Compute gating parameters
        // beta = sigmoid(b)
        // g = -exp(A_log) * softplus(a + dt_bias)
        std::vector<float> beta(num_v_heads);
        std::vector<float> g(num_v_heads);
        for (int h = 0; h < num_v_heads; h++) {
            beta[h] = 1.0f / (1.0f + std::exp(-b[h]));
            float A = std::exp(A_log[h]);
            g[h] = -A * softplus(a[h] + dt_bias[h]);
        }

        // Scale query
        float q_scale = 1.0f / std::sqrt(static_cast<float>(head_k_dim));

        // Recurrent step for each head
        // State S: [num_v_heads, head_k_dim, head_v_dim]
        float* S = cache.recurrent_state.data();
        std::vector<float> attn_out(num_v_heads * head_v_dim, 0.0f);

        for (int h = 0; h < num_v_heads; h++) {
            float* Sh = S + h * head_k_dim * head_v_dim;
            float exp_g = std::exp(g[h]);

            // S = S * exp(g)
            vec_scale(Sh, Sh, exp_g, head_k_dim * head_v_dim);

            // Compute memory readout: kv_mem = sum_k(S[k,:] * key[k])
            // kv_mem[v] = sum_k S[k,v]*key[k]  = key^T @ S  (as matmul)
            std::vector<float> kv_mem(head_v_dim, 0.0f);
            for (int ki = 0; ki < head_k_dim; ki++) {
                float kval = k_expanded[h * head_k_dim + ki];
                for (int vi = 0; vi < head_v_dim; vi++)
                    kv_mem[vi] += Sh[ki * head_v_dim + vi] * kval;
            }

            // delta = (v - kv_mem) * beta
            for (int vi = 0; vi < head_v_dim; vi++) {
                float delta = (v_ptr[h * head_v_dim + vi] - kv_mem[vi])
                              * beta[h];

                // S += key * delta^T  (outer product update)
                for (int ki = 0; ki < head_k_dim; ki++)
                    Sh[ki * head_v_dim + vi] +=
                        k_expanded[h * head_k_dim + ki] * delta;
            }

            // Output: o[v] = sum_k(S[k,v] * q[k] * scale)
            for (int vi = 0; vi < head_v_dim; vi++) {
                float sum = 0.0f;
                for (int ki = 0; ki < head_k_dim; ki++)
                    sum += Sh[ki * head_v_dim + vi]
                           * q_expanded[h * head_k_dim + ki];
                attn_out[h * head_v_dim + vi] = sum * q_scale;
            }
        }

        cache.has_state = true;

        // Apply RMSNormGated
        // z is reshaped to [num_v_heads, head_v_dim]
        std::vector<float> normed(num_v_heads * head_v_dim);
        for (int h = 0; h < num_v_heads; h++) {
            norm.forward(normed.data() + h * head_v_dim,
                         attn_out.data() + h * head_v_dim,
                         z.data() + h * head_v_dim);
        }

        // Out projection: [1, value_dim] → [1, hidden]
        linear_bf16(out, normed.data(), out_proj_w,
                    1, value_dim, hidden_size, scratch);
    }

    // ── Chunk mode (prefill) ────────────────────────────────────────────
    void forward_chunk(float* out, const float* hidden_states,
                       int seq_len, GDNCache& cache, Scratch& scratch) const {
        int M = seq_len;

        // Project all inputs
        std::vector<float> qkv_raw(static_cast<size_t>(M) * conv_dim);
        linear_bf16(qkv_raw.data(), hidden_states, in_proj_qkv,
                    M, hidden_size, conv_dim, scratch);

        std::vector<float> z(static_cast<size_t>(M) * value_dim);
        linear_bf16(z.data(), hidden_states, in_proj_z,
                    M, hidden_size, value_dim, scratch);

        std::vector<float> b(static_cast<size_t>(M) * num_v_heads);
        linear_bf16(b.data(), hidden_states, in_proj_b,
                    M, hidden_size, num_v_heads, scratch);

        std::vector<float> a(static_cast<size_t>(M) * num_v_heads);
        linear_bf16(a.data(), hidden_states, in_proj_a,
                    M, hidden_size, num_v_heads, scratch);

        // Causal conv1d: [seq_len, conv_dim]
        std::vector<float> conv_out(static_cast<size_t>(M) * conv_dim);
        causal_conv1d_bf16(conv_out.data(), qkv_raw.data(), conv1d_w,
                           M, conv_dim, conv_kernel_size, scratch);

        // Store conv state for future decode (last kernel_size-1 steps)
        {
            int ks1 = conv_kernel_size - 1;
            int state_start = std::max(0, M - ks1);
            for (int c = 0; c < conv_dim; c++) {
                for (int k = 0; k < ks1; k++) {
                    int t = state_start + k;
                    if (t < M)
                        cache.conv_state[c * ks1 + k] =
                            qkv_raw[t * conv_dim + c];
                    else
                        cache.conv_state[c * ks1 + k] = 0.0f;
                }
            }
        }

        // SiLU
        silu_inplace(conv_out.data(), M * conv_dim);

        // Split & reshape
        // q: [M, num_k_heads, head_k_dim]
        // k: [M, num_k_heads, head_k_dim]
        // v: [M, num_v_heads, head_v_dim]
        // (Already contiguous in conv_out as [M, key_dim, key_dim, value_dim])

        // Compute beta, g for all positions
        std::vector<float> beta(static_cast<size_t>(M) * num_v_heads);
        std::vector<float> g(static_cast<size_t>(M) * num_v_heads);
        for (int t = 0; t < M; t++) {
            for (int h = 0; h < num_v_heads; h++) {
                int idx = t * num_v_heads + h;
                beta[idx] = 1.0f / (1.0f + std::exp(-b[idx]));
                float A = std::exp(A_log[h]);
                g[idx] = -A * softplus(a[idx] + dt_bias[h]);
            }
        }

        // L2 normalize q, k per head for all tokens
        for (int t = 0; t < M; t++) {
            float* q_t = conv_out.data() + t * conv_dim;
            float* k_t = q_t + key_dim;
            for (int h = 0; h < num_k_heads; h++) {
                l2norm(q_t + h * head_k_dim, q_t + h * head_k_dim,
                       head_k_dim);
                l2norm(k_t + h * head_k_dim, k_t + h * head_k_dim,
                       head_k_dim);
            }
        }

        // Expand q, k for GQA: num_k_heads → num_v_heads
        int groups = num_v_heads / num_k_heads;
        std::vector<float> q_exp(static_cast<size_t>(M) * num_v_heads * head_k_dim);
        std::vector<float> k_exp(static_cast<size_t>(M) * num_v_heads * head_k_dim);
        std::vector<float> v_all(static_cast<size_t>(M) * num_v_heads * head_v_dim);

        for (int t = 0; t < M; t++) {
            float* q_src = conv_out.data() + t * conv_dim;
            float* k_src = q_src + key_dim;
            float* v_src = k_src + key_dim;

            for (int vh = 0; vh < num_v_heads; vh++) {
                int kh = vh / groups;
                std::memcpy(q_exp.data() + (t * num_v_heads + vh) * head_k_dim,
                            q_src + kh * head_k_dim,
                            head_k_dim * sizeof(float));
                std::memcpy(k_exp.data() + (t * num_v_heads + vh) * head_k_dim,
                            k_src + kh * head_k_dim,
                            head_k_dim * sizeof(float));
            }
            std::memcpy(v_all.data() + t * value_dim,
                        v_src, value_dim * sizeof(float));
        }

        // ── Recurrent processing (token-by-token for correctness)
        // This is the simpler recurrent fallback, not the chunk algorithm.
        // For prefill, this is O(seq_len * head_k_dim * head_v_dim) per head.

        float q_scale = 1.0f / std::sqrt(static_cast<float>(head_k_dim));

        float* S = cache.recurrent_state.data();
        std::vector<float> attn_out(static_cast<size_t>(M) * num_v_heads
                                    * head_v_dim, 0.0f);

        for (int h = 0; h < num_v_heads; h++) {
            float* Sh = S + h * head_k_dim * head_v_dim;

            for (int t = 0; t < M; t++) {
                float* q_t = q_exp.data()
                             + (t * num_v_heads + h) * head_k_dim;
                float* k_t = k_exp.data()
                             + (t * num_v_heads + h) * head_k_dim;
                float* v_t = v_all.data() + t * value_dim
                             + h * head_v_dim;
                float exp_g = std::exp(g[t * num_v_heads + h]);
                float beta_t = beta[t * num_v_heads + h];

                // S *= exp(g)
                vec_scale(Sh, Sh, exp_g, head_k_dim * head_v_dim);

                // kv_mem = k^T @ S → [head_v_dim]
                std::vector<float> kv_mem(head_v_dim, 0.0f);
                for (int ki = 0; ki < head_k_dim; ki++) {
                    float kval = k_t[ki];
                    for (int vi = 0; vi < head_v_dim; vi++)
                        kv_mem[vi] += Sh[ki * head_v_dim + vi] * kval;
                }

                // delta = (v - kv_mem) * beta
                // S += k * delta^T
                for (int vi = 0; vi < head_v_dim; vi++) {
                    float delta = (v_t[vi] - kv_mem[vi]) * beta_t;
                    for (int ki = 0; ki < head_k_dim; ki++)
                        Sh[ki * head_v_dim + vi] += k_t[ki] * delta;
                }

                // o = q^T @ S * scale → [head_v_dim]
                float* o_t = attn_out.data()
                             + (t * num_v_heads + h) * head_v_dim;
                for (int vi = 0; vi < head_v_dim; vi++) {
                    float sum = 0.0f;
                    for (int ki = 0; ki < head_k_dim; ki++)
                        sum += Sh[ki * head_v_dim + vi] * q_t[ki];
                    o_t[vi] = sum * q_scale;
                }
            }
        }

        cache.has_state = true;

        // Apply RMSNormGated per head
        std::vector<float> normed(static_cast<size_t>(M) * value_dim);
        for (int t = 0; t < M; t++) {
            for (int h = 0; h < num_v_heads; h++) {
                int offset = t * value_dim + h * head_v_dim;
                norm.forward(normed.data() + offset,
                             attn_out.data() + (t * num_v_heads + h) * head_v_dim,
                             z.data() + offset);
            }
        }

        // Out projection: [M, value_dim] → [M, hidden]
        linear_bf16(out, normed.data(), out_proj_w,
                    M, value_dim, hidden_size, scratch);
    }

    void forward(float* out, const float* hidden_states,
                 int seq_len, GDNCache& cache, Scratch& scratch) const {
        if (cache.has_state && seq_len == 1) {
            forward_recurrent(out, hidden_states, cache, scratch);
        } else {
            forward_chunk(out, hidden_states, seq_len, cache, scratch);
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// 3f. DecoderLayer
// ═══════════════════════════════════════════════════════════════════════════

struct Qwen3_5DecoderLayer {
    LayerType layer_type;
    Qwen3_5RMSNorm input_layernorm;
    Qwen3_5RMSNorm post_attention_layernorm;
    Qwen3_5MLP mlp;

    // One of these will be used depending on layer_type
    std::unique_ptr<Qwen3_5Attention> self_attn;
    std::unique_ptr<Qwen3_5GatedDeltaNet> linear_attn;

    int hidden_size = 0;
    int layer_idx = 0;

    void load(const SafetensorsBundle& bundle, const std::string& prefix,
              const Qwen3_5TextConfig& cfg, int idx) {
        hidden_size = cfg.hidden_size;
        layer_idx = idx;
        layer_type = cfg.layer_types[idx];

        input_layernorm.load(bundle, prefix + ".input_layernorm");
        post_attention_layernorm.load(bundle, prefix + ".post_attention_layernorm");
        mlp.load(bundle, prefix + ".mlp", cfg);

        if (layer_type == LayerType::FullAttention) {
            self_attn = std::make_unique<Qwen3_5Attention>();
            self_attn->load(bundle, prefix + ".self_attn", cfg, idx);
        } else {
            linear_attn = std::make_unique<Qwen3_5GatedDeltaNet>();
            linear_attn->load(bundle, prefix + ".linear_attn", cfg, idx);
        }
    }

    // hidden_states: [seq_len, hidden_size]  (in-place residual)
    void forward(float* hidden_states,
                 const float* cos_vals, const float* sin_vals,
                 const float* mask,
                 int seq_len, ModelCache& cache, Scratch& scratch) const {
        size_t sz = static_cast<size_t>(seq_len) * hidden_size;

        // ── residual = hidden_states
        std::vector<float> residual(sz);
        std::memcpy(residual.data(), hidden_states, sz * sizeof(float));

        // ── input_layernorm
        std::vector<float> normed(sz);
        input_layernorm.forward(normed.data(), hidden_states, seq_len);

        // ── Token mixer (attention or GDN)
        std::vector<float> mixer_out(sz);

        if (layer_type == LayerType::FullAttention) {
            self_attn->forward(mixer_out.data(), normed.data(),
                               cos_vals, sin_vals, mask, seq_len,
                               cache.kv_caches[layer_idx], scratch);
        } else {
            linear_attn->forward(mixer_out.data(), normed.data(),
                                 seq_len, cache.gdn_caches[layer_idx],
                                 scratch);
        }

        // ── residual + mixer_out → hidden_states
        vec_add(hidden_states, residual.data(), mixer_out.data(),
                static_cast<int>(sz));

        // ── residual = hidden_states
        std::memcpy(residual.data(), hidden_states, sz * sizeof(float));

        // ── post_attention_layernorm → MLP
        post_attention_layernorm.forward(normed.data(), hidden_states, seq_len);
        mlp.forward(mixer_out.data(), normed.data(), seq_len, scratch);

        // ── residual + mlp_out → hidden_states
        vec_add(hidden_states, residual.data(), mixer_out.data(),
                static_cast<int>(sz));
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// 3g. TextModel
// ═══════════════════════════════════════════════════════════════════════════

struct Qwen3_5TextModel {
    const uint16_t* embed_tokens = nullptr;  // [vocab_size, hidden_size]
    Qwen3_5RotaryEmbedding rotary_emb;
    std::vector<Qwen3_5DecoderLayer> layers;
    Qwen3_5RMSNorm final_norm;

    Qwen3_5TextConfig config;

    void load(const SafetensorsBundle& bundle, const std::string& prefix,
              const Qwen3_5TextConfig& cfg) {
        config = cfg;
        embed_tokens = bundle.bf16(prefix + ".embed_tokens.weight");

        rotary_emb.init(cfg);

        layers.resize(cfg.num_hidden_layers);
        for (int i = 0; i < cfg.num_hidden_layers; i++) {
            std::string lp = prefix + ".layers." + std::to_string(i);
            layers[i].load(bundle, lp, cfg, i);
        }

        final_norm.load(bundle, prefix + ".norm");
    }

    ModelCache create_cache() const {
        ModelCache cache;
        cache.kv_caches.resize(config.num_hidden_layers);
        cache.gdn_caches.resize(config.num_hidden_layers);
        cache.seq_offset = 0;

        for (int i = 0; i < config.num_hidden_layers; i++) {
            if (config.layer_types[i] == LayerType::FullAttention) {
                cache.kv_caches[i].init(
                    config.num_key_value_heads, config.head_dim);
            } else {
                int key_dim = config.linear_key_head_dim * config.linear_num_key_heads;
                int value_dim = config.linear_value_head_dim * config.linear_num_value_heads;
                int conv_dim = key_dim * 2 + value_dim;

                // Need to find the conv weight for this layer
                std::string conv_name = "model.language_model.layers."
                    + std::to_string(i) + ".linear_attn.conv1d.weight";
                // We need the bundle reference, but don't have it here.
                // The GDN layer has the pointer already.
                cache.gdn_caches[i].init(
                    conv_dim, config.linear_conv_kernel_dim,
                    config.linear_num_value_heads,
                    config.linear_key_head_dim,
                    config.linear_value_head_dim,
                    layers[i].linear_attn->conv1d_w);
            }
        }
        return cache;
    }

    // token_ids: [seq_len]
    // out: [seq_len, hidden_size]
    void forward(float* out, const int* token_ids, int seq_len,
                 ModelCache& cache, Scratch& scratch) const {
        int H = config.hidden_size;

        // ── Embedding lookup
        std::vector<float> hidden(static_cast<size_t>(seq_len) * H);
        embedding_bf16(hidden.data(), embed_tokens, token_ids, seq_len, H);

        // ── Position IDs: [3, seq_len]  (for MRoPE, text-only: all same)
        std::vector<int> position_ids(3 * seq_len);
        for (int d = 0; d < 3; d++)
            for (int s = 0; s < seq_len; s++)
                position_ids[d * seq_len + s] = cache.seq_offset + s;

        // ── RoPE cos/sin tables
        int rot_dim = config.rotary_dim();
        std::vector<float> cos_vals(seq_len * rot_dim);
        std::vector<float> sin_vals(seq_len * rot_dim);
        rotary_emb.forward(cos_vals.data(), sin_vals.data(),
                           position_ids.data(), seq_len);

        // ── Causal mask: [seq_q, seq_kv]  (additive)
        int kv_len = cache.seq_offset + seq_len;
        std::vector<float> mask;
        const float* mask_ptr = nullptr;
        if (seq_len > 1) {
            mask.resize(static_cast<size_t>(seq_len) * kv_len);
            for (int sq = 0; sq < seq_len; sq++) {
                for (int sk = 0; sk < kv_len; sk++) {
                    int q_pos = cache.seq_offset + sq;
                    mask[sq * kv_len + sk] =
                        (sk <= q_pos) ? 0.0f : -1e9f;
                }
            }
            mask_ptr = mask.data();
        }

        // ── Layer loop
        for (int i = 0; i < config.num_hidden_layers; i++) {
            layers[i].forward(hidden.data(), cos_vals.data(), sin_vals.data(),
                              mask_ptr, seq_len, cache, scratch);
        }

        // ── Final norm
        final_norm.forward(out, hidden.data(), seq_len);

        cache.seq_offset += seq_len;
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// 3h. ForCausalLM
// ═══════════════════════════════════════════════════════════════════════════

struct Qwen3_5ForCausalLM {
    Qwen3_5TextModel model;
    const uint16_t* lm_head = nullptr;   // may be shared with embed_tokens
    int vocab_size = 0;
    int hidden_size = 0;
    bool tie_embeddings = false;

    void load(const SafetensorsBundle& bundle, const std::string& prefix,
              const Qwen3_5Config& cfg) {
        model.load(bundle, prefix + ".language_model", cfg.text_config);
        vocab_size = cfg.text_config.vocab_size;
        hidden_size = cfg.text_config.hidden_size;
        tie_embeddings = cfg.tie_word_embeddings;

        if (tie_embeddings) {
            lm_head = model.embed_tokens;   // shared weight
        } else {
            lm_head = bundle.bf16(prefix + ".lm_head.weight");
        }
    }

    // ── Single forward pass (logits for all positions) ──────────────────
    // token_ids: [seq_len]
    // logits_out: [seq_len, vocab_size]  (or just last token if logits_to_keep=1)
    void forward(float* logits_out, const int* token_ids, int seq_len,
                 int logits_to_keep, ModelCache& cache, Scratch& scratch) const {
        // Get hidden states from text model
        std::vector<float> hidden(static_cast<size_t>(seq_len) * hidden_size);
        model.forward(hidden.data(), token_ids, seq_len, cache, scratch);

        // Compute logits for last `logits_to_keep` positions
        int start = seq_len - logits_to_keep;
        int M = logits_to_keep;
        const float* h = hidden.data() + start * hidden_size;

        linear_bf16(logits_out, h, lm_head,
                    M, hidden_size, vocab_size, scratch);
    }

    // ── Greedy generation ───────────────────────────────────────────────
    // prompt_ids: input token IDs
    // max_new_tokens: how many tokens to generate
    // eos_token_id: stop generation on this token
    // Returns: generated token IDs (including prompt)
    std::vector<int> generate(const std::vector<int>& prompt_ids,
                              int max_new_tokens,
                              int eos_token_id = 248049) const {
        Scratch scratch;
        ModelCache cache = model.create_cache();

        std::vector<int> output(prompt_ids);

        // ── Prefill
        std::vector<float> logits(vocab_size);
        forward(logits.data(), prompt_ids.data(),
                static_cast<int>(prompt_ids.size()), 1, cache, scratch);

        // Argmax
        int next_token = static_cast<int>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
        output.push_back(next_token);

        if (next_token == eos_token_id)
            return output;

        // ── Decode loop
        for (int step = 1; step < max_new_tokens; step++) {
            forward(logits.data(), &next_token, 1, 1, cache, scratch);

            next_token = static_cast<int>(
                std::max_element(logits.begin(), logits.end()) - logits.begin());
            output.push_back(next_token);

            if (next_token == eos_token_id)
                break;
        }

        return output;
    }
};
