#pragma once

// Qwen3.5 Vision Model — C++ inference implementation
// Phase 3v: Vision encoder + VLM integration
//
// All computation is f32 on CPU (Accelerate/BLAS).
// Weights stay mmap'd as bf16, converted on-the-fly per operation.

#include "models/qwen3_5/config.h"
#include "models/qwen3_5/modeling.h"
#include "utils/ops.h"
#include "utils/safetensors.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

// ═══════════════════════════════════════════════════════════════════════════
// 3v-a. VisionRotaryEmbedding (2D spatial RoPE)
// ═══════════════════════════════════════════════════════════════════════════

struct Qwen3_5VisionRotaryEmbedding {
    std::vector<float> inv_freq;   // [dim/2]
    int dim = 0;                   // head_dim // 2

    void init(int d, float theta = 10000.0f) {
        dim = d;
        int half = dim / 2;
        inv_freq.resize(half);
        for (int i = 0; i < half; i++) {
            double exp = static_cast<double>(2 * i) / dim;
            inv_freq[i] = static_cast<float>(1.0 / std::pow(theta, exp));
        }
    }

    // Compute freq table: [max_pos, dim/2]
    // freqs[p][f] = p * inv_freq[f]
    std::vector<float> compute_freqs(int max_pos) const {
        int half = dim / 2;
        std::vector<float> freqs(max_pos * half);
        for (int p = 0; p < max_pos; p++)
            for (int f = 0; f < half; f++)
                freqs[p * half + f] = static_cast<float>(p) * inv_freq[f];
        return freqs;
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// 3v-b. VisionPatchEmbed (Conv3d)
// ═══════════════════════════════════════════════════════════════════════════

struct Qwen3_5VisionPatchEmbed {
    const uint16_t* proj_w = nullptr;   // [embed_dim, 3, T, H, W] bf16
    std::vector<float> proj_bias;       // [embed_dim]
    int patch_size = 16;
    int temporal_patch_size = 2;
    int in_channels = 3;
    int embed_dim = 0;

    void load(const SafetensorsBundle& bundle, const std::string& prefix,
              const Qwen3_5VisionConfig& cfg) {
        patch_size = cfg.patch_size;
        temporal_patch_size = cfg.temporal_patch_size;
        in_channels = cfg.in_channels;
        embed_dim = cfg.hidden_size;

        proj_w = bundle.bf16(prefix + ".proj.weight");
        proj_bias = bundle.load_f32(prefix + ".proj.bias");
    }

    // pixel_values: [num_patches, in_channels * temporal_patch_size * patch_size * patch_size]
    // out: [num_patches, embed_dim]
    void forward(float* out, const float* pixel_values, int num_patches,
                 Scratch& scratch) const {
        conv3d_patch_bf16(out, pixel_values, proj_w, proj_bias.data(),
                          num_patches, in_channels,
                          temporal_patch_size, patch_size, patch_size,
                          embed_dim, scratch);
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// 3v-c. VisionAttention (multi-head with cu_seqlens)
// ═══════════════════════════════════════════════════════════════════════════

struct Qwen3_5VisionAttention {
    const uint16_t* qkv_w = nullptr;    // [3*dim, dim]
    std::vector<float> qkv_bias;        // [3*dim]
    const uint16_t* proj_w = nullptr;   // [dim, dim]
    std::vector<float> proj_bias;       // [dim]

    int dim = 0;
    int num_heads = 0;
    int head_dim = 0;
    float scale = 0.0f;

    void load(const SafetensorsBundle& bundle, const std::string& prefix,
              const Qwen3_5VisionConfig& cfg) {
        dim = cfg.hidden_size;
        num_heads = cfg.num_heads;
        head_dim = dim / num_heads;
        scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        qkv_w = bundle.bf16(prefix + ".qkv.weight");
        qkv_bias = bundle.load_f32(prefix + ".qkv.bias");
        proj_w = bundle.bf16(prefix + ".proj.weight");
        proj_bias = bundle.load_f32(prefix + ".proj.bias");
    }

    // hidden_states: [total_tokens, dim]
    // cos/sin: [total_tokens, head_dim]
    // cu_seqlens: [num_seqs+1]  (cumulative lengths)
    // out: [total_tokens, dim]
    void forward(float* out, const float* hidden_states,
                 const float* cos_vals, const float* sin_vals,
                 const int* cu_seqlens, int num_seqs,
                 int total_tokens, Scratch& scratch) const {
        // QKV projection: [total, dim] → [total, 3*dim]
        int qkv_dim = 3 * dim;
        std::vector<float> qkv(static_cast<size_t>(total_tokens) * qkv_dim);
        linear_bf16_bias(qkv.data(), hidden_states, qkv_w, qkv_bias.data(),
                         total_tokens, dim, qkv_dim, scratch);

        // Reshape: [total, 3, num_heads, head_dim] → split Q, K, V
        // q/k/v each: [total, num_heads, head_dim]
        std::vector<float> q(static_cast<size_t>(total_tokens) * dim);
        std::vector<float> k(static_cast<size_t>(total_tokens) * dim);
        std::vector<float> v(static_cast<size_t>(total_tokens) * dim);

        for (int t = 0; t < total_tokens; t++) {
            const float* src = qkv.data() + t * qkv_dim;
            // QKV is laid out as [3, num_heads, head_dim] per token
            for (int h = 0; h < num_heads; h++) {
                std::memcpy(q.data() + (t * num_heads + h) * head_dim,
                            src + (0 * num_heads + h) * head_dim,
                            head_dim * sizeof(float));
                std::memcpy(k.data() + (t * num_heads + h) * head_dim,
                            src + (1 * num_heads + h) * head_dim,
                            head_dim * sizeof(float));
                std::memcpy(v.data() + (t * num_heads + h) * head_dim,
                            src + (2 * num_heads + h) * head_dim,
                            head_dim * sizeof(float));
            }
        }

        // Apply vision RoPE (full rotation, all heads same #)
        apply_rope_vision(q.data(), k.data(), cos_vals, sin_vals,
                          total_tokens, num_heads, head_dim);

        // Per-sequence attention (split by cu_seqlens)
        std::vector<float> attn_out(static_cast<size_t>(total_tokens) * dim, 0.0f);
        for (int s = 0; s < num_seqs; s++) {
            int start = cu_seqlens[s];
            int end = cu_seqlens[s + 1];
            int seq_len = end - start;

            const float* q_s = q.data() + start * dim;
            const float* k_s = k.data() + start * dim;
            const float* v_s = v.data() + start * dim;
            float* o_s = attn_out.data() + start * dim;

            // Non-causal full attention (is_causal=False for vision)
            attention(o_s, q_s, k_s, v_s, nullptr,
                      seq_len, seq_len, num_heads, num_heads, head_dim,
                      scale, scratch);
        }

        // Reshape to [total, dim] and project
        linear_bf16_bias(out, attn_out.data(), proj_w, proj_bias.data(),
                         total_tokens, dim, dim, scratch);
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// 3v-d. VisionMLP (with bias, GELU)
// ═══════════════════════════════════════════════════════════════════════════

struct Qwen3_5VisionMLP {
    const uint16_t* fc1_w = nullptr;    // [intermediate, hidden]
    std::vector<float> fc1_bias;        // [intermediate]
    const uint16_t* fc2_w = nullptr;    // [hidden, intermediate]
    std::vector<float> fc2_bias;        // [hidden]
    int hidden_size = 0;
    int intermediate_size = 0;

    void load(const SafetensorsBundle& bundle, const std::string& prefix,
              const Qwen3_5VisionConfig& cfg) {
        hidden_size = cfg.hidden_size;
        intermediate_size = cfg.intermediate_size;

        fc1_w = bundle.bf16(prefix + ".linear_fc1.weight");
        fc1_bias = bundle.load_f32(prefix + ".linear_fc1.bias");
        fc2_w = bundle.bf16(prefix + ".linear_fc2.weight");
        fc2_bias = bundle.load_f32(prefix + ".linear_fc2.bias");
    }

    // x: [tokens, hidden], out: [tokens, hidden]
    void forward(float* out, const float* x, int tokens,
                 Scratch& scratch) const {
        int M = tokens;
        std::vector<float> mid(static_cast<size_t>(M) * intermediate_size);
        linear_bf16_bias(mid.data(), x, fc1_w, fc1_bias.data(),
                         M, hidden_size, intermediate_size, scratch);
        gelu_tanh_inplace(mid.data(), M * intermediate_size);
        linear_bf16_bias(out, mid.data(), fc2_w, fc2_bias.data(),
                         M, intermediate_size, hidden_size, scratch);
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// 3v-e. VisionBlock (LayerNorm → Attn → residual → LayerNorm → MLP → residual)
// ═══════════════════════════════════════════════════════════════════════════

struct VisionLayerNorm {
    std::vector<float> weight;   // [dim]
    std::vector<float> bias;     // [dim]
    int dim = 0;
    float eps = 1e-6f;

    void load(const SafetensorsBundle& bundle, const std::string& prefix) {
        weight = bundle.load_f32(prefix + ".weight");
        bias = bundle.load_f32(prefix + ".bias");
        dim = static_cast<int>(weight.size());
    }

    void forward(float* out, const float* x, int tokens) const {
        for (int t = 0; t < tokens; t++)
            layernorm(out + t * dim, x + t * dim,
                      weight.data(), bias.data(), dim, eps);
    }
};

struct Qwen3_5VisionBlock {
    VisionLayerNorm norm1;
    VisionLayerNorm norm2;
    Qwen3_5VisionAttention attn;
    Qwen3_5VisionMLP mlp;
    int dim = 0;

    void load(const SafetensorsBundle& bundle, const std::string& prefix,
              const Qwen3_5VisionConfig& cfg) {
        dim = cfg.hidden_size;
        norm1.load(bundle, prefix + ".norm1");
        norm2.load(bundle, prefix + ".norm2");
        attn.load(bundle, prefix + ".attn", cfg);
        mlp.load(bundle, prefix + ".mlp", cfg);
    }

    // hidden_states: [total_tokens, dim]  (in-place residual)
    void forward(float* hidden_states,
                 const float* cos_vals, const float* sin_vals,
                 const int* cu_seqlens, int num_seqs,
                 int total_tokens, Scratch& scratch) const {
        size_t sz = static_cast<size_t>(total_tokens) * dim;

        // Norm1 → Attention → residual
        std::vector<float> normed(sz);
        norm1.forward(normed.data(), hidden_states, total_tokens);

        std::vector<float> attn_out(sz);
        attn.forward(attn_out.data(), normed.data(), cos_vals, sin_vals,
                     cu_seqlens, num_seqs, total_tokens, scratch);

        vec_add(hidden_states, hidden_states, attn_out.data(),
                static_cast<int>(sz));

        // Norm2 → MLP → residual
        norm2.forward(normed.data(), hidden_states, total_tokens);
        std::vector<float> mlp_out(sz);
        mlp.forward(mlp_out.data(), normed.data(), total_tokens, scratch);

        vec_add(hidden_states, hidden_states, mlp_out.data(),
                static_cast<int>(sz));
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// 3v-f. VisionPatchMerger
// ═══════════════════════════════════════════════════════════════════════════

struct Qwen3_5VisionPatchMerger {
    VisionLayerNorm norm;
    const uint16_t* fc1_w = nullptr;    // [merged_dim, merged_dim]
    std::vector<float> fc1_bias;
    const uint16_t* fc2_w = nullptr;    // [out_hidden, merged_dim]
    std::vector<float> fc2_bias;

    int hidden_size = 0;
    int merged_dim = 0;        // hidden_size * spatial_merge_size^2
    int out_hidden_size = 0;
    int spatial_merge_size = 0;

    void load(const SafetensorsBundle& bundle, const std::string& prefix,
              const Qwen3_5VisionConfig& cfg) {
        hidden_size = cfg.hidden_size;
        spatial_merge_size = cfg.spatial_merge_size;
        merged_dim = hidden_size * spatial_merge_size * spatial_merge_size;
        out_hidden_size = cfg.out_hidden_size;

        norm.load(bundle, prefix + ".norm");
        fc1_w = bundle.bf16(prefix + ".linear_fc1.weight");
        fc1_bias = bundle.load_f32(prefix + ".linear_fc1.bias");
        fc2_w = bundle.bf16(prefix + ".linear_fc2.weight");
        fc2_bias = bundle.load_f32(prefix + ".linear_fc2.bias");
    }

    // x: [total_tokens, hidden_size]  (pre-merge, spatial_merge_size^2 consecutive
    //     tokens form one merged token)
    // out: [total_tokens / merge^2, out_hidden_size]
    void forward(float* out, const float* x, int total_tokens,
                 Scratch& scratch) const {
        // Norm first (per-token, hidden_size dim)
        std::vector<float> normed(static_cast<size_t>(total_tokens) * hidden_size);
        norm.forward(normed.data(), x, total_tokens);

        // Concatenate spatial_merge_size^2 tokens → merged_dim
        // After spatial permutation, consecutive merge^2 tokens are already
        // adjacent in memory (done in VisionModel::forward)
        int num_merged = total_tokens / (spatial_merge_size * spatial_merge_size);

        // View normed as [num_merged, merged_dim]
        // The spatial permutation in VisionModel ensures tokens are in merge order

        // FC1 → GELU → FC2
        std::vector<float> mid(static_cast<size_t>(num_merged) * merged_dim);
        linear_bf16_bias(mid.data(), normed.data(), fc1_w, fc1_bias.data(),
                         num_merged, merged_dim, merged_dim, scratch);
        gelu_tanh_inplace(mid.data(), num_merged * merged_dim);
        linear_bf16_bias(out, mid.data(), fc2_w, fc2_bias.data(),
                         num_merged, merged_dim, out_hidden_size, scratch);
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// 3v-g. VisionModel
// ═══════════════════════════════════════════════════════════════════════════

struct Qwen3_5VisionModel {
    Qwen3_5VisionPatchEmbed patch_embed;
    std::vector<float> pos_embed_weight;   // [num_position_embeddings, hidden_size]
    int num_grid_per_side = 0;             // sqrt(num_position_embeddings)
    Qwen3_5VisionRotaryEmbedding rotary_pos_emb;
    std::vector<Qwen3_5VisionBlock> blocks;
    Qwen3_5VisionPatchMerger merger;

    Qwen3_5VisionConfig config;

    void load(const SafetensorsBundle& bundle, const std::string& prefix,
              const Qwen3_5VisionConfig& cfg) {
        config = cfg;
        patch_embed.load(bundle, prefix + ".patch_embed", cfg);

        pos_embed_weight = bundle.load_f32(prefix + ".pos_embed.weight");
        num_grid_per_side = static_cast<int>(
            std::sqrt(static_cast<double>(cfg.num_position_embeddings)));

        int head_dim = cfg.hidden_size / cfg.num_heads;
        rotary_pos_emb.init(head_dim / 2);

        blocks.resize(cfg.depth);
        for (int i = 0; i < cfg.depth; i++) {
            std::string bp = prefix + ".blocks." + std::to_string(i);
            blocks[i].load(bundle, bp, cfg);
        }

        merger.load(bundle, prefix + ".merger", cfg);
    }

    // ── Compute 2D rotary position embeddings from grid ─────────────────
    // grid_thw: array of (T, H, W) per image/video
    // Returns: [total_tokens, head_dim]  (cos/sin-ready freq values)
    std::vector<float> rot_pos_emb(
        const std::vector<std::array<int, 3>>& grid_thw) const
    {
        int merge = config.spatial_merge_size;
        int half = rotary_pos_emb.dim / 2;   // dim/2 = head_dim/4
        int head_dim_half = rotary_pos_emb.dim;   // head_dim/2

        // Find max spatial dimension for freq table
        int max_hw = 0;
        for (auto& g : grid_thw)
            max_hw = std::max(max_hw, std::max(g[1], g[2]));
        auto freq_table = rotary_pos_emb.compute_freqs(max_hw);

        // Compute total tokens
        int total = 0;
        for (auto& g : grid_thw)
            total += g[0] * g[1] * g[2];

        // Build 2D position indices → lookup freqs → concat (row_freq, col_freq)
        std::vector<float> emb(static_cast<size_t>(total) * head_dim_half * 2);
        int offset = 0;

        for (auto& g : grid_thw) {
            int T = g[0], H = g[1], W = g[2];
            int mH = H / merge, mW = W / merge;

            for (int t = 0; t < T; t++) {
                // Generate positions in merge-block order:
                // For each merge block (bh, bw), iterate intra-block (ih, iw)
                for (int bh = 0; bh < mH; bh++) {
                    for (int bw = 0; bw < mW; bw++) {
                        for (int ih = 0; ih < merge; ih++) {
                            for (int iw = 0; iw < merge; iw++) {
                                int row = bh * merge + ih;
                                int col = bw * merge + iw;

                                float* dst = emb.data()
                                    + offset * head_dim_half * 2;

                                // row freqs: freq_table[row, :]
                                const float* rf = freq_table.data()
                                    + row * half;
                                std::memcpy(dst, rf, half * sizeof(float));

                                // col freqs: freq_table[col, :]
                                const float* cf = freq_table.data()
                                    + col * half;
                                std::memcpy(dst + half, cf, half * sizeof(float));

                                offset++;
                            }
                        }
                    }
                }
            }
        }

        return emb;   // [total_tokens, head_dim_half * 2] = [total, head_dim]
    }

    // ── Bilinear interpolation of position embeddings ───────────────────
    // Returns: [total_tokens, hidden_size]
    std::vector<float> fast_pos_embed_interpolate(
        const std::vector<std::array<int, 3>>& grid_thw) const
    {
        int H_dim = config.hidden_size;
        int G = num_grid_per_side;
        int merge = config.spatial_merge_size;

        // Count total tokens (pre-merge)
        int total = 0;
        for (auto& g : grid_thw) total += g[0] * g[1] * g[2];

        std::vector<float> result(static_cast<size_t>(total) * H_dim, 0.0f);
        int out_offset = 0;

        for (auto& g : grid_thw) {
            int T = g[0], H = g[1], W = g[2];
            int hw_tokens = H * W;

            // Map grid positions to [0, G-1]
            std::vector<float> h_idxs(H), w_idxs(W);
            for (int i = 0; i < H; i++)
                h_idxs[i] = (H > 1) ? static_cast<float>(i) * (G - 1) / (H - 1) : 0.0f;
            for (int i = 0; i < W; i++)
                w_idxs[i] = (W > 1) ? static_cast<float>(i) * (G - 1) / (W - 1) : 0.0f;

            // Bilinear interpolation for single frame
            std::vector<float> frame_pos(hw_tokens * H_dim, 0.0f);
            for (int hi = 0; hi < H; hi++) {
                float hf = h_idxs[hi];
                int h0 = std::min(static_cast<int>(hf), G - 1);
                int h1 = std::min(h0 + 1, G - 1);
                float dh = hf - h0;

                for (int wi = 0; wi < W; wi++) {
                    float wf = w_idxs[wi];
                    int w0 = std::min(static_cast<int>(wf), G - 1);
                    int w1 = std::min(w0 + 1, G - 1);
                    float dw = wf - w0;

                    // 4 corner indices into pos_embed
                    int i00 = h0 * G + w0;
                    int i01 = h0 * G + w1;
                    int i10 = h1 * G + w0;
                    int i11 = h1 * G + w1;

                    // Weights
                    float w00 = (1.0f - dh) * (1.0f - dw);
                    float w01 = (1.0f - dh) * dw;
                    float w10 = dh * (1.0f - dw);
                    float w11 = dh * dw;

                    float* dst = frame_pos.data()
                        + (hi * W + wi) * H_dim;
                    const float* e00 = pos_embed_weight.data() + i00 * H_dim;
                    const float* e01 = pos_embed_weight.data() + i01 * H_dim;
                    const float* e10 = pos_embed_weight.data() + i10 * H_dim;
                    const float* e11 = pos_embed_weight.data() + i11 * H_dim;

                    for (int d = 0; d < H_dim; d++)
                        dst[d] = w00 * e00[d] + w01 * e01[d]
                               + w10 * e10[d] + w11 * e11[d];
                }
            }

            // Replicate across T frames, then permute for spatial merge order
            // Permutation: (T, H//m, m, W//m, m, D) → (T, H//m, W//m, m, m, D)
            int mH = H / merge, mW = W / merge;
            for (int t = 0; t < T; t++) {
                for (int bh = 0; bh < mH; bh++) {
                    for (int bw = 0; bw < mW; bw++) {
                        for (int ih = 0; ih < merge; ih++) {
                            for (int iw = 0; iw < merge; iw++) {
                                int row = bh * merge + ih;
                                int col = bw * merge + iw;
                                int src_idx = row * W + col;
                                float* dst = result.data()
                                    + out_offset * H_dim;
                                std::memcpy(dst,
                                            frame_pos.data() + src_idx * H_dim,
                                            H_dim * sizeof(float));
                                out_offset++;
                            }
                        }
                    }
                }
            }
        }

        return result;
    }

    // ── Spatial merge permutation ───────────────────────────────────────
    // Reorder tokens from raster order to merge-block order
    // Input:  [T*H*W, dim]  in raster order
    // Output: [T*H*W, dim]  in (T, H//m, W//m, m, m) order
    // This is needed for patch merging: consecutive merge^2 tokens
    // become one merged token.
    void permute_for_merge(float* out, const float* in,
                           const std::vector<std::array<int, 3>>& grid_thw) const {
        int merge = config.spatial_merge_size;
        int D = config.hidden_size;
        int in_off = 0, out_off = 0;

        for (auto& g : grid_thw) {
            int T = g[0], H = g[1], W = g[2];
            int mH = H / merge, mW = W / merge;

            for (int t = 0; t < T; t++) {
                for (int bh = 0; bh < mH; bh++) {
                    for (int bw = 0; bw < mW; bw++) {
                        for (int ih = 0; ih < merge; ih++) {
                            for (int iw = 0; iw < merge; iw++) {
                                int row = bh * merge + ih;
                                int col = bw * merge + iw;
                                int src = in_off + (t * H * W + row * W + col) * D;
                                std::memcpy(out + out_off * D,
                                            in + src,
                                            D * sizeof(float));
                                out_off++;
                            }
                        }
                    }
                }
            }
            in_off += T * H * W * D;
        }
    }

    // ── Forward: pixels → vision features ───────────────────────────────
    // pixel_values: [total_patch_pixels]  (flattened patch pixel values)
    // grid_thw: per-image/video (T, H, W) after patch extraction
    // merged_out: [total_merged_tokens, out_hidden_size]
    // Returns total merged tokens
    int forward(float* merged_out,
                const float* pixel_values,
                const std::vector<std::array<int, 3>>& grid_thw,
                Scratch& scratch) const {
        int D = config.hidden_size;
        int merge = config.spatial_merge_size;

        // Count tokens
        int total_tokens = 0;
        for (auto& g : grid_thw)
            total_tokens += g[0] * g[1] * g[2];
        int total_merged = total_tokens / (merge * merge);

        // 1. Patch embedding (Conv3d)
        std::vector<float> hidden(static_cast<size_t>(total_tokens) * D);
        patch_embed.forward(hidden.data(), pixel_values, total_tokens, scratch);

        // 2. Position embeddings (bilinear interpolation)
        auto pos_embeds = fast_pos_embed_interpolate(grid_thw);
        // pos_embeds are already in merge-block order from fast_pos_embed_interpolate
        // But patch_embed output is in raster order, so we need to reorder

        // Permute hidden to merge-block order
        std::vector<float> hidden_perm(static_cast<size_t>(total_tokens) * D);
        permute_for_merge(hidden_perm.data(), hidden.data(), grid_thw);

        // Add position embeddings
        vec_add(hidden_perm.data(), hidden_perm.data(), pos_embeds.data(),
                total_tokens * D);

        // 3. Rotary embeddings
        auto rope_freqs = rot_pos_emb(grid_thw);
        // rope_freqs: [total_tokens, head_dim] — already in merge-block order

        // Build cos/sin: emb = cat(freqs, freqs), cos = cos(emb), sin = sin(emb)
        int head_dim = D / config.num_heads;
        std::vector<float> cos_vals(static_cast<size_t>(total_tokens) * head_dim);
        std::vector<float> sin_vals(static_cast<size_t>(total_tokens) * head_dim);
        for (int t = 0; t < total_tokens; t++) {
            const float* freq = rope_freqs.data() + t * head_dim;
            for (int d = 0; d < head_dim; d++) {
                cos_vals[t * head_dim + d] = std::cos(freq[d]);
                sin_vals[t * head_dim + d] = std::sin(freq[d]);
            }
        }

        // 4. cu_seqlens for variable-length attention
        // Each temporal frame of each image/video is a separate sequence
        std::vector<int> cu_seqlens;
        cu_seqlens.push_back(0);
        for (auto& g : grid_thw) {
            int T = g[0], H = g[1], W = g[2];
            int frame_tokens = H * W;
            for (int t = 0; t < T; t++)
                cu_seqlens.push_back(cu_seqlens.back() + frame_tokens);
        }
        int num_seqs = static_cast<int>(cu_seqlens.size()) - 1;

        // 5. Transformer blocks
        for (int i = 0; i < config.depth; i++) {
            blocks[i].forward(hidden_perm.data(),
                              cos_vals.data(), sin_vals.data(),
                              cu_seqlens.data(), num_seqs,
                              total_tokens, scratch);
        }

        // 6. Patch merger
        merger.forward(merged_out, hidden_perm.data(), total_tokens, scratch);

        return total_merged;
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// 3v-h. Qwen3_5VLModel (Vision + Language integration)
// ═══════════════════════════════════════════════════════════════════════════

struct Qwen3_5VLModel {
    Qwen3_5VisionModel visual;
    Qwen3_5TextModel language_model;
    Qwen3_5Config config;

    void load(const SafetensorsBundle& bundle, const std::string& prefix,
              const Qwen3_5Config& cfg) {
        config = cfg;
        visual.load(bundle, prefix + ".visual", cfg.vision_config);
        language_model.load(bundle, prefix + ".language_model", cfg.text_config);
    }

    // ── Get vision features for images ──────────────────────────────────
    // pixel_values: flattened patch pixels
    // grid_thw: per-image (T, H, W)
    // Returns: [total_merged_tokens, text_hidden_size]
    std::vector<float> get_image_features(
        const float* pixel_values,
        const std::vector<std::array<int, 3>>& grid_thw,
        Scratch& scratch) const
    {
        int merge = config.vision_config.spatial_merge_size;
        int total_tokens = 0;
        for (auto& g : grid_thw) total_tokens += g[0] * g[1] * g[2];
        int total_merged = total_tokens / (merge * merge);

        int out_dim = config.vision_config.out_hidden_size;
        std::vector<float> features(static_cast<size_t>(total_merged) * out_dim);
        visual.forward(features.data(), pixel_values, grid_thw, scratch);
        return features;
    }

    // ── Compute 3D MRoPE position IDs with vision tokens ────────────────
    // input_ids: [seq_len]
    // mm_types: [seq_len]  (0=text, 1=image, 2=video)
    // image_grid_thw / video_grid_thw: per-image/video grid info
    // position_ids_out: [3, seq_len]
    // Returns rope_delta (offset for incremental generation)
    int compute_3d_position_ids(
        int* position_ids_out,
        const int* /*input_ids*/, const int* mm_types, int seq_len,
        const std::vector<std::array<int, 3>>& image_grid_thw,
        const std::vector<std::array<int, 3>>& video_grid_thw) const
    {
        int spatial_merge = config.vision_config.spatial_merge_size;
        int img_idx = 0, vid_idx = 0;

        // Group consecutive tokens by type
        struct Group { int type; int start; int end; };
        std::vector<Group> groups;
        if (seq_len > 0) {
            int cur_type = mm_types[0];
            int start = 0;
            for (int i = 1; i <= seq_len; i++) {
                if (i == seq_len || mm_types[i] != cur_type) {
                    groups.push_back({cur_type, start, i});
                    if (i < seq_len) {
                        cur_type = mm_types[i];
                        start = i;
                    }
                }
            }
        }

        int current_pos = 0;
        int max_pos = 0;

        for (auto& grp : groups) {
            int len = grp.end - grp.start;

            if (grp.type == 0) {
                // Text: same position for all 3 dims
                for (int i = 0; i < len; i++) {
                    int pos = current_pos + i;
                    position_ids_out[0 * seq_len + grp.start + i] = pos;
                    position_ids_out[1 * seq_len + grp.start + i] = pos;
                    position_ids_out[2 * seq_len + grp.start + i] = pos;
                    max_pos = std::max(max_pos, pos);
                }
                current_pos += len;
            } else {
                // Vision token: get grid info
                std::array<int, 3> grid;
                if (grp.type == 1 && img_idx < static_cast<int>(image_grid_thw.size())) {
                    grid = image_grid_thw[img_idx++];
                } else if (grp.type == 2 && vid_idx < static_cast<int>(video_grid_thw.size())) {
                    grid = video_grid_thw[vid_idx++];
                } else {
                    grid = {1, 1, 1};
                }

                int llm_h = grid[1] / spatial_merge;
                int llm_w = grid[2] / spatial_merge;
                int llm_t = grid[0];

                // Fill 3D positions
                int idx = 0;
                for (int t = 0; t < llm_t; t++) {
                    for (int h = 0; h < llm_h; h++) {
                        for (int w = 0; w < llm_w; w++) {
                            if (grp.start + idx < seq_len) {
                                position_ids_out[0 * seq_len + grp.start + idx] = current_pos;  // temporal
                                position_ids_out[1 * seq_len + grp.start + idx] = current_pos + h;  // height
                                position_ids_out[2 * seq_len + grp.start + idx] = current_pos + w;  // width
                                max_pos = std::max(max_pos,
                                    std::max(current_pos,
                                    std::max(current_pos + h, current_pos + w)));
                                idx++;
                            }
                        }
                    }
                }

                current_pos += std::max(llm_h, llm_w);
            }
        }

        // rope_delta: max_pos + 1 - seq_len
        return max_pos + 1 - seq_len;
    }

    // ── Forward: multimodal input → hidden states ───────────────────────
    // input_ids: [seq_len]
    // mm_types: [seq_len]  (0=text, 1=image, 2=video, or nullptr for text-only)
    // pixel_values: flattened image pixels (or nullptr)
    // grid_thw: per-image grid info
    // out: [seq_len, text_hidden_size]
    void forward(float* out, const int* input_ids, int seq_len,
                 const int* mm_types,
                 const float* pixel_values,
                 const std::vector<std::array<int, 3>>& image_grid_thw,
                 const std::vector<std::array<int, 3>>& video_grid_thw,
                 ModelCache& cache, Scratch& scratch) const {
        int H = config.text_config.hidden_size;

        // 1. Embed text tokens
        std::vector<float> embeds(static_cast<size_t>(seq_len) * H);
        std::vector<int> token_ids(input_ids, input_ids + seq_len);
        embedding_bf16(embeds.data(), language_model.embed_tokens,
                       token_ids.data(), seq_len, H);

        // 2. Process images and inject features
        if (pixel_values && !image_grid_thw.empty()) {
            auto image_features = get_image_features(
                pixel_values, image_grid_thw, scratch);
            int out_dim = config.vision_config.out_hidden_size;

            // Replace image placeholder tokens with vision features
            int feat_idx = 0;
            for (int i = 0; i < seq_len; i++) {
                if (input_ids[i] == config.image_token_id) {
                    if (feat_idx * out_dim < static_cast<int>(image_features.size())) {
                        std::memcpy(embeds.data() + i * H,
                                    image_features.data() + feat_idx * out_dim,
                                    H * sizeof(float));
                        feat_idx++;
                    }
                }
            }
        }

        // 3. Compute 3D position IDs
        // For text-only, use simple ascending position IDs
        std::vector<int> position_ids(3 * seq_len);
        if (mm_types) {
            compute_3d_position_ids(position_ids.data(),
                                    input_ids, mm_types, seq_len,
                                    image_grid_thw, video_grid_thw);
        } else {
            for (int d = 0; d < 3; d++)
                for (int s = 0; s < seq_len; s++)
                    position_ids[d * seq_len + s] = cache.seq_offset + s;
        }

        // 4. RoPE cos/sin
        int rot_dim = config.text_config.rotary_dim();
        std::vector<float> cos_vals(seq_len * rot_dim);
        std::vector<float> sin_vals(seq_len * rot_dim);
        language_model.rotary_emb.forward(
            cos_vals.data(), sin_vals.data(),
            position_ids.data(), seq_len);

        // 5. Causal mask
        int kv_len = cache.seq_offset + seq_len;
        std::vector<float> mask;
        const float* mask_ptr = nullptr;
        if (seq_len > 1) {
            mask.resize(static_cast<size_t>(seq_len) * kv_len);
            for (int sq = 0; sq < seq_len; sq++)
                for (int sk = 0; sk < kv_len; sk++)
                    mask[sq * kv_len + sk] =
                        (sk <= cache.seq_offset + sq) ? 0.0f : -1e9f;
            mask_ptr = mask.data();
        }

        // 6. Layer loop (reuse text model layers)
        for (int i = 0; i < config.text_config.num_hidden_layers; i++) {
            language_model.layers[i].forward(
                embeds.data(), cos_vals.data(), sin_vals.data(),
                mask_ptr, seq_len, cache, scratch);
        }

        // 7. Final norm
        language_model.final_norm.forward(out, embeds.data(), seq_len);
        cache.seq_offset += seq_len;
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// 3v-i. ForConditionalGeneration (VLM)
// ═══════════════════════════════════════════════════════════════════════════

struct Qwen3_5ForConditionalGeneration {
    Qwen3_5VLModel model;
    const uint16_t* lm_head = nullptr;
    int vocab_size = 0;
    int hidden_size = 0;
    bool tie_embeddings = false;

    void load(const SafetensorsBundle& bundle, const std::string& prefix,
              const Qwen3_5Config& cfg) {
        model.load(bundle, prefix, cfg);
        vocab_size = cfg.text_config.vocab_size;
        hidden_size = cfg.text_config.hidden_size;
        tie_embeddings = cfg.tie_word_embeddings;

        if (tie_embeddings) {
            lm_head = model.language_model.embed_tokens;
        } else {
            lm_head = bundle.bf16(prefix + ".lm_head.weight");
        }
    }

    ModelCache create_cache() const {
        return model.language_model.create_cache();
    }

    // ── Forward (first step with vision, subsequent decode text-only) ───
    void forward(float* logits_out,
                 const int* input_ids, int seq_len,
                 int logits_to_keep,
                 const int* mm_types,
                 const float* pixel_values,
                 const std::vector<std::array<int, 3>>& image_grid_thw,
                 const std::vector<std::array<int, 3>>& video_grid_thw,
                 ModelCache& cache, Scratch& scratch) const {
        std::vector<float> hidden(static_cast<size_t>(seq_len) * hidden_size);
        model.forward(hidden.data(), input_ids, seq_len, mm_types,
                      pixel_values, image_grid_thw, video_grid_thw,
                      cache, scratch);

        int start = seq_len - logits_to_keep;
        int M = logits_to_keep;
        const float* h = hidden.data() + start * hidden_size;
        linear_bf16(logits_out, h, lm_head, M, hidden_size, vocab_size, scratch);
    }

    // ── Text-only decode step (no vision) ───────────────────────────────
    void forward_decode(float* logits_out, const int* token_id,
                        ModelCache& cache, Scratch& scratch) const {
        std::vector<float> hidden(hidden_size);
        model.forward(hidden.data(), token_id, 1, nullptr,
                      nullptr, {}, {}, cache, scratch);
        linear_bf16(logits_out, hidden.data(), lm_head,
                    1, hidden_size, vocab_size, scratch);
    }

    // ── Greedy generate with optional vision ────────────────────────────
    std::vector<int> generate(
        const std::vector<int>& input_ids,
        int max_new_tokens,
        const std::vector<int>* mm_types = nullptr,
        const float* pixel_values = nullptr,
        const std::vector<std::array<int, 3>>& image_grid_thw = {},
        const std::vector<std::array<int, 3>>& video_grid_thw = {},
        int eos_token_id = 248049) const
    {
        Scratch scratch;
        auto cache = create_cache();
        std::vector<int> output(input_ids);

        // Prefill (with vision if provided)
        std::vector<float> logits(vocab_size);
        int seq_len = static_cast<int>(input_ids.size());
        forward(logits.data(), input_ids.data(), seq_len, 1,
                mm_types ? mm_types->data() : nullptr,
                pixel_values, image_grid_thw, video_grid_thw,
                cache, scratch);

        int next_token = static_cast<int>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
        output.push_back(next_token);
        if (next_token == eos_token_id) return output;

        // Decode loop (text-only)
        for (int step = 1; step < max_new_tokens; step++) {
            forward_decode(logits.data(), &next_token, cache, scratch);
            next_token = static_cast<int>(
                std::max_element(logits.begin(), logits.end()) - logits.begin());
            output.push_back(next_token);
            if (next_token == eos_token_id) break;
        }

        return output;
    }
};
