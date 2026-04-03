#pragma once

#include "utils/safetensors.h"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

// ── Tensor (owns f32 data) ──────────────────────────────────────────────────

struct Tensor {
    std::vector<int> shape;
    std::vector<float> data;

    int ndim() const { return static_cast<int>(shape.size()); }

    int size(int dim) const {
        if (dim < 0) dim += ndim();
        return shape[dim];
    }

    int numel() const {
        int n = 1;
        for (int s : shape) n *= s;
        return n;
    }

    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }

    static Tensor zeros(std::vector<int> sh) {
        Tensor t;
        t.shape = std::move(sh);
        t.data.resize(static_cast<size_t>(t.numel()), 0.0f);
        return t;
    }

    static Tensor empty(std::vector<int> sh) {
        Tensor t;
        t.shape = std::move(sh);
        t.data.resize(static_cast<size_t>(t.numel()));
        return t;
    }
};

// ── Scratch buffer (thread-local reusable workspace) ────────────────────────

struct Scratch {
    std::vector<float> buf;

    float* get(size_t n) {
        if (buf.size() < n) buf.resize(n);
        return buf.data();
    }
};

inline Scratch& global_scratch() {
    static Scratch s;
    return s;
}

// ── Element-wise operations ─────────────────────────────────────────────────

inline void vec_add(float* out, const float* a, const float* b, int n) {
    vDSP_vadd(a, 1, b, 1, out, 1, static_cast<vDSP_Length>(n));
}

inline void vec_mul(float* out, const float* a, const float* b, int n) {
    vDSP_vmul(a, 1, b, 1, out, 1, static_cast<vDSP_Length>(n));
}

inline void vec_scale(float* out, const float* a, float s, int n) {
    vDSP_vsmul(a, 1, &s, out, 1, static_cast<vDSP_Length>(n));
}

inline void silu_inplace(float* x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = x[i] / (1.0f + std::exp(-x[i]));
}

inline void sigmoid_inplace(float* x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = 1.0f / (1.0f + std::exp(-x[i]));
}

inline float softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return std::exp(x);
    return std::log1p(std::exp(x));
}

// ── Softmax (in-place, over last n elements) ────────────────────────────────

inline void softmax_inplace(float* x, int n) {
    float maxv = *std::max_element(x, x + n);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = std::exp(x[i] - maxv);
        sum += x[i];
    }
    float inv = 1.0f / sum;
    vDSP_vsmul(x, 1, &inv, x, 1, static_cast<vDSP_Length>(n));
}

// ── L2 norm ─────────────────────────────────────────────────────────────────

inline void l2norm(float* out, const float* x, int n, float eps = 1e-6f) {
    float sum_sq = 0.0f;
    vDSP_dotpr(x, 1, x, 1, &sum_sq, static_cast<vDSP_Length>(n));
    float inv = 1.0f / std::sqrt(sum_sq + eps);
    vDSP_vsmul(x, 1, &inv, out, 1, static_cast<vDSP_Length>(n));
}

// ── RMSNorm: out = x * rsqrt(mean(x²) + eps) * (1 + weight) ───────────────

inline void rmsnorm(float* out, const float* x, const float* weight,
                    int size, float eps) {
    float sum_sq = 0.0f;
    vDSP_dotpr(x, 1, x, 1, &sum_sq, static_cast<vDSP_Length>(size));
    float scale = 1.0f / std::sqrt(sum_sq / size + eps);
    for (int i = 0; i < size; i++)
        out[i] = x[i] * scale * (1.0f + weight[i]);
}

// ── RMSNormGated: weight * (x * rsqrt(var+eps)) * silu(gate) ───────────────

inline void rmsnorm_gated(float* out, const float* x, const float* gate,
                          const float* weight, int size, float eps) {
    float sum_sq = 0.0f;
    vDSP_dotpr(x, 1, x, 1, &sum_sq, static_cast<vDSP_Length>(size));
    float scale = 1.0f / std::sqrt(sum_sq / size + eps);
    for (int i = 0; i < size; i++) {
        float g = gate[i] / (1.0f + std::exp(-gate[i]));   // silu(gate)
        out[i] = weight[i] * (x[i] * scale) * g;
    }
}

// ── Linear (y[M,N] = x[M,K] @ w[N,K]^T)  ──────────────────────────────────
// Weight stored as bf16 in safetensors (row-major [N, K]).
// Converts to f32 in scratch, then calls BLAS.

inline void linear_bf16(float* y, const float* x,
                        const uint16_t* w_bf16, int M, int K, int N,
                        Scratch& scratch) {
    // Tile conversion to limit scratch usage
    constexpr size_t MAX_TILE_FLOATS = 64 * 1024 * 1024;   // 256 MB
    size_t total = static_cast<size_t>(N) * K;

    if (total <= MAX_TILE_FLOATS) {
        // Convert all at once
        float* w = scratch.get(total);
        bf16_to_f32_array(w, w_bf16, total);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    M, N, K, 1.0f, x, K, w, K, 0.0f, y, N);
    } else {
        // Tiled: process TILE rows of w at a time
        int tile_n = static_cast<int>(MAX_TILE_FLOATS / K);
        if (tile_n < 1) tile_n = 1;
        float* w_tile = scratch.get(static_cast<size_t>(tile_n) * K
                                    + static_cast<size_t>(M) * tile_n);
        float* y_tile = w_tile + static_cast<size_t>(tile_n) * K;

        for (int n0 = 0; n0 < N; n0 += tile_n) {
            int tn = std::min(tile_n, N - n0);
            bf16_to_f32_array(w_tile, w_bf16 + n0 * K,
                              static_cast<size_t>(tn) * K);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, tn, K, 1.0f, x, K, w_tile, K, 0.0f, y_tile, tn);
            // Scatter tile columns into full output
            for (int m = 0; m < M; m++)
                std::memcpy(y + m * N + n0, y_tile + m * tn,
                            static_cast<size_t>(tn) * sizeof(float));
        }
    }
}

// Linear with f32 weights (for small norms, etc.)
inline void linear_f32(float* y, const float* x,
                       const float* w, int M, int K, int N) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0f, x, K, w, K, 0.0f, y, N);
}

// ── Linear + bias (bf16 weight, f32 bias) ───────────────────────────────────

inline void linear_bf16_bias(float* y, const float* x,
                             const uint16_t* w_bf16, const float* bias,
                             int M, int K, int N, Scratch& scratch) {
    linear_bf16(y, x, w_bf16, M, K, N, scratch);
    if (bias) {
        for (int m = 0; m < M; m++)
            vec_add(y + m * N, y + m * N, bias, N);
    }
}

// ── LayerNorm: out = (x - mean) / sqrt(var + eps) * weight + bias ───────────

inline void layernorm(float* out, const float* x,
                      const float* weight, const float* bias,
                      int size, float eps) {
    float mean = 0.0f;
    vDSP_meanv(x, 1, &mean, static_cast<vDSP_Length>(size));

    float var = 0.0f;
    for (int i = 0; i < size; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= size;

    float inv = 1.0f / std::sqrt(var + eps);
    for (int i = 0; i < size; i++)
        out[i] = (x[i] - mean) * inv * weight[i] + bias[i];
}

// ── GELU (PyTorch tanh approximation) ───────────────────────────────────────
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

inline void gelu_tanh_inplace(float* x, int n) {
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float inner = SQRT_2_OVER_PI * (v + 0.044715f * v * v * v);
        x[i] = 0.5f * v * (1.0f + std::tanh(inner));
    }
}

// ── Conv3d (non-overlapping patch extraction) ───────────────────────────────
// Input: [N, C_in, D, H, W]  (N patches, each C_in channels, D×H×W spatial)
// Weight: bf16 [C_out, C_in, kD, kH, kW]
// Bias: f32 [C_out]
// Stride = kernel_size (non-overlapping)
// Output: [N, C_out]  (one output per patch since stride==kernel)
//
// For VisionPatchEmbed: each input is already reshaped to
//   [num_patches, 3, temporal_patch_size, patch_size, patch_size]
// so output is just one value per output channel per patch.

inline void conv3d_patch_bf16(float* out, const float* input,
                              const uint16_t* w_bf16, const float* bias,
                              int num_patches, int C_in, int kD, int kH, int kW,
                              int C_out, Scratch& scratch) {
    int kernel_vol = C_in * kD * kH * kW;   // input elements per patch
    // This is effectively a linear layer: out = input @ W^T + bias
    // where input is [num_patches, kernel_vol] and W is [C_out, kernel_vol]
    linear_bf16(out, input, w_bf16, num_patches, kernel_vol, C_out, scratch);
    if (bias) {
        for (int i = 0; i < num_patches; i++)
            vec_add(out + i * C_out, out + i * C_out, bias, C_out);
    }
}

// ── Embedding lookup (bf16 → f32) ──────────────────────────────────────────

inline void embedding_bf16(float* out, const uint16_t* table,
                           const int* token_ids, int num_tokens,
                           int hidden_size) {
    for (int t = 0; t < num_tokens; t++) {
        int id = token_ids[t];
        const uint16_t* row = table + static_cast<size_t>(id) * hidden_size;
        bf16_to_f32_array(out + t * hidden_size, row, hidden_size);
    }
}

// ── Causal depthwise Conv1d ─────────────────────────────────────────────────
// x: [seq_len, channels]   (input, transposed from [channels, seq_len])
// w: bf16 [channels, 1, kernel_size]  → depthwise weights
// out: [seq_len, channels]
// Causal: output[t] depends on x[t-kernel_size+1 .. t]

inline void causal_conv1d_bf16(float* out, const float* x,
                               const uint16_t* w_bf16,
                               int seq_len, int channels, int kernel_size,
                               Scratch& scratch) {
    // Convert weights: [channels, 1, kernel_size] row-major
    int wn = channels * kernel_size;
    float* w = scratch.get(static_cast<size_t>(wn));
    bf16_to_f32_array(w, w_bf16, static_cast<size_t>(wn));

    for (int t = 0; t < seq_len; t++) {
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            for (int k = 0; k < kernel_size; k++) {
                int src_t = t - (kernel_size - 1) + k;
                float xv = (src_t >= 0) ? x[src_t * channels + c] : 0.0f;
                sum += xv * w[c * kernel_size + k];
            }
            out[t * channels + c] = sum;
        }
    }
}

// Causal conv1d single-step update (decode mode)
// conv_state: [channels, kernel_size-1]  (ring buffer of past inputs)
// new_x: [channels]
// Returns output for the single new position

inline void causal_conv1d_step(float* out, float* conv_state,
                               const float* new_x,
                               const float* w_f32,
                               int channels, int kernel_size) {
    int state_len = kernel_size - 1;
    for (int c = 0; c < channels; c++) {
        // Compute convolution on [old_state..., new_x] first.
        // This matches torch_causal_conv1d_update(hidden_states_new = cat([state, new_x])).
        float sum = 0.0f;
        for (int k = 0; k < state_len; k++)
            sum += conv_state[c * state_len + k] * w_f32[c * kernel_size + k];
        sum += new_x[c] * w_f32[c * kernel_size + state_len];
        out[c] = sum;

        // Then roll the state and append new_x for the next decode step.
        for (int k = 0; k < state_len - 1; k++)
            conv_state[c * state_len + k] = conv_state[c * state_len + k + 1];
        if (state_len > 0)
            conv_state[c * state_len + state_len - 1] = new_x[c];
    }
}

// ── RoPE: apply_rotary_pos_emb ──────────────────────────────────────────────
// Applies rotation to the first rotary_dim dimensions, passes the rest through.
// q/k shape: [seq_len, num_heads, head_dim]
// cos/sin shape: [seq_len, rotary_dim]

inline void apply_rope(float* q, float* k,
                       const float* cos_vals, const float* sin_vals,
                       int seq_len, int num_q_heads, int num_k_heads,
                       int head_dim, int rotary_dim) {
    int half_rot = rotary_dim / 2;

    auto rotate = [&](float* x, int num_heads) {
        for (int s = 0; s < seq_len; s++) {
            const float* c = cos_vals + s * rotary_dim;
            const float* si = sin_vals + s * rotary_dim;
            for (int h = 0; h < num_heads; h++) {
                float* head = x + (s * num_heads + h) * head_dim;
                // rotate_half: [-x2, x1] where x1 = first half, x2 = second half
                for (int i = 0; i < half_rot; i++) {
                    float x1 = head[i];
                    float x2 = head[i + half_rot];
                    head[i]            = x1 * c[i]            - x2 * si[i];
                    head[i + half_rot] = x2 * c[i + half_rot] + x1 * si[i + half_rot];
                }
                // dimensions beyond rotary_dim are left unchanged
            }
        }
    };

    rotate(q, num_q_heads);
    rotate(k, num_k_heads);
}

// ── RoPE for vision (full rotation, no partial, no head-dim passthrough) ────
// q/k shape: [seq_len, num_heads, head_dim]
// cos/sin shape: [seq_len, head_dim]
// Rotates ALL dimensions (unlike text which uses partial_rotary_factor)

inline void apply_rope_vision(float* q, float* k,
                               const float* cos_vals, const float* sin_vals,
                               int seq_len, int num_heads, int head_dim) {
    int half = head_dim / 2;
    auto rotate = [&](float* x) {
        for (int s = 0; s < seq_len; s++) {
            const float* c = cos_vals + s * head_dim;
            const float* si = sin_vals + s * head_dim;
            for (int h = 0; h < num_heads; h++) {
                float* hd = x + (s * num_heads + h) * head_dim;
                for (int i = 0; i < half; i++) {
                    float x1 = hd[i];
                    float x2 = hd[i + half];
                    hd[i]        = x1 * c[i]        - x2 * si[i];
                    hd[i + half] = x2 * c[i + half] + x1 * si[i + half];
                }
            }
        }
    };
    rotate(q);
    rotate(k);
}

// ── Scaled dot-product attention ────────────────────────────────────────────
// q: [seq_q, num_heads, head_dim]
// k: [seq_kv, num_kv_heads, head_dim]
// v: [seq_kv, num_kv_heads, head_dim]
// out: [seq_q, num_heads, head_dim]
// mask: optional [seq_q, seq_kv]  (additive, -inf for masked positions)
// GQA: num_heads / num_kv_heads groups

inline void attention(float* out,
                      const float* q, const float* k, const float* v,
                      const float* mask,
                      int seq_q, int seq_kv,
                      int num_heads, int num_kv_heads, int head_dim,
                      float scale, Scratch& scratch) {
    int groups = num_heads / num_kv_heads;

    // Scratch for attention scores: [seq_kv]
    float* scores = scratch.get(static_cast<size_t>(seq_kv));

    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / groups;
        for (int sq = 0; sq < seq_q; sq++) {
            const float* q_vec = q + (sq * num_heads + h) * head_dim;

            // Compute scores: Q @ K^T * scale
            for (int sk = 0; sk < seq_kv; sk++) {
                const float* k_vec = k + (sk * num_kv_heads + kv_h) * head_dim;
                float dot = 0.0f;
                vDSP_dotpr(q_vec, 1, k_vec, 1, &dot,
                           static_cast<vDSP_Length>(head_dim));
                scores[sk] = dot * scale;
            }

            // Apply mask
            if (mask) {
                for (int sk = 0; sk < seq_kv; sk++)
                    scores[sk] += mask[sq * seq_kv + sk];
            }

            // Softmax
            softmax_inplace(scores, seq_kv);

            // Weighted sum of V
            float* o_vec = out + (sq * num_heads + h) * head_dim;
            std::memset(o_vec, 0, head_dim * sizeof(float));
            for (int sk = 0; sk < seq_kv; sk++) {
                const float* v_vec = v + (sk * num_kv_heads + kv_h) * head_dim;
                // o_vec += scores[sk] * v_vec
                cblas_saxpy(head_dim, scores[sk], v_vec, 1, o_vec, 1);
            }
        }
    }
}
