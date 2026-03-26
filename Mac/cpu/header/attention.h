#ifndef ATTENTION_H
#define ATTENTION_H

#include "header/matrix.h"
#include "header/matrix_ops.h"
#include "header/activation.h"
#include <cmath>
#include <algorithm>

// ============================================================================
// Attention Mechanisms
// ============================================================================

// Scaled Dot-Product Attention
// Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
template<typename T>
matrix<T> ScaledDotProductAttention(const matrix<T>& Q, const matrix<T>& K, 
                                     const matrix<T>& V, T scale_factor = T(-1)) {
    // Q: (seq_len_q, d_k)
    // K: (seq_len_k, d_k)
    // V: (seq_len_k, d_v)
    
    if (K.row_count() != V.row_count()) {
        throw std::invalid_argument("K and V must have same sequence length");
    }
    if (Q.col_count() != K.col_count()) {
        throw std::invalid_argument("Q and K must have same feature dimension");
    }
    
    const int seq_len_q = Q.row_count();
    const int seq_len_k = K.row_count();
    const int d_k = Q.col_count();
    const int d_v = V.col_count();
    
    // Compute scale factor if not provided
    if (scale_factor < T(0)) {
        scale_factor = T(1) / std::sqrt(static_cast<T>(d_k));
    }
    
    // Compute Q @ K^T
    matrix<T> K_T(K);
    K_T.transpose();
    matrix<T> scores = MatMul(Q, K_T);
    
    // Scale
    T* scores_data = scores.raw_data();
    const int scores_size = seq_len_q * seq_len_k;
    
    int i = 0;
    for (; i + 7 < scores_size; i += 8) {
        scores_data[i] *= scale_factor;
        scores_data[i + 1] *= scale_factor;
        scores_data[i + 2] *= scale_factor;
        scores_data[i + 3] *= scale_factor;
        scores_data[i + 4] *= scale_factor;
        scores_data[i + 5] *= scale_factor;
        scores_data[i + 6] *= scale_factor;
        scores_data[i + 7] *= scale_factor;
    }
    for (; i < scores_size; ++i) {
        scores_data[i] *= scale_factor;
    }
    
    // Apply softmax
    matrix<T> attention_weights = Softmax(scores, 1);
    
    // Compute attention_weights @ V
    matrix<T> output = MatMul(attention_weights, V);
    
    return output;
}

// Masked Scaled Dot-Product Attention (for causal/decoder attention)
// mask: if true at position (i,j), set attention score to -inf
template<typename T>
matrix<T> MaskedScaledDotProductAttention(const matrix<T>& Q, const matrix<T>& K, 
                                          const matrix<T>& V, const matrix<int>& mask,
                                          T scale_factor = T(-1)) {
    if (K.row_count() != V.row_count()) {
        throw std::invalid_argument("K and V must have same sequence length");
    }
    if (Q.col_count() != K.col_count()) {
        throw std::invalid_argument("Q and K must have same feature dimension");
    }
    
    const int seq_len_q = Q.row_count();
    const int seq_len_k = K.row_count();
    const int d_k = Q.col_count();
    
    if (mask.row_count() != seq_len_q || mask.col_count() != seq_len_k) {
        throw std::invalid_argument("mask dimensions must match (seq_len_q, seq_len_k)");
    }
    
    // Compute scale factor if not provided
    if (scale_factor < T(0)) {
        scale_factor = T(1) / std::sqrt(static_cast<T>(d_k));
    }
    
    // Compute Q @ K^T
    matrix<T> K_T(K);
    K_T.transpose();
    matrix<T> scores = MatMul(Q, K_T);
    
    // Scale and apply mask
    T* scores_data = scores.raw_data();
    const int* mask_data = mask.raw_data();
    const T neg_inf = -std::numeric_limits<T>::infinity();
    
    const int scores_size = seq_len_q * seq_len_k;
    for (int i = 0; i < scores_size; ++i) {
        scores_data[i] *= scale_factor;
        if (mask_data[i]) {
            scores_data[i] = neg_inf;
        }
    }
    
    // Apply softmax
    matrix<T> attention_weights = Softmax(scores, 1);
    
    // Compute attention_weights @ V
    matrix<T> output = MatMul(attention_weights, V);
    
    return output;
}

// Create causal mask (upper triangular mask for autoregressive generation)
// Returns mask where mask[i][j] = 1 if i < j (future positions), 0 otherwise
inline matrix<int> CreateCausalMask(int seq_len) {
    matrix<int> mask(seq_len, seq_len);
    int* mask_data = mask.raw_data();
    
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            mask_data[i * seq_len + j] = (j > i) ? 1 : 0;
        }
    }
    
    return mask;
}

// Rotary Position Embedding (RoPE) - used in Llama
// Applies rotary embeddings to query or key matrices
template<typename T>
matrix<T> ApplyRotaryEmbedding(const matrix<T>& x, int start_pos = 0) {
    const int seq_len = x.row_count();
    const int d_model = x.col_count();
    
    if (d_model % 2 != 0) {
        throw std::invalid_argument("d_model must be even for RoPE");
    }
    
    const T* x_data = x.raw_data();
    matrix<T> result(seq_len, d_model);
    T* result_data = result.raw_data();
    
    const int d_half = d_model / 2;
    
    // Apply rotation to each position
    for (int pos = 0; pos < seq_len; ++pos) {
        int absolute_pos = start_pos + pos;
        const T* x_row = x_data + pos * d_model;
        T* result_row = result_data + pos * d_model;
        
        // Apply rotation to each pair of dimensions
        for (int i = 0; i < d_half; ++i) {
            // Compute rotation angle
            T theta = static_cast<T>(absolute_pos) / 
                      std::pow(T(10000), static_cast<T>(2 * i) / static_cast<T>(d_model));
            
            T cos_theta = std::cos(theta);
            T sin_theta = std::sin(theta);
            
            // Apply rotation matrix
            T x1 = x_row[2 * i];
            T x2 = x_row[2 * i + 1];
            
            result_row[2 * i] = x1 * cos_theta - x2 * sin_theta;
            result_row[2 * i + 1] = x1 * sin_theta + x2 * cos_theta;
        }
    }
    
    return result;
}

// Multi-Head Attention (simplified version)
// In practice, this would involve splitting Q, K, V into multiple heads
// For now, this is a helper to split and recombine
template<typename T>
matrix<T> MultiHeadAttention(const matrix<T>& Q, const matrix<T>& K, const matrix<T>& V,
                             int num_heads, T scale_factor = T(-1)) {
    const int seq_len = Q.row_count();
    const int d_model = Q.col_count();
    
    if (d_model % num_heads != 0) {
        throw std::invalid_argument("d_model must be divisible by num_heads");
    }
    
    const int d_k = d_model / num_heads;
    
    // Compute scale factor if not provided
    if (scale_factor < T(0)) {
        scale_factor = T(1) / std::sqrt(static_cast<T>(d_k));
    }
    
    // For simplicity, we'll process this as single-head attention
    // In a real implementation, you'd split into heads, process separately, and concatenate
    // This is a placeholder for the full multi-head attention logic
    
    matrix<T> K_T(K);
    K_T.transpose();
    matrix<T> scores = MatMul(Q, K_T);
    
    // Scale
    T* scores_data = scores.raw_data();
    const int scores_size = scores.row_count() * scores.col_count();
    for (int i = 0; i < scores_size; ++i) {
        scores_data[i] *= scale_factor;
    }
    
    // Apply softmax
    matrix<T> attention_weights = Softmax(scores, 1);
    
    // Compute output
    matrix<T> output = MatMul(attention_weights, V);
    
    return output;
}

// Split matrix into multiple heads
// Input: (batch * seq_len, d_model)
// Output: reshaped view conceptually as (batch, num_heads, seq_len, d_k)
// In practice, returns same data reorganized
template<typename T>
matrix<T> SplitHeads(const matrix<T>& x, int num_heads) {
    const int seq_len = x.row_count();
    const int d_model = x.col_count();
    
    if (d_model % num_heads != 0) {
        throw std::invalid_argument("d_model must be divisible by num_heads");
    }
    
    const int d_k = d_model / num_heads;
    const T* x_data = x.raw_data();
    
    matrix<T> result(seq_len * num_heads, d_k);
    T* result_data = result.raw_data();
    
    // Reshape: (seq_len, num_heads * d_k) -> (seq_len * num_heads, d_k)
    for (int seq = 0; seq < seq_len; ++seq) {
        for (int head = 0; head < num_heads; ++head) {
            for (int k = 0; k < d_k; ++k) {
                int src_idx = seq * d_model + head * d_k + k;
                int dst_idx = (seq * num_heads + head) * d_k + k;
                result_data[dst_idx] = x_data[src_idx];
            }
        }
    }
    
    return result;
}

// Merge multiple heads back
// Inverse of SplitHeads
template<typename T>
matrix<T> MergeHeads(const matrix<T>& x, int num_heads) {
    const int total_rows = x.row_count();
    const int d_k = x.col_count();
    
    if (total_rows % num_heads != 0) {
        throw std::invalid_argument("number of rows must be divisible by num_heads");
    }
    
    const int seq_len = total_rows / num_heads;
    const int d_model = num_heads * d_k;
    const T* x_data = x.raw_data();
    
    matrix<T> result(seq_len, d_model);
    T* result_data = result.raw_data();
    
    // Reshape: (seq_len * num_heads, d_k) -> (seq_len, num_heads * d_k)
    for (int seq = 0; seq < seq_len; ++seq) {
        for (int head = 0; head < num_heads; ++head) {
            for (int k = 0; k < d_k; ++k) {
                int src_idx = (seq * num_heads + head) * d_k + k;
                int dst_idx = seq * d_model + head * d_k + k;
                result_data[dst_idx] = x_data[src_idx];
            }
        }
    }
    
    return result;
}

#endif // ATTENTION_H
