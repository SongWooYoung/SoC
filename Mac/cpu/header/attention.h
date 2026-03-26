#ifndef ATTENTION_H
#define ATTENTION_H

#include "header/matrix.h"
#include "header/matrix_ops.h"
#include "header/activation.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>

// ============================================================================
// Attention Mechanisms
// ============================================================================

template<typename T>
class batched_matrix {
private:
    int batches;
    int rows;
    int cols;
    std::vector<T> data;

    int flat_index(int batch, int row, int col) const {
        return ((batch * rows) + row) * cols + col;
    }

    void validate_indices(int batch, int row, int col) const {
        if (batch < 0 || batch >= batches || row < 0 || row >= rows || col < 0 || col >= cols) {
            throw std::out_of_range("batched_matrix index out of range");
        }
    }

public:
    batched_matrix(int batch_count, int row_count, int col_count)
        : batches(batch_count), rows(row_count), cols(col_count),
          data(static_cast<std::size_t>(batch_count) * row_count * col_count, T{}) {
        if (batch_count < 0 || row_count < 0 || col_count < 0) {
            throw std::invalid_argument("batched_matrix dimensions must be non-negative");
        }
    }

    int batch_count() const {
        return batches;
    }

    int row_count() const {
        return rows;
    }

    int col_count() const {
        return cols;
    }

    T& operator()(int batch, int row, int col) {
        validate_indices(batch, row, col);
        return data[flat_index(batch, row, col)];
    }

    const T& operator()(int batch, int row, int col) const {
        validate_indices(batch, row, col);
        return data[flat_index(batch, row, col)];
    }

    matrix<T> batch_slice(int batch) const {
        if (batch < 0 || batch >= batches) {
            throw std::out_of_range("batch index out of range");
        }

        matrix<T> result(rows, cols);
        if (rows == 0 || cols == 0) {
            return result;
        }

        const T* source = data.data() + batch * rows * cols;
        std::copy(source, source + rows * cols, result.raw_data());
        return result;
    }

    matrix<T> rows_slice(int batch, int start_row, int row_count) const {
        if (batch < 0 || batch >= batches) {
            throw std::out_of_range("batch index out of range");
        }
        if (start_row < 0 || row_count < 0 || start_row + row_count > rows) {
            throw std::out_of_range("row slice out of range");
        }

        matrix<T> result(row_count, cols);
        T* destination = result.raw_data();

        for (int row = 0; row < row_count; ++row) {
            const T* source = data.data() + flat_index(batch, start_row + row, 0);
            std::copy(source, source + cols, destination + row * cols);
        }

        return result;
    }

    void set_batch_slice(int batch, const matrix<T>& source) {
        if (batch < 0 || batch >= batches) {
            throw std::out_of_range("batch index out of range");
        }
        if (source.row_count() != rows || source.col_count() != cols) {
            throw std::invalid_argument("batch slice dimensions do not match batched_matrix dimensions");
        }

        T* destination = data.data() + batch * rows * cols;
        std::copy(source.raw_data(), source.raw_data() + rows * cols, destination);
    }

    void set_rows_slice(int batch, int start_row, const matrix<T>& source) {
        if (batch < 0 || batch >= batches) {
            throw std::out_of_range("batch index out of range");
        }
        if (source.col_count() != cols || start_row < 0 || start_row + source.row_count() > rows) {
            throw std::invalid_argument("row slice dimensions do not match batched_matrix dimensions");
        }

        for (int row = 0; row < source.row_count(); ++row) {
            T* destination = data.data() + flat_index(batch, start_row + row, 0);
            const T* source_row = source.raw_data() + row * cols;
            std::copy(source_row, source_row + cols, destination);
        }
    }
};

template<typename T>
class KVCache {
private:
    batched_matrix<T> cached_keys;
    batched_matrix<T> cached_values;
    int current_length;

public:
    KVCache(int batch_size, int max_seq_len, int key_dim, int value_dim)
        : cached_keys(batch_size, max_seq_len, key_dim),
          cached_values(batch_size, max_seq_len, value_dim),
          current_length(0) {}

    int batch_count() const {
        return cached_keys.batch_count();
    }

    int length() const {
        return current_length;
    }

    int capacity() const {
        return cached_keys.row_count();
    }

    int key_dim() const {
        return cached_keys.col_count();
    }

    int value_dim() const {
        return cached_values.col_count();
    }

    void Clear() {
        current_length = 0;
    }

    void Append(const batched_matrix<T>& new_keys, const batched_matrix<T>& new_values) {
        if (new_keys.batch_count() != batch_count() || new_values.batch_count() != batch_count()) {
            throw std::invalid_argument("KV cache append batch size mismatch");
        }
        if (new_keys.row_count() != new_values.row_count()) {
            throw std::invalid_argument("KV cache append sequence length mismatch");
        }
        if (new_keys.col_count() != key_dim() || new_values.col_count() != value_dim()) {
            throw std::invalid_argument("KV cache append feature dimension mismatch");
        }
        if (current_length + new_keys.row_count() > capacity()) {
            throw std::out_of_range("KV cache capacity exceeded");
        }

        for (int batch = 0; batch < batch_count(); ++batch) {
            cached_keys.set_rows_slice(batch, current_length, new_keys.batch_slice(batch));
            cached_values.set_rows_slice(batch, current_length, new_values.batch_slice(batch));
        }

        current_length += new_keys.row_count();
    }

    matrix<T> KeysForBatch(int batch) const {
        return cached_keys.rows_slice(batch, 0, current_length);
    }

    matrix<T> ValuesForBatch(int batch) const {
        return cached_values.rows_slice(batch, 0, current_length);
    }
};

namespace attention_detail {

template<typename T>
matrix<T> SliceColumns(const matrix<T>& source, int start_col, int width) {
    if (start_col < 0 || width < 0 || start_col + width > source.col_count()) {
        throw std::out_of_range("column slice out of range");
    }

    matrix<T> result(source.row_count(), width);
    for (int row = 0; row < source.row_count(); ++row) {
        for (int col = 0; col < width; ++col) {
            result[row][col] = source[row][start_col + col];
        }
    }

    return result;
}

template<typename T>
matrix<T> ApplyOutputProjection(const matrix<T>& input,
                                const matrix<T>* output_projection,
                                const matrix<T>* output_bias) {
    if (output_projection == nullptr) {
        return input;
    }

    if (input.col_count() != output_projection->row_count()) {
        throw std::invalid_argument("output projection dimensions do not match attention output");
    }

    matrix<T> projected = MatMul(input, *output_projection);
    if (output_bias == nullptr) {
        return projected;
    }

    if (output_bias->row_count() != 1 || output_bias->col_count() != projected.col_count()) {
        throw std::invalid_argument("output bias must have shape (1, output_dim)");
    }

    T* projected_data = projected.raw_data();
    const T* bias_data = output_bias->raw_data();
    const int rows = projected.row_count();
    const int cols = projected.col_count();

    for (int row = 0; row < rows; ++row) {
        T* projected_row = projected_data + row * cols;
        for (int col = 0; col < cols; ++col) {
            projected_row[col] += bias_data[col];
        }
    }

    return projected;
}

template<typename T>
matrix<T> MultiHeadAttentionImpl(const matrix<T>& Q,
                                 const matrix<T>& K,
                                 const matrix<T>& V,
                                 int num_heads,
                                 const matrix<int>* mask,
                                 const matrix<T>* output_projection,
                                 const matrix<T>* output_bias,
                                 T scale_factor) {
    if (num_heads <= 0) {
        throw std::invalid_argument("num_heads must be positive");
    }
    if (K.row_count() != V.row_count()) {
        throw std::invalid_argument("K and V must have same sequence length");
    }
    if (Q.col_count() != K.col_count()) {
        throw std::invalid_argument("Q and K must have same feature dimension");
    }

    const int seq_len_q = Q.row_count();
    const int seq_len_k = K.row_count();
    const int d_model = Q.col_count();
    const int v_model = V.col_count();

    if (d_model % num_heads != 0) {
        throw std::invalid_argument("d_model must be divisible by num_heads");
    }
    if (v_model % num_heads != 0) {
        throw std::invalid_argument("V feature dimension must be divisible by num_heads");
    }
    if (mask != nullptr && (mask->row_count() != seq_len_q || mask->col_count() != seq_len_k)) {
        throw std::invalid_argument("attention mask dimensions must match attention score dimensions");
    }

    const int d_k = d_model / num_heads;
    const int d_v = v_model / num_heads;

    if (scale_factor < T(0)) {
        scale_factor = T(1) / std::sqrt(static_cast<T>(d_k));
    }

    matrix<T> concatenated(seq_len_q, v_model);
    T* concatenated_data = concatenated.raw_data();

    for (int head = 0; head < num_heads; ++head) {
        matrix<T> q_head = SliceColumns(Q, head * d_k, d_k);
        matrix<T> k_head = SliceColumns(K, head * d_k, d_k);
        matrix<T> v_head = SliceColumns(V, head * d_v, d_v);

        matrix<T> head_output = mask == nullptr
            ? ScaledDotProductAttention(q_head, k_head, v_head, scale_factor)
            : MaskedScaledDotProductAttention(q_head, k_head, v_head, *mask, scale_factor);

        const T* head_output_data = head_output.raw_data();
        for (int row = 0; row < seq_len_q; ++row) {
            T* destination = concatenated_data + row * v_model + head * d_v;
            const T* source = head_output_data + row * d_v;
            std::copy(source, source + d_v, destination);
        }
    }

    return ApplyOutputProjection(concatenated, output_projection, output_bias);
}

} // namespace attention_detail

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
    
    // Compute scale factor if not provided
    if (scale_factor < T(0)) {
        scale_factor = T(1) / std::sqrt(static_cast<T>(d_k));
    }
    
    // Compute Q @ K^T using K directly as the cached transposed rhs view.
    matrix<T> scores = MatMulTransposedB(Q, K);
    
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
    
    // Compute Q @ K^T using K directly as the cached transposed rhs view.
    matrix<T> scores = MatMulTransposedB(Q, K);
    
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

inline matrix<int> CreateIncrementalCausalMask(int query_len, int total_len, int prefix_len) {
    if (query_len < 0 || total_len < 0 || prefix_len < 0 || prefix_len > total_len) {
        throw std::invalid_argument("invalid dimensions for incremental causal mask");
    }

    matrix<int> mask(query_len, total_len);
    int* mask_data = mask.raw_data();

    for (int query_index = 0; query_index < query_len; ++query_index) {
        int visible_until = prefix_len + query_index;
        for (int key_index = 0; key_index < total_len; ++key_index) {
            mask_data[query_index * total_len + key_index] = (key_index > visible_until) ? 1 : 0;
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
    return attention_detail::MultiHeadAttentionImpl(
        Q,
        K,
        V,
        num_heads,
        static_cast<const matrix<int>*>(nullptr),
        static_cast<const matrix<T>*>(nullptr),
        static_cast<const matrix<T>*>(nullptr),
        scale_factor);
}

template<typename T>
matrix<T> MultiHeadAttention(const matrix<T>& Q, const matrix<T>& K, const matrix<T>& V,
                             int num_heads, const matrix<T>& output_projection,
                             const matrix<T>* output_bias = nullptr, T scale_factor = T(-1)) {
    return attention_detail::MultiHeadAttentionImpl(
        Q,
        K,
        V,
        num_heads,
        static_cast<const matrix<int>*>(nullptr),
        &output_projection,
        output_bias,
        scale_factor);
}

template<typename T>
batched_matrix<T> MultiHeadAttention(const batched_matrix<T>& Q,
                                     const batched_matrix<T>& K,
                                     const batched_matrix<T>& V,
                                     int num_heads,
                                     T scale_factor = T(-1)) {
    if (Q.batch_count() != K.batch_count() || Q.batch_count() != V.batch_count()) {
        throw std::invalid_argument("batched attention batch sizes must match");
    }
    if (Q.row_count() != K.row_count() || K.row_count() != V.row_count()) {
        throw std::invalid_argument("batched attention sequence lengths must match");
    }

    batched_matrix<T> result(Q.batch_count(), Q.row_count(), V.col_count());
    for (int batch = 0; batch < Q.batch_count(); ++batch) {
        result.set_batch_slice(batch, MultiHeadAttention(Q.batch_slice(batch), K.batch_slice(batch), V.batch_slice(batch), num_heads, scale_factor));
    }

    return result;
}

template<typename T>
batched_matrix<T> MultiHeadAttention(const batched_matrix<T>& Q,
                                     const batched_matrix<T>& K,
                                     const batched_matrix<T>& V,
                                     int num_heads,
                                     const matrix<T>& output_projection,
                                     const matrix<T>* output_bias = nullptr,
                                     T scale_factor = T(-1)) {
    if (Q.batch_count() != K.batch_count() || Q.batch_count() != V.batch_count()) {
        throw std::invalid_argument("batched attention batch sizes must match");
    }
    if (Q.row_count() != K.row_count() || K.row_count() != V.row_count()) {
        throw std::invalid_argument("batched attention sequence lengths must match");
    }

    batched_matrix<T> result(Q.batch_count(), Q.row_count(), output_projection.col_count());
    for (int batch = 0; batch < Q.batch_count(); ++batch) {
        result.set_batch_slice(batch, MultiHeadAttention(
            Q.batch_slice(batch),
            K.batch_slice(batch),
            V.batch_slice(batch),
            num_heads,
            output_projection,
            output_bias,
            scale_factor));
    }

    return result;
}

template<typename T>
batched_matrix<T> IncrementalMultiHeadAttention(const batched_matrix<T>& Q,
                                                const batched_matrix<T>& K_new,
                                                const batched_matrix<T>& V_new,
                                                KVCache<T>& cache,
                                                int num_heads,
                                                T scale_factor = T(-1)) {
    if (Q.batch_count() != K_new.batch_count() || Q.batch_count() != V_new.batch_count()) {
        throw std::invalid_argument("incremental attention batch sizes must match");
    }
    if (Q.row_count() != K_new.row_count() || Q.row_count() != V_new.row_count()) {
        throw std::invalid_argument("incremental attention chunk lengths must match");
    }

    const int prefix_length = cache.length();
    cache.Append(K_new, V_new);
    const int total_length = cache.length();
    const matrix<int> mask = CreateIncrementalCausalMask(Q.row_count(), total_length, prefix_length);

    batched_matrix<T> result(Q.batch_count(), Q.row_count(), V_new.col_count());
    for (int batch = 0; batch < Q.batch_count(); ++batch) {
        result.set_batch_slice(batch, attention_detail::MultiHeadAttentionImpl(
            Q.batch_slice(batch),
            cache.KeysForBatch(batch),
            cache.ValuesForBatch(batch),
            num_heads,
            &mask,
            nullptr,
            nullptr,
            scale_factor));
    }

    return result;
}

template<typename T>
batched_matrix<T> IncrementalMultiHeadAttention(const batched_matrix<T>& Q,
                                                const batched_matrix<T>& K_new,
                                                const batched_matrix<T>& V_new,
                                                KVCache<T>& cache,
                                                int num_heads,
                                                const matrix<T>& output_projection,
                                                const matrix<T>* output_bias = nullptr,
                                                T scale_factor = T(-1)) {
    if (Q.batch_count() != K_new.batch_count() || Q.batch_count() != V_new.batch_count()) {
        throw std::invalid_argument("incremental attention batch sizes must match");
    }
    if (Q.row_count() != K_new.row_count() || Q.row_count() != V_new.row_count()) {
        throw std::invalid_argument("incremental attention chunk lengths must match");
    }

    const int prefix_length = cache.length();
    cache.Append(K_new, V_new);
    const int total_length = cache.length();
    const matrix<int> mask = CreateIncrementalCausalMask(Q.row_count(), total_length, prefix_length);

    batched_matrix<T> result(Q.batch_count(), Q.row_count(), output_projection.col_count());
    for (int batch = 0; batch < Q.batch_count(); ++batch) {
        result.set_batch_slice(batch, attention_detail::MultiHeadAttentionImpl(
            Q.batch_slice(batch),
            cache.KeysForBatch(batch),
            cache.ValuesForBatch(batch),
            num_heads,
            &mask,
            &output_projection,
            output_bias,
            scale_factor));
    }

    return result;
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
