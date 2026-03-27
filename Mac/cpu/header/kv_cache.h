#ifndef KV_CACHE_H
#define KV_CACHE_H

#include <cstddef>
#include <utility>
#include <vector>

#include "header/tensor.h"

class TensorKVCache {
public:
    TensorKVCache(std::size_t num_layers, std::size_t max_batch_size, std::size_t num_key_value_heads, std::size_t head_dim, std::size_t max_sequence_length);

    void Reserve(std::size_t max_batch_size, std::size_t max_sequence_length);
    void Write(std::size_t layer_index, std::size_t batch_index, std::size_t position, const Tensor& key, const Tensor& value);
    std::pair<Tensor, Tensor> Read(std::size_t layer_index, std::size_t batch_index, std::size_t seq_start, std::size_t seq_end) const;
    void ClearSequence(std::size_t batch_index);

    std::size_t num_layers() const;
    std::size_t max_batch_size() const;
    std::size_t num_key_value_heads() const;
    std::size_t head_dim() const;
    std::size_t max_sequence_length() const;
    std::size_t length(std::size_t batch_index) const;

private:
    void AllocateStorage();
    std::size_t FlatOffset(std::size_t layer_index, std::size_t batch_index, std::size_t head_index, std::size_t position_index, std::size_t dim_index) const;

    std::size_t num_layers_;
    std::size_t max_batch_size_;
    std::size_t num_key_value_heads_;
    std::size_t head_dim_;
    std::size_t max_sequence_length_;
    Storage key_storage_;
    Storage value_storage_;
    std::vector<std::size_t> lengths_;
};

#endif