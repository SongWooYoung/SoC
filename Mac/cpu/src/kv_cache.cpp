#include "header/kv_cache.h"

#include <stdexcept>
#include <string>

namespace {
void Require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void RequireContiguous(const Tensor& tensor, const char* name) {
    if (!tensor.is_contiguous()) {
        throw std::runtime_error(std::string(name) + " must be contiguous");
    }
}
}

TensorKVCache::TensorKVCache(
    std::size_t num_layers,
    std::size_t max_batch_size,
    std::size_t num_key_value_heads,
    std::size_t head_dim,
    std::size_t max_sequence_length)
    : num_layers_(num_layers),
      max_batch_size_(max_batch_size),
      num_key_value_heads_(num_key_value_heads),
      head_dim_(head_dim),
      max_sequence_length_(max_sequence_length),
      lengths_(max_batch_size, 0) {
    Require(num_layers_ != 0, "kv cache requires at least one layer");
    Require(max_batch_size_ != 0, "kv cache requires positive batch capacity");
    Require(num_key_value_heads_ != 0, "kv cache requires positive kv head count");
    Require(head_dim_ != 0, "kv cache requires positive head_dim");
    Require(max_sequence_length_ != 0, "kv cache requires positive max_sequence_length");
    AllocateStorage();
}

void TensorKVCache::Reserve(std::size_t max_batch_size, std::size_t max_sequence_length) {
    Require(max_batch_size != 0, "kv cache reserve requires positive batch capacity");
    Require(max_sequence_length != 0, "kv cache reserve requires positive max_sequence_length");

    max_batch_size_ = max_batch_size;
    max_sequence_length_ = max_sequence_length;
    lengths_.assign(max_batch_size_, 0);
    AllocateStorage();
}

void TensorKVCache::Write(std::size_t layer_index, std::size_t batch_index, std::size_t position, const Tensor& key, const Tensor& value) {
    Require(layer_index < num_layers_, "kv cache write layer_index out of range");
    Require(batch_index < max_batch_size_, "kv cache write batch_index out of range");
    Require(position < max_sequence_length_, "kv cache write position out of range");
    Require(key.dtype() == DType::Float32 && value.dtype() == DType::Float32, "kv cache write inputs must be float32");
    Require(key.shape() == value.shape(), "kv cache write key/value shapes must match");
    Require(key.dim() == 2, "kv cache write tensors must have shape [kv_heads, head_dim]");
    Require(key.shape()[0] == num_key_value_heads_ && key.shape()[1] == head_dim_, "kv cache write tensors must match configured kv shape");
    RequireContiguous(key, "kv cache key");
    RequireContiguous(value, "kv cache value");

    const float* key_data = key.data<const float>();
    const float* value_data = value.data<const float>();
    float* cache_key_data = reinterpret_cast<float*>(key_storage_.data());
    float* cache_value_data = reinterpret_cast<float*>(value_storage_.data());

    for (std::size_t head = 0; head < num_key_value_heads_; ++head) {
        for (std::size_t dim = 0; dim < head_dim_; ++dim) {
            const std::size_t source_offset = head * head_dim_ + dim;
            const std::size_t destination_offset = FlatOffset(layer_index, batch_index, head, position, dim);
            cache_key_data[destination_offset] = key_data[source_offset];
            cache_value_data[destination_offset] = value_data[source_offset];
        }
    }

    if (position + 1 > lengths_[batch_index]) {
        lengths_[batch_index] = position + 1;
    }
}

std::pair<Tensor, Tensor> TensorKVCache::Read(std::size_t layer_index, std::size_t batch_index, std::size_t seq_start, std::size_t seq_end) const {
    Require(layer_index < num_layers_, "kv cache read layer_index out of range");
    Require(batch_index < max_batch_size_, "kv cache read batch_index out of range");
    Require(seq_start <= seq_end, "kv cache read range must be ordered");
    Require(seq_end <= lengths_[batch_index], "kv cache read range exceeds cached sequence length");

    const std::size_t sequence_length = seq_end - seq_start;
    const std::vector<std::size_t> output_shape = {sequence_length, num_key_value_heads_, head_dim_};
    Storage key_output_storage = Storage::AllocateOwned(sequence_length * num_key_value_heads_ * head_dim_ * sizeof(float));
    Storage value_output_storage = Storage::AllocateOwned(sequence_length * num_key_value_heads_ * head_dim_ * sizeof(float));
    Tensor key_output(std::move(key_output_storage), DType::Float32, output_shape);
    Tensor value_output(std::move(value_output_storage), DType::Float32, output_shape);

    const float* cache_key_data = reinterpret_cast<const float*>(key_storage_.data());
    const float* cache_value_data = reinterpret_cast<const float*>(value_storage_.data());
    float* key_output_data = key_output.data<float>();
    float* value_output_data = value_output.data<float>();

    for (std::size_t position = 0; position < sequence_length; ++position) {
        for (std::size_t head = 0; head < num_key_value_heads_; ++head) {
            for (std::size_t dim = 0; dim < head_dim_; ++dim) {
                const std::size_t output_offset = (position * num_key_value_heads_ + head) * head_dim_ + dim;
                const std::size_t cache_offset = FlatOffset(layer_index, batch_index, head, seq_start + position, dim);
                key_output_data[output_offset] = cache_key_data[cache_offset];
                value_output_data[output_offset] = cache_value_data[cache_offset];
            }
        }
    }

    return {std::move(key_output), std::move(value_output)};
}

void TensorKVCache::ClearSequence(std::size_t batch_index) {
    Require(batch_index < max_batch_size_, "kv cache clear batch_index out of range");
    lengths_[batch_index] = 0;
}

std::size_t TensorKVCache::num_layers() const {
    return num_layers_;
}

std::size_t TensorKVCache::max_batch_size() const {
    return max_batch_size_;
}

std::size_t TensorKVCache::num_key_value_heads() const {
    return num_key_value_heads_;
}

std::size_t TensorKVCache::head_dim() const {
    return head_dim_;
}

std::size_t TensorKVCache::max_sequence_length() const {
    return max_sequence_length_;
}

std::size_t TensorKVCache::length(std::size_t batch_index) const {
    Require(batch_index < max_batch_size_, "kv cache length batch_index out of range");
    return lengths_[batch_index];
}

void TensorKVCache::AllocateStorage() {
    const std::size_t total_elements = num_layers_ * max_batch_size_ * num_key_value_heads_ * max_sequence_length_ * head_dim_;
    key_storage_ = Storage::AllocateOwned(total_elements * sizeof(float));
    value_storage_ = Storage::AllocateOwned(total_elements * sizeof(float));
}

std::size_t TensorKVCache::FlatOffset(
    std::size_t layer_index,
    std::size_t batch_index,
    std::size_t head_index,
    std::size_t position_index,
    std::size_t dim_index) const {
    return ((((layer_index * max_batch_size_ + batch_index) * num_key_value_heads_ + head_index) * max_sequence_length_ + position_index) * head_dim_ + dim_index);
}