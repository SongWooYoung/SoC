#include "header/grouped_query_attention.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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

GroupedQueryAttention::GroupedQueryAttention(
    std::size_t num_query_heads,
    std::size_t num_key_value_heads,
    std::size_t head_dim,
    bool causal)
    : num_query_heads_(num_query_heads),
      num_key_value_heads_(num_key_value_heads),
      head_dim_(head_dim),
      causal_(causal) {
    Require(num_query_heads_ != 0, "grouped query attention requires at least one query head");
    Require(num_key_value_heads_ != 0, "grouped query attention requires at least one key/value head");
    Require(head_dim_ != 0, "grouped query attention requires non-zero head_dim");
    Require(num_query_heads_ % num_key_value_heads_ == 0,
            "grouped query attention requires num_query_heads divisible by num_key_value_heads");
}

std::size_t GroupedQueryAttention::num_query_heads() const {
    return num_query_heads_;
}

std::size_t GroupedQueryAttention::num_key_value_heads() const {
    return num_key_value_heads_;
}

std::size_t GroupedQueryAttention::head_dim() const {
    return head_dim_;
}

bool GroupedQueryAttention::causal() const {
    return causal_;
}

Tensor GroupedQueryAttention::Forward(const Tensor& query, const Tensor& key, const Tensor& value) const {
    return ForwardWithPrefix(query, key, value, 0);
}

Tensor GroupedQueryAttention::ForwardWithPrefix(const Tensor& query, const Tensor& key, const Tensor& value, std::size_t prefix_length) const {
    Require(query.dtype() == DType::Float32, "attention query must be float32");
    Require(key.dtype() == DType::Float32, "attention key must be float32");
    Require(value.dtype() == DType::Float32, "attention value must be float32");
    Require(query.dim() == 4, "attention query must have shape [batch, seq, q_heads, head_dim]");
    Require(key.dim() == 4 && value.dim() == 4, "attention key/value must have shape [batch, seq, kv_heads, head_dim]");
    Require(query.shape()[0] == key.shape()[0] && query.shape()[0] == value.shape()[0], "attention batch dimensions must match");
    Require(key.shape()[1] == value.shape()[1], "attention key and value sequence lengths must match");
    Require(key.shape()[2] == value.shape()[2], "attention key and value head counts must match");
    Require(query.shape()[3] == key.shape()[3] && query.shape()[3] == value.shape()[3], "attention head dimensions must match");
    Require(query.shape()[2] == num_query_heads_, "attention query head count must match constructor configuration");
    Require(key.shape()[2] == num_key_value_heads_, "attention key/value head count must match constructor configuration");
    Require(query.shape()[3] == head_dim_, "attention head_dim must match constructor configuration");
    RequireContiguous(query, "attention query");
    RequireContiguous(key, "attention key");
    RequireContiguous(value, "attention value");

    const std::size_t batch_size = query.shape()[0];
    const std::size_t query_length = query.shape()[1];
    const std::size_t key_length = key.shape()[1];
    Require(prefix_length + query_length <= key_length, "attention prefix length exceeds available key sequence length");
    const std::size_t kv_group_size = num_query_heads_ / num_key_value_heads_;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    Storage output_storage = Storage::AllocateOwned(query.numel() * sizeof(float));
    Tensor output(std::move(output_storage), DType::Float32, query.shape());

    const float* query_data = query.data<const float>();
    const float* key_data = key.data<const float>();
    const float* value_data = value.data<const float>();
    float* output_data = output.data<float>();

    const std::size_t query_batch_stride = query_length * num_query_heads_ * head_dim_;
    const std::size_t key_batch_stride = key_length * num_key_value_heads_ * head_dim_;
    std::vector<float> scores(key_length, 0.0f);
    std::vector<float> weights(key_length, 0.0f);

    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        for (std::size_t query_position = 0; query_position < query_length; ++query_position) {
            const std::size_t visible_key_count = causal_ ? std::min(key_length, prefix_length + query_position + 1) : key_length;

            for (std::size_t query_head = 0; query_head < num_query_heads_; ++query_head) {
                const std::size_t kv_head = query_head / kv_group_size;
                const float* query_vector = query_data + batch * query_batch_stride + (query_position * num_query_heads_ + query_head) * head_dim_;

                float max_score = -std::numeric_limits<float>::infinity();
                for (std::size_t key_position = 0; key_position < visible_key_count; ++key_position) {
                    const float* key_vector = key_data + batch * key_batch_stride + (key_position * num_key_value_heads_ + kv_head) * head_dim_;
                    float score = 0.0f;
                    for (std::size_t dim = 0; dim < head_dim_; ++dim) {
                        score += query_vector[dim] * key_vector[dim];
                    }
                    score *= scale;
                    scores[key_position] = score;
                    if (score > max_score) {
                        max_score = score;
                    }
                }

                float weight_sum = 0.0f;
                for (std::size_t key_position = 0; key_position < visible_key_count; ++key_position) {
                    const float weight = std::exp(scores[key_position] - max_score);
                    weights[key_position] = weight;
                    weight_sum += weight;
                }

                float* output_vector = output_data + batch * query_batch_stride + (query_position * num_query_heads_ + query_head) * head_dim_;
                for (std::size_t dim = 0; dim < head_dim_; ++dim) {
                    output_vector[dim] = 0.0f;
                }

                for (std::size_t key_position = 0; key_position < visible_key_count; ++key_position) {
                    const float normalized_weight = weights[key_position] / weight_sum;
                    const float* value_vector = value_data + batch * key_batch_stride + (key_position * num_key_value_heads_ + kv_head) * head_dim_;
                    for (std::size_t dim = 0; dim < head_dim_; ++dim) {
                        output_vector[dim] += normalized_weight * value_vector[dim];
                    }
                }
            }
        }
    }

    return output;
}