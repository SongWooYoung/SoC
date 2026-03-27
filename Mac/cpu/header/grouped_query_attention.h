#ifndef GROUPED_QUERY_ATTENTION_H
#define GROUPED_QUERY_ATTENTION_H

#include "header/tensor.h"

class GroupedQueryAttention {
public:
    GroupedQueryAttention(std::size_t num_query_heads, std::size_t num_key_value_heads, std::size_t head_dim, bool causal = true);

    std::size_t num_query_heads() const;
    std::size_t num_key_value_heads() const;
    std::size_t head_dim() const;
    bool causal() const;

    Tensor Forward(const Tensor& query, const Tensor& key, const Tensor& value) const;
    Tensor ForwardWithPrefix(const Tensor& query, const Tensor& key, const Tensor& value, std::size_t prefix_length) const;

private:
    std::size_t num_query_heads_;
    std::size_t num_key_value_heads_;
    std::size_t head_dim_;
    bool causal_;
};

#endif