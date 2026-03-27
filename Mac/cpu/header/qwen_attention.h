#ifndef QWEN_ATTENTION_H
#define QWEN_ATTENTION_H

#include "header/kv_cache.h"
#include "header/grouped_query_attention.h"
#include "header/linear.h"
#include "header/rope.h"

class QwenAttention {
public:
    QwenAttention(
        Linear q_proj,
        Linear k_proj,
        Linear v_proj,
        Linear o_proj,
        RoPE rope,
        GroupedQueryAttention attention);

    std::size_t num_query_heads() const;
    std::size_t num_key_value_heads() const;
    std::size_t head_dim() const;

    Tensor Forward(const Tensor& hidden_states, std::size_t position_offset = 0) const;
    Tensor ForwardCached(const Tensor& hidden_states, TensorKVCache& cache, std::size_t layer_index, std::size_t position_offset = 0) const;

private:
    Linear q_proj_;
    Linear k_proj_;
    Linear v_proj_;
    Linear o_proj_;
    RoPE rope_;
    GroupedQueryAttention attention_;
};

#endif