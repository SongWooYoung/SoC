#ifndef QWEN_BLOCK_H
#define QWEN_BLOCK_H

#include "header/kv_cache.h"
#include "header/qwen_attention.h"
#include "header/qwen_mlp.h"
#include "header/rms_norm_module.h"

class QwenBlock {
public:
    QwenBlock(RMSNormModule input_norm, QwenAttention attention, RMSNormModule post_attention_norm, QwenMLP mlp);

    std::size_t num_key_value_heads() const;
    std::size_t head_dim() const;

    Tensor Forward(const Tensor& hidden_states, std::size_t position_offset = 0) const;
    Tensor ForwardCached(const Tensor& hidden_states, TensorKVCache& cache, std::size_t layer_index, std::size_t position_offset = 0) const;

private:
    RMSNormModule input_norm_;
    QwenAttention attention_;
    RMSNormModule post_attention_norm_;
    QwenMLP mlp_;
};

#endif