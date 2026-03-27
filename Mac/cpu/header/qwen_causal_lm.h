#ifndef QWEN_CAUSAL_LM_H
#define QWEN_CAUSAL_LM_H

#include <optional>
#include <vector>

#include "header/embedding.h"
#include "header/kv_cache.h"
#include "header/linear.h"
#include "header/qwen_block.h"
#include "header/rms_norm_module.h"

class QwenCausalLM {
public:
    QwenCausalLM(Embedding embed_tokens, std::vector<QwenBlock> blocks, RMSNormModule final_norm);
    QwenCausalLM(Embedding embed_tokens, std::vector<QwenBlock> blocks, RMSNormModule final_norm, Linear lm_head);

    std::size_t num_layers() const;
    std::size_t num_key_value_heads() const;
    std::size_t head_dim() const;

    Tensor ForwardHidden(const Tensor& token_ids, std::size_t position_offset = 0) const;
    Tensor ForwardLogits(const Tensor& token_ids, std::size_t position_offset = 0) const;
    Tensor ForwardHiddenCached(const Tensor& token_ids, TensorKVCache& cache, std::size_t position_offset = 0) const;
    Tensor ForwardLogitsCached(const Tensor& token_ids, TensorKVCache& cache, std::size_t position_offset = 0) const;

private:
    Tensor ComputeTiedLogits(const Tensor& hidden_states) const;

    Embedding embed_tokens_;
    std::vector<QwenBlock> blocks_;
    RMSNormModule final_norm_;
    std::optional<Linear> lm_head_;
};

#endif