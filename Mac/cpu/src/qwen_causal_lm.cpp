#include "header/qwen_causal_lm.h"

#include "header/dtype.h"

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

void RequireLayerRange(std::size_t start_layer, std::size_t end_layer, std::size_t total_layers) {
    if (start_layer > end_layer || end_layer > total_layers) {
        throw std::runtime_error("invalid qwen layer range");
    }
}

Tensor RunBlockRange(const std::vector<QwenBlock>& blocks,
                     Tensor hidden_states,
                     TensorKVCache* cache,
                     std::size_t start_layer,
                     std::size_t end_layer,
                     std::size_t position_offset) {
    for (std::size_t layer_index = start_layer; layer_index < end_layer; ++layer_index) {
        hidden_states = cache == nullptr
            ? blocks[layer_index].Forward(hidden_states, position_offset)
            : blocks[layer_index].ForwardCached(hidden_states, *cache, layer_index, position_offset);
    }
    return hidden_states;
}
}

QwenCausalLM::QwenCausalLM(Embedding embed_tokens, std::vector<QwenBlock> blocks, RMSNormModule final_norm)
    : embed_tokens_(std::move(embed_tokens)),
      blocks_(std::move(blocks)),
      final_norm_(std::move(final_norm)) {
    Require(final_norm_.weight().shape()[0] == embed_tokens_.embedding_dim(), "final_norm hidden size must match embedding dimension");
    if (!blocks_.empty()) {
        Require(blocks_[0].head_dim() != 0, "qwen causal lm blocks must expose valid head_dim");
    }
}

QwenCausalLM::QwenCausalLM(Embedding embed_tokens, std::vector<QwenBlock> blocks, RMSNormModule final_norm, Linear lm_head)
    : QwenCausalLM(std::move(embed_tokens), std::move(blocks), std::move(final_norm)) {
    Require(lm_head.input_dim() == embed_tokens_.embedding_dim(), "lm_head input dimension must match embedding dimension");
    lm_head_ = std::move(lm_head);
}

std::size_t QwenCausalLM::num_layers() const {
    return blocks_.size();
}

std::size_t QwenCausalLM::num_key_value_heads() const {
    return blocks_.empty() ? 0 : blocks_.front().num_key_value_heads();
}

std::size_t QwenCausalLM::head_dim() const {
    return blocks_.empty() ? 0 : blocks_.front().head_dim();
}

Tensor QwenCausalLM::ForwardHidden(const Tensor& token_ids, std::size_t position_offset) const {
    return ForwardHiddenRange(token_ids, 0, blocks_.size(), position_offset, true);
}

Tensor QwenCausalLM::ForwardHiddenCached(const Tensor& token_ids, TensorKVCache& cache, std::size_t position_offset) const {
    return ForwardHiddenCachedRange(token_ids, cache, 0, blocks_.size(), position_offset, true);
}

Tensor QwenCausalLM::ForwardLogits(const Tensor& token_ids, std::size_t position_offset) const {
    const Tensor hidden_states = ForwardHidden(token_ids, position_offset);
    return ForwardLogitsFromHidden(hidden_states);
}

Tensor QwenCausalLM::ForwardLogitsCached(const Tensor& token_ids, TensorKVCache& cache, std::size_t position_offset) const {
    const Tensor hidden_states = ForwardHiddenCached(token_ids, cache, position_offset);
    return ForwardLogitsFromHidden(hidden_states);
}

Tensor QwenCausalLM::ForwardHiddenRange(const Tensor& token_ids,
                                        std::size_t start_layer,
                                        std::size_t end_layer,
                                        std::size_t position_offset,
                                        bool apply_final_norm) const {
    RequireLayerRange(start_layer, end_layer, blocks_.size());
    Require(start_layer == 0, "token-based qwen layer ranges must start at layer 0");
    Tensor hidden_states = embed_tokens_.Forward(token_ids);
    hidden_states = RunBlockRange(blocks_, std::move(hidden_states), nullptr, start_layer, end_layer, position_offset);
    return apply_final_norm ? final_norm_.Forward(hidden_states) : hidden_states;
}

Tensor QwenCausalLM::ForwardHiddenCachedRange(const Tensor& token_ids,
                                              TensorKVCache& cache,
                                              std::size_t start_layer,
                                              std::size_t end_layer,
                                              std::size_t position_offset,
                                              bool apply_final_norm) const {
    RequireLayerRange(start_layer, end_layer, blocks_.size());
    Require(start_layer == 0, "token-based qwen layer ranges must start at layer 0");
    Tensor hidden_states = embed_tokens_.Forward(token_ids);
    hidden_states = RunBlockRange(blocks_, std::move(hidden_states), &cache, start_layer, end_layer, position_offset);
    return apply_final_norm ? final_norm_.Forward(hidden_states) : hidden_states;
}

Tensor QwenCausalLM::ForwardHiddenFromStatesRange(const Tensor& hidden_states,
                                                  std::size_t start_layer,
                                                  std::size_t end_layer,
                                                  std::size_t position_offset,
                                                  bool apply_final_norm) const {
    RequireLayerRange(start_layer, end_layer, blocks_.size());
    Tensor output = RunBlockRange(blocks_, hidden_states, nullptr, start_layer, end_layer, position_offset);
    return apply_final_norm ? final_norm_.Forward(output) : output;
}

Tensor QwenCausalLM::ForwardHiddenFromStatesCachedRange(const Tensor& hidden_states,
                                                        TensorKVCache& cache,
                                                        std::size_t start_layer,
                                                        std::size_t end_layer,
                                                        std::size_t position_offset,
                                                        bool apply_final_norm) const {
    RequireLayerRange(start_layer, end_layer, blocks_.size());
    Tensor output = RunBlockRange(blocks_, hidden_states, &cache, start_layer, end_layer, position_offset);
    return apply_final_norm ? final_norm_.Forward(output) : output;
}

Tensor QwenCausalLM::ForwardLogitsFromHidden(const Tensor& hidden_states) const {
    if (lm_head_.has_value()) {
        return lm_head_->Forward(hidden_states);
    }
    return ComputeTiedLogits(hidden_states);
}

Tensor QwenCausalLM::ComputeTiedLogits(const Tensor& hidden_states) const {
    Require(hidden_states.dtype() == DType::Float32, "lm hidden_states must be float32");
    Require(hidden_states.dim() == 3, "lm hidden_states must have shape [batch, seq, hidden]");
    RequireContiguous(hidden_states, "lm hidden_states");

    const std::size_t batch_size = hidden_states.shape()[0];
    const std::size_t sequence_length = hidden_states.shape()[1];
    const std::size_t hidden_size = hidden_states.shape()[2];
    const std::size_t vocab_size = embed_tokens_.vocab_size();
    const Tensor& embedding_weight = embed_tokens_.weight();

    Storage logits_storage = Storage::AllocateOwned(batch_size * sequence_length * vocab_size * sizeof(float));
    Tensor logits(std::move(logits_storage), DType::Float32, {batch_size, sequence_length, vocab_size});

    const float* hidden_data = hidden_states.data<const float>();
    const void* embedding_data = static_cast<const void*>(embedding_weight.data<const std::byte>());
    float* logits_data = logits.data<float>();

    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        for (std::size_t position = 0; position < sequence_length; ++position) {
            const float* hidden_vector = hidden_data + (batch * sequence_length + position) * hidden_size;
            float* logits_vector = logits_data + (batch * sequence_length + position) * vocab_size;
            for (std::size_t vocab_index = 0; vocab_index < vocab_size; ++vocab_index) {
                const std::size_t embedding_base = vocab_index * hidden_size;
                float value = 0.0f;
                for (std::size_t hidden_index = 0; hidden_index < hidden_size; ++hidden_index) {
                    value += hidden_vector[hidden_index] * DTypeReadFloat(embedding_data, embedding_weight.dtype(), embedding_base + hidden_index);
                }
                logits_vector[vocab_index] = value;
            }
        }
    }

    return logits;
}