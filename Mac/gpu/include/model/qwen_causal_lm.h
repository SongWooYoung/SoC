#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "module/qwen_block.h"
#include "runtime/kv_cache.h"
#include "tensor/device_tensor.h"

namespace soc::gpu {

class BufferArena;
class MetalContext;

struct QwenCausalLMWeights {
    DeviceTensor embed_tokens_weight;
    std::vector<QwenBlockWeights> blocks;
    DeviceTensor final_norm_weight;
    DeviceTensor lm_head_weight;
    bool tie_word_embeddings = true;
};

struct QwenCausalLMParams {
    std::size_t vocab_size = 0;
    std::size_t hidden_size = 0;
    std::size_t num_hidden_layers = 0;
    std::size_t num_attention_heads = 0;
    std::size_t num_key_value_heads = 0;
    std::size_t head_dim = 0;
    std::size_t intermediate_size = 0;
    std::size_t max_position_embeddings = 0;
    float rope_theta = 1000000.0f;
    float rms_norm_eps = 1.0e-6f;
};

class QwenCausalLM {
public:
    QwenCausalLM(QwenCausalLMWeights weights, QwenCausalLMParams params);

    std::size_t num_layers() const;
    std::size_t num_key_value_heads() const;
    std::size_t head_dim() const;
    std::size_t vocab_size() const;
    std::size_t max_position_embeddings() const;

    bool ForwardHidden(const MetalContext& context,
                       PipelineCache* pipeline_cache,
                       const DeviceTensor& token_ids,
                       const DeviceTensor& output,
                       BufferArena* temporary_arena,
                       std::size_t position_offset,
                       std::string* error_message) const;

    bool ForwardHiddenCached(const MetalContext& context,
                             PipelineCache* pipeline_cache,
                             const DeviceTensor& token_ids,
                             KVCache* kv_cache,
                             const DeviceTensor& output,
                             BufferArena* temporary_arena,
                             std::size_t position_offset,
                             std::string* error_message) const;

    bool ForwardLogits(const MetalContext& context,
                       PipelineCache* pipeline_cache,
                       const DeviceTensor& token_ids,
                       const DeviceTensor& logits_output,
                       BufferArena* temporary_arena,
                       std::size_t position_offset,
                       std::string* error_message) const;

    bool ForwardLogitsCached(const MetalContext& context,
                             PipelineCache* pipeline_cache,
                             const DeviceTensor& token_ids,
                             KVCache* kv_cache,
                             const DeviceTensor& logits_output,
                             BufferArena* temporary_arena,
                             std::size_t position_offset,
                             std::string* error_message) const;

    const QwenCausalLMWeights& weights() const;
    const QwenCausalLMParams& params() const;

private:
    bool ComputeLogitsFromHidden(const MetalContext& context,
                                 PipelineCache* pipeline_cache,
                                 const DeviceTensor& hidden_states,
                                 const DeviceTensor& logits_output,
                                 BufferArena* temporary_arena,
                                 std::string* error_message) const;

    QwenCausalLMWeights weights_;
    QwenCausalLMParams params_;
};

}  // namespace soc::gpu