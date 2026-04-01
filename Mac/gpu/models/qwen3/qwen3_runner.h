#pragma once

#include "model/qwen_causal_lm.h"
#include "models/qwen3/qwen3_decode_planner.h"
#include "runtime/layer_range_model_runner.h"
#include "runtime/model_runner.h"

namespace soc::gpu::models::qwen3 {

class Qwen3Runner final : public ModelRunner, public LayerRangeModelRunner {
public:
    explicit Qwen3Runner(QwenCausalLM model);

    std::size_t num_layers() const override;
    std::size_t hidden_size() const override;
    std::size_t num_key_value_heads() const override;
    std::size_t head_dim() const override;
    std::size_t vocab_size() const override;
    std::size_t max_position_embeddings() const override;

    bool ForwardHiddenCachedRange(const MetalContext& context,
                                  PipelineCache* pipeline_cache,
                                  const DeviceTensor& token_ids,
                                  KVCache* kv_cache,
                                  const DeviceTensor& output,
                                  BufferArena* temporary_arena,
                                  const LayerRangeExecutionOptions& options,
                                  std::string* error_message) const override;

    bool ForwardHiddenFromStatesCachedRange(const MetalContext& context,
                                            PipelineCache* pipeline_cache,
                                            const DeviceTensor& hidden_states,
                                            KVCache* kv_cache,
                                            const DeviceTensor& output,
                                            BufferArena* temporary_arena,
                                            const LayerRangeExecutionOptions& options,
                                            std::string* error_message) const override;

    bool ForwardLogitsCached(const MetalContext& context,
                             PipelineCache* pipeline_cache,
                             const DeviceTensor& token_ids,
                             KVCache* kv_cache,
                             const DeviceTensor& logits_output,
                             BufferArena* temporary_arena,
                             std::size_t position_offset,
                             std::string* error_message) const override;

    bool ForwardLogitsFromHidden(const MetalContext& context,
                                 PipelineCache* pipeline_cache,
                                 const DeviceTensor& hidden_states,
                                 const DeviceTensor& logits_output,
                                 BufferArena* temporary_arena,
                                 std::string* error_message) const override;

    DecodePlanner* GetDecodePlanner() override;
    const DecodePlanner* GetDecodePlanner() const override;

    const QwenCausalLM& model() const;
    QwenCausalLM&& ReleaseModel();

private:
    QwenCausalLM model_;
    mutable Qwen3DecodePlanner decode_planner_;
};

}  // namespace soc::gpu::models::qwen3
