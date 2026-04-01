#include "models/qwen3/qwen3_runner.h"

#include <utility>

namespace soc::gpu::models::qwen3 {

Qwen3Runner::Qwen3Runner(QwenCausalLM model) : model_(std::move(model)) {}

std::size_t Qwen3Runner::num_layers() const { return model_.num_layers(); }
std::size_t Qwen3Runner::hidden_size() const { return model_.params().hidden_size; }
std::size_t Qwen3Runner::num_key_value_heads() const { return model_.num_key_value_heads(); }
std::size_t Qwen3Runner::head_dim() const { return model_.head_dim(); }
std::size_t Qwen3Runner::vocab_size() const { return model_.vocab_size(); }
std::size_t Qwen3Runner::max_position_embeddings() const { return model_.max_position_embeddings(); }

bool Qwen3Runner::ForwardHiddenCachedRange(const MetalContext& context,
                                           PipelineCache* pipeline_cache,
                                           const DeviceTensor& token_ids,
                                           KVCache* kv_cache,
                                           const DeviceTensor& output,
                                           BufferArena* temporary_arena,
                                           const LayerRangeExecutionOptions& options,
                                           std::string* error_message) const {
    return model_.ForwardHiddenCachedRange(context,
                                           pipeline_cache,
                                           token_ids,
                                           kv_cache,
                                           output,
                                           temporary_arena,
                                           options.position_offset,
                                           options.start_layer,
                                           options.end_layer,
                                           options.apply_final_norm,
                                           RangeCommandStreamMode::kDefault,
                                           error_message);
}

bool Qwen3Runner::ForwardHiddenFromStatesCachedRange(const MetalContext& context,
                                                     PipelineCache* pipeline_cache,
                                                     const DeviceTensor& hidden_states,
                                                     KVCache* kv_cache,
                                                     const DeviceTensor& output,
                                                     BufferArena* temporary_arena,
                                                     const LayerRangeExecutionOptions& options,
                                                     std::string* error_message) const {
    return model_.ForwardHiddenFromStatesCachedRange(context,
                                                     pipeline_cache,
                                                     hidden_states,
                                                     kv_cache,
                                                     output,
                                                     temporary_arena,
                                                     options.position_offset,
                                                     options.start_layer,
                                                     options.end_layer,
                                                     options.apply_final_norm,
                                                     RangeCommandStreamMode::kDefault,
                                                     error_message);
}

bool Qwen3Runner::ForwardLogitsCached(const MetalContext& context,
                                      PipelineCache* pipeline_cache,
                                      const DeviceTensor& token_ids,
                                      KVCache* kv_cache,
                                      const DeviceTensor& logits_output,
                                      BufferArena* temporary_arena,
                                      std::size_t position_offset,
                                      std::string* error_message) const {
    return model_.ForwardLogitsCached(context,
                                      pipeline_cache,
                                      token_ids,
                                      kv_cache,
                                      logits_output,
                                      temporary_arena,
                                      position_offset,
                                      error_message);
}

bool Qwen3Runner::ForwardLogitsFromHidden(const MetalContext& context,
                                          PipelineCache* pipeline_cache,
                                          const DeviceTensor& hidden_states,
                                          const DeviceTensor& logits_output,
                                          BufferArena* temporary_arena,
                                          std::string* error_message) const {
    return model_.ForwardLogitsFromHidden(context,
                                          pipeline_cache,
                                          hidden_states,
                                          logits_output,
                                          temporary_arena,
                                          error_message);
}

DecodePlanner* Qwen3Runner::GetDecodePlanner() { return &decode_planner_; }
const DecodePlanner* Qwen3Runner::GetDecodePlanner() const { return &decode_planner_; }

const QwenCausalLM& Qwen3Runner::model() const { return model_; }
QwenCausalLM&& Qwen3Runner::ReleaseModel() { return std::move(model_); }

}  // namespace soc::gpu::models::qwen3
