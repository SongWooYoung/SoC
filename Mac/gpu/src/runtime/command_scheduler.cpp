#include "runtime/command_scheduler.h"

namespace soc::gpu {

bool CommandScheduler::RunPrefill(const MetalContext& context,
                                  PipelineCache* pipeline_cache,
                                  const QwenCausalLM& model,
                                  const DeviceTensor& token_ids,
                                  KVCache* kv_cache,
                                  const DeviceTensor& logits_output,
                                  BufferArena* temporary_arena,
                                  std::size_t position_offset,
                                  std::string* error_message) const {
    return model.ForwardLogitsCached(context,
                                     pipeline_cache,
                                     token_ids,
                                     kv_cache,
                                     logits_output,
                                     temporary_arena,
                                     position_offset,
                                     error_message);
}

bool CommandScheduler::RunDecode(const MetalContext& context,
                                 PipelineCache* pipeline_cache,
                                 const QwenCausalLM& model,
                                 const DeviceTensor& token_ids,
                                 KVCache* kv_cache,
                                 const DeviceTensor& logits_output,
                                 BufferArena* temporary_arena,
                                 std::size_t position_offset,
                                 std::string* error_message) const {
    return model.ForwardLogitsCached(context,
                                     pipeline_cache,
                                     token_ids,
                                     kv_cache,
                                     logits_output,
                                     temporary_arena,
                                     position_offset,
                                     error_message);
}

}  // namespace soc::gpu