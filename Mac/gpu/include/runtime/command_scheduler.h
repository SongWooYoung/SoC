#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "model/qwen_causal_lm.h"

namespace soc::gpu {

class BufferArena;
class MetalContext;

class CommandScheduler {
public:
    bool RunPrefill(const MetalContext& context,
                    PipelineCache* pipeline_cache,
                    const QwenCausalLM& model,
                    const DeviceTensor& token_ids,
                    KVCache* kv_cache,
                    const DeviceTensor& logits_output,
                    BufferArena* temporary_arena,
                    std::size_t position_offset,
                    std::string* error_message) const;

    bool RunDecode(const MetalContext& context,
                   PipelineCache* pipeline_cache,
                   const QwenCausalLM& model,
                   const DeviceTensor& token_ids,
                   KVCache* kv_cache,
                   const DeviceTensor& logits_output,
                   BufferArena* temporary_arena,
                   std::size_t position_offset,
                   std::string* error_message) const;
};

}  // namespace soc::gpu