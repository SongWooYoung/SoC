#pragma once

#include <cstddef>
#include <string>

#include "runtime/decode_planner.h"
#include "runtime/model_runner.h"

namespace soc::gpu {

class BufferArena;
class MetalContext;

class CommandScheduler {
public:
    CommandScheduler() = default;
    CommandScheduler(const CommandScheduler&) = default;
    CommandScheduler& operator=(const CommandScheduler&) = default;
    CommandScheduler(CommandScheduler&&) noexcept = default;
    CommandScheduler& operator=(CommandScheduler&&) noexcept = default;

    bool RunPrefill(const MetalContext& context,
                    PipelineCache* pipeline_cache,
                    const ModelRunner& model,
                    const DeviceTensor& token_ids,
                    KVCache* kv_cache,
                    const DeviceTensor& logits_output,
                    BufferArena* temporary_arena,
                    std::size_t position_offset,
                    std::string* error_message) const;

    bool RunDecode(const MetalContext& context,
                   PipelineCache* pipeline_cache,
                   const ModelRunner& model,
                   const DeviceTensor& token_ids,
                   KVCache* kv_cache,
                   const DeviceTensor& logits_output,
                   BufferArena* temporary_arena,
                   std::size_t position_offset,
                   std::string* error_message) const;

    const DecodePlanRunStats& last_decode_plan_run_stats() const;

    mutable DecodePlanRunStats last_decode_plan_run_stats_;
};

}  // namespace soc::gpu
