#include "runtime/command_scheduler.h"

#include "tensor/device_tensor.h"

#include <cstdlib>
#include <vector>

namespace soc::gpu {
namespace {

bool UseExperimentalPrebuiltDecodePlan() {
    const char* value = std::getenv("SOC_GPU_ENABLE_EXPERIMENTAL_PREBUILT_DECODE_PLAN");
    return value != nullptr && std::string(value) == "1";
}

bool IsSingleTokenTensor(const DeviceTensor& token_ids) {
    return token_ids.IsValid() &&
           token_ids.GetDesc().Rank() == 1 &&
           token_ids.GetDesc().GetShape() == std::vector<std::size_t>{1};
}

}  // namespace

bool CommandScheduler::RunPrefill(const MetalContext& context,
                                  PipelineCache* pipeline_cache,
                                  const ModelRunner& model,
                                  const DeviceTensor& token_ids,
                                  KVCache* kv_cache,
                                  const DeviceTensor& logits_output,
                                  BufferArena* temporary_arena,
                                  std::size_t position_offset,
                                  std::string* error_message) const {
    last_decode_plan_run_stats_ = {};
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
                                 const ModelRunner& model,
                                 const DeviceTensor& token_ids,
                                 KVCache* kv_cache,
                                 const DeviceTensor& logits_output,
                                 BufferArena* temporary_arena,
                                 std::size_t position_offset,
                                 std::string* error_message) const {
    last_decode_plan_run_stats_ = {};
    const DecodePlanner* decode_planner = model.GetDecodePlanner();

    const bool use_prebuilt_plan =
        UseExperimentalPrebuiltDecodePlan() && decode_planner != nullptr && IsSingleTokenTensor(token_ids);
    if (use_prebuilt_plan) {
        if (!decode_planner->RunDecode(context,
                                       pipeline_cache,
                                       model,
                                       token_ids,
                                       kv_cache,
                                       logits_output,
                                       temporary_arena,
                                       position_offset,
                                       error_message)) {
            return false;
        }
        last_decode_plan_run_stats_ = decode_planner->last_run_stats();
        return true;
    }

    return model.ForwardLogitsCached(context,
                                     pipeline_cache,
                                     token_ids,
                                     kv_cache,
                                     logits_output,
                                     temporary_arena,
                                     position_offset,
                                     error_message);
}

const DecodePlanRunStats& CommandScheduler::last_decode_plan_run_stats() const {
    return last_decode_plan_run_stats_;
}

}  // namespace soc::gpu
