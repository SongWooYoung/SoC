#include "runtime/runtime_policy.h"

#include <cerrno>
#include <cstdlib>

#include "metal/metal_context.h"

namespace soc::gpu {
namespace {

std::size_t ParsePositiveSizeEnv(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return 0;
    }

    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || end == value || (end != nullptr && *end != '\0') || parsed == 0) {
        return 0;
    }
    return static_cast<std::size_t>(parsed);
}

std::uint64_t ResolveWorkingSetBudgetBytes(const MetalDeviceInfo& device_info) {
    const std::size_t budget_mb = ParsePositiveSizeEnv("SOC_GPU_WORKING_SET_BUDGET_MB");
    if (budget_mb > 0) {
        return static_cast<std::uint64_t>(budget_mb) * 1024ull * 1024ull;
    }
    return device_info.recommended_max_working_set_size;
}

}  // namespace

RuntimePolicy ResolveRuntimePolicy(const MetalContext& context,
                                  const std::size_t configured_prefill_step_size) {
    RuntimePolicy policy;
    policy.prefill_step_size = configured_prefill_step_size;
    policy.recommended_max_working_set_size = context.GetDeviceInfo().recommended_max_working_set_size;
    policy.working_set_budget_bytes = ResolveWorkingSetBudgetBytes(context.GetDeviceInfo());
    policy.command_stream_encoder_budget = ParsePositiveSizeEnv("SOC_GPU_COMMAND_STREAM_ENCODER_BUDGET");
    return policy;
}

std::size_t ResolveCommandStreamEncoderBudget(const MetalContext& context) {
    return ResolveRuntimePolicy(context, 0).command_stream_encoder_budget;
}

}  // namespace soc::gpu