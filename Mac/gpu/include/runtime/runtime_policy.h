#pragma once

#include <cstddef>
#include <cstdint>

namespace soc::gpu {

class MetalContext;

struct RuntimePolicy {
    std::size_t prefill_step_size = 0;
    std::uint64_t recommended_max_working_set_size = 0;
    std::uint64_t working_set_budget_bytes = 0;
    std::size_t command_stream_encoder_budget = 0;
};

RuntimePolicy ResolveRuntimePolicy(const MetalContext& context,
                                  std::size_t configured_prefill_step_size);
std::size_t ResolveCommandStreamEncoderBudget(const MetalContext& context);

}  // namespace soc::gpu