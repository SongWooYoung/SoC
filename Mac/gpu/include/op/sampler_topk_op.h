#pragma once

#include <cstdint>
#include <string>

#include "kernel/pipeline_cache.h"
#include "tensor/device_tensor.h"

namespace soc::gpu {

class BufferArena;
class MetalContext;

struct SamplerTopKParams {
    static constexpr std::uint32_t kMaxTopK = 64;

    std::uint32_t row_count = 0;
    std::uint32_t row_size = 0;
    std::uint32_t top_k = 1;
};

class SamplerTopKOp {
public:
    static bool Run(const MetalContext& context,
                    PipelineCache* pipeline_cache,
                    const DeviceTensor& logits,
                    const DeviceTensor& top_values,
                    const DeviceTensor& top_indices,
                    const SamplerTopKParams& params,
                    BufferArena* temporary_arena,
                    std::string* error_message);
};

}  // namespace soc::gpu