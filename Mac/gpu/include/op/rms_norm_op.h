#pragma once

#include <cstdint>
#include <string>

#include "kernel/pipeline_cache.h"
#include "tensor/device_tensor.h"

namespace soc::gpu {

class BufferArena;

struct RmsNormParams {
    float epsilon = 1.0e-5f;
    std::uint32_t row_size = 0;
    std::uint32_t row_count = 0;
};

class MetalContext;

class RmsNormOp {
public:
    static bool Run(const MetalContext& context,
                    PipelineCache* pipeline_cache,
                    const DeviceTensor& input,
                    const DeviceTensor& weight,
                    const DeviceTensor& output,
                    const RmsNormParams& params,
                    BufferArena* temporary_arena,
                    std::string* error_message);
};

}  // namespace soc::gpu