#pragma once

#include <cstdint>
#include <string>

#include "kernel/pipeline_cache.h"
#include "tensor/device_tensor.h"

namespace soc::gpu {

class BufferArena;
class CommandStream;
class MetalContext;

struct AffineQmmWeight {
    DeviceTensor qweight;
    DeviceTensor scales;
    DeviceTensor qbiases;
    std::uint32_t bits = 4;
    std::uint32_t group_size = 128;

    bool IsValid() const {
        return qweight.IsValid() && scales.IsValid() && qbiases.IsValid();
    }
};

struct AffineQmmParams {
    std::uint32_t row_count = 0;
    std::uint32_t inner_dim = 0;
    std::uint32_t column_count = 0;
    std::uint32_t output_row_stride = 0;
    bool enable_silu = false;
    bool add_residual = false;
    const char* profile_label = nullptr;
};

class AffineQmmOp {
public:
    static bool Run(const MetalContext& context,
                    PipelineCache* pipeline_cache,
                    const DeviceTensor& lhs,
                    const AffineQmmWeight& weight,
                    const DeviceTensor* residual,
                    const DeviceTensor& output,
                    const AffineQmmParams& params,
                    BufferArena* temporary_arena,
                    CommandStream* stream,
                    std::string* error_message);
};

}  // namespace soc::gpu
