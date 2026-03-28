#pragma once

#include <cstdint>
#include <string>

#include "kernel/pipeline_cache.h"
#include "tensor/device_tensor.h"

namespace soc::gpu {

class BufferArena;
class MetalContext;

struct MatMulParams {
    std::uint32_t row_count = 0;
    std::uint32_t inner_dim = 0;
    std::uint32_t column_count = 0;
    bool use_bias = false;
    bool decode_mode = false;
    bool transpose_rhs = false;
    bool enable_silu = false;
    bool add_residual = false;
    std::uint32_t preferred_threadgroup_width = 0;
    std::uint32_t preferred_threadgroup_height = 0;
    std::uint32_t preferred_tile_columns = 0;
    std::uint32_t preferred_tile_rows = 0;
};

struct MatMulExecutionPolicy {
    std::uint32_t threadgroup_width = 1;
    std::uint32_t threadgroup_height = 1;
    std::uint32_t tile_columns = 1;
    std::uint32_t tile_rows = 1;
    bool use_tiled_kernel = false;
};

class MatMulOp {
public:
    static MatMulExecutionPolicy SelectPolicy(const MatMulParams& params,
                                              std::uint32_t row_count,
                                              std::uint32_t column_count,
                                              std::uint32_t inner_dim);

    static bool Run(const MetalContext& context,
                    PipelineCache* pipeline_cache,
                    const DeviceTensor& lhs,
                    const DeviceTensor& rhs,
                    const DeviceTensor* bias,
                    const DeviceTensor* residual,
                    const DeviceTensor& output,
                    const MatMulParams& params,
                    BufferArena* temporary_arena,
                    std::string* error_message);
};

}  // namespace soc::gpu