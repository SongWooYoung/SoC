#pragma once

#include <string>

#include "op/matmul_op.h"

namespace soc::gpu {

class BufferArena;
class MetalContext;

enum class LinearActivation {
    kNone,
    kSiLU,
};

struct LinearParams {
    MatMulParams matmul;
    LinearActivation activation = LinearActivation::kNone;
    bool add_residual = false;
};

class LinearOp {
public:
    static bool Run(const MetalContext& context,
                    PipelineCache* pipeline_cache,
                    const DeviceTensor& input,
                    const DeviceTensor& weight,
                    const DeviceTensor* bias,
                    const DeviceTensor* residual,
                    const DeviceTensor& output,
                    const LinearParams& params,
                    BufferArena* temporary_arena,
                    std::string* error_message);
};

}  // namespace soc::gpu