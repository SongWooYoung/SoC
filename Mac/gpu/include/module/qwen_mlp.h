#pragma once

#include <cstddef>
#include <string>

#include "kernel/pipeline_cache.h"
#include "op/affine_qmm_op.h"
#include "tensor/device_tensor.h"

namespace soc::gpu {

class BufferArena;
class CommandStream;
class MetalContext;

struct QwenMlpWeights {
    DeviceTensor gate_proj_weight;
    AffineQmmWeight gate_proj_q4_weight;
    DeviceTensor up_proj_weight;
    AffineQmmWeight up_proj_q4_weight;
    DeviceTensor down_proj_weight;
    AffineQmmWeight down_proj_q4_weight;
};

struct QwenMlpParams {
    std::size_t intermediate_size = 0;
    bool add_residual = false;
};

class QwenMLP {
public:
    static bool Run(const MetalContext& context,
                    PipelineCache* pipeline_cache,
                    const DeviceTensor& input,
                    const DeviceTensor* residual,
                    const QwenMlpWeights& weights,
                    const DeviceTensor& output,
                    const QwenMlpParams& params,
                    BufferArena* temporary_arena,
                    CommandStream* stream,
                    std::string* error_message);
};

}  // namespace soc::gpu
