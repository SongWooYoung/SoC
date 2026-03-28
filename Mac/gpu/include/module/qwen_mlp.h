#pragma once

#include <cstddef>
#include <string>

#include "kernel/pipeline_cache.h"
#include "tensor/device_tensor.h"

namespace soc::gpu {

class BufferArena;
class MetalContext;

struct QwenMlpWeights {
    DeviceTensor gate_proj_weight;
    DeviceTensor up_proj_weight;
    DeviceTensor down_proj_weight;
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
                    std::string* error_message);
};

}  // namespace soc::gpu