#pragma once

#include <cstddef>

#include "tensor/device_tensor.h"

namespace soc::gpu::models::qwen3_5 {

struct Qwen3_5MlpParams {
    std::size_t intermediate_size = 0;
};

struct Qwen3_5MlpWeights {
    DeviceTensor gate_proj_weight;
    DeviceTensor up_proj_weight;
    DeviceTensor down_proj_weight;
};

}  // namespace soc::gpu::models::qwen3_5
