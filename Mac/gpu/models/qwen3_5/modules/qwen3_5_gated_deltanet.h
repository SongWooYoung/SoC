#pragma once

#include <cstddef>

#include "tensor/device_tensor.h"

namespace soc::gpu::models::qwen3_5 {

struct Qwen3_5GatedDeltaNetParams {
    std::size_t linear_num_key_heads = 0;
    std::size_t linear_num_value_heads = 0;
    std::size_t linear_key_head_dim = 0;
    std::size_t linear_value_head_dim = 0;
    std::size_t conv_kernel_dim = 0;
};

struct Qwen3_5GatedDeltaNetWeights {
    DeviceTensor norm_weight;
    DeviceTensor in_proj_qkv_weight;
    DeviceTensor in_proj_z_weight;
    DeviceTensor in_proj_a_weight;
    DeviceTensor in_proj_b_weight;
    DeviceTensor out_proj_weight;
    DeviceTensor conv1d_weight;
    DeviceTensor a_log;
    DeviceTensor dt_bias;
};

}  // namespace soc::gpu::models::qwen3_5
