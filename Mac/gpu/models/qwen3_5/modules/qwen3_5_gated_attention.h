#pragma once

#include <cstddef>

#include "tensor/device_tensor.h"

namespace soc::gpu::models::qwen3_5 {

struct Qwen3_5GatedAttentionParams {
    std::size_t num_attention_heads = 0;
    std::size_t num_key_value_heads = 0;
    std::size_t head_dim = 0;
    std::size_t rotary_dim = 0;
};

struct Qwen3_5GatedAttentionWeights {
    DeviceTensor q_proj_weight;
    DeviceTensor k_proj_weight;
    DeviceTensor v_proj_weight;
    DeviceTensor o_proj_weight;
    DeviceTensor q_norm_weight;
    DeviceTensor k_norm_weight;
};

}  // namespace soc::gpu::models::qwen3_5
