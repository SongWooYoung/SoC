#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "models/qwen3_5/qwen3_5_architecture.h"

namespace soc::gpu::models::qwen3_5 {

struct Qwen3_5DeltaNetLayerStateLayout {
    std::size_t layer_index = 0;
    std::string state_dtype = "float32";
    std::size_t state_element_bytes = 0;
    std::size_t matrix_count = 0;
    std::size_t matrix_rows = 0;
    std::size_t matrix_cols = 0;
    std::size_t matrix_state_bytes = 0;
    std::size_t conv_channel_count = 0;
    std::size_t conv_kernel_dim = 0;
    std::size_t conv_state_bytes = 0;
    std::size_t total_bytes = 0;
};

struct Qwen3_5StateLayout {
    std::vector<Qwen3_5DeltaNetLayerStateLayout> deltanet_layers;
    std::string recurrent_state_dtype = "float32";
    std::size_t recurrent_state_element_bytes = 0;
    std::size_t total_recurrent_state_bytes = 0;
    std::size_t full_attention_layer_count = 0;
};

Qwen3_5StateLayout BuildStateLayout(const Qwen3_5ArchitectureSpec& spec);

}  // namespace soc::gpu::models::qwen3_5
