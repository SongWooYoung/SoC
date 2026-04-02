#include "models/qwen3_5/qwen3_5_state_layout.h"

namespace soc::gpu::models::qwen3_5 {
namespace {

std::size_t StateElementBytesForDType(const std::string& dtype) {
    if (dtype == "float16" || dtype == "bfloat16") {
        return 2;
    }
    return sizeof(float);
}

}  // namespace

Qwen3_5StateLayout BuildStateLayout(const Qwen3_5ArchitectureSpec& spec) {
    Qwen3_5StateLayout layout;
    layout.recurrent_state_dtype = spec.recurrent_state_dtype;
    layout.recurrent_state_element_bytes = StateElementBytesForDType(spec.recurrent_state_dtype);
    for (std::size_t layer_index = 0; layer_index < spec.layer_types.size(); ++layer_index) {
        if (spec.layer_types[layer_index] == Qwen3_5LayerType::kGatedAttention) {
            layout.full_attention_layer_count += 1;
            continue;
        }

        Qwen3_5DeltaNetLayerStateLayout layer;
        layer.layer_index = layer_index;
        layer.state_dtype = layout.recurrent_state_dtype;
        layer.state_element_bytes = layout.recurrent_state_element_bytes;
        layer.matrix_count = spec.linear_num_key_heads;
        layer.matrix_rows = spec.linear_key_head_dim;
        layer.matrix_cols = spec.linear_value_head_dim;
        layer.matrix_state_bytes =
            layer.matrix_count * layer.matrix_rows * layer.matrix_cols * layer.state_element_bytes;
        layer.conv_channel_count = spec.linear_num_value_heads;
        layer.conv_kernel_dim = spec.linear_conv_kernel_dim;
        layer.conv_state_bytes =
            layer.conv_channel_count * spec.linear_value_head_dim * layer.conv_kernel_dim * layer.state_element_bytes;
        layer.total_bytes = layer.matrix_state_bytes + layer.conv_state_bytes;
        layout.total_recurrent_state_bytes += layer.total_bytes;
        layout.deltanet_layers.push_back(layer);
    }
    return layout;
}

}  // namespace soc::gpu::models::qwen3_5
