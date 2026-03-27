#include "header/qwen_block.h"

#include <stdexcept>
#include <string>

namespace {
void Require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void RequireContiguous(const Tensor& tensor, const char* name) {
    if (!tensor.is_contiguous()) {
        throw std::runtime_error(std::string(name) + " must be contiguous");
    }
}

Tensor AddTensors(const Tensor& left, const Tensor& right) {
    Require(left.dtype() == DType::Float32 && right.dtype() == DType::Float32, "residual add inputs must be float32");
    Require(left.shape() == right.shape(), "residual add inputs must have matching shape");
    RequireContiguous(left, "residual add left input");
    RequireContiguous(right, "residual add right input");

    Storage output_storage = Storage::AllocateOwned(left.numel() * sizeof(float));
    Tensor output(std::move(output_storage), DType::Float32, left.shape());
    const float* left_data = left.data<const float>();
    const float* right_data = right.data<const float>();
    float* output_data = output.data<float>();
    for (std::size_t index = 0; index < left.numel(); ++index) {
        output_data[index] = left_data[index] + right_data[index];
    }
    return output;
}
}

QwenBlock::QwenBlock(RMSNormModule input_norm, QwenAttention attention, RMSNormModule post_attention_norm, QwenMLP mlp)
    : input_norm_(std::move(input_norm)),
      attention_(std::move(attention)),
      post_attention_norm_(std::move(post_attention_norm)),
      mlp_(std::move(mlp)) {}

std::size_t QwenBlock::num_key_value_heads() const {
    return attention_.num_key_value_heads();
}

std::size_t QwenBlock::head_dim() const {
    return attention_.head_dim();
}

Tensor QwenBlock::Forward(const Tensor& hidden_states, std::size_t position_offset) const {
    Require(hidden_states.dtype() == DType::Float32, "qwen block hidden_states must be float32");
    Require(hidden_states.dim() == 3, "qwen block hidden_states must have shape [batch, seq, hidden]");

    const Tensor normed_attention_input = input_norm_.Forward(hidden_states);
    const Tensor attention_output = attention_.Forward(normed_attention_input, position_offset);
    const Tensor residual_after_attention = AddTensors(hidden_states, attention_output);
    const Tensor normed_mlp_input = post_attention_norm_.Forward(residual_after_attention);
    const Tensor mlp_output = mlp_.Forward(normed_mlp_input);
    return AddTensors(residual_after_attention, mlp_output);
}

Tensor QwenBlock::ForwardCached(const Tensor& hidden_states, TensorKVCache& cache, std::size_t layer_index, std::size_t position_offset) const {
    Require(hidden_states.dtype() == DType::Float32, "qwen block hidden_states must be float32");
    Require(hidden_states.dim() == 3, "qwen block hidden_states must have shape [batch, seq, hidden]");

    const Tensor normed_attention_input = input_norm_.Forward(hidden_states);
    const Tensor attention_output = attention_.ForwardCached(normed_attention_input, cache, layer_index, position_offset);
    const Tensor residual_after_attention = AddTensors(hidden_states, attention_output);
    const Tensor normed_mlp_input = post_attention_norm_.Forward(residual_after_attention);
    const Tensor mlp_output = mlp_.Forward(normed_mlp_input);
    return AddTensors(residual_after_attention, mlp_output);
}