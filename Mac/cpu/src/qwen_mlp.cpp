#include "header/qwen_mlp.h"

#include <cmath>
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

Tensor ApplySiLU(const Tensor& input) {
    Require(input.dtype() == DType::Float32, "SiLU input must be float32");
    RequireContiguous(input, "SiLU input");

    Storage output_storage = Storage::AllocateOwned(input.numel() * sizeof(float));
    Tensor output(std::move(output_storage), DType::Float32, input.shape());

    const float* input_data = input.data<const float>();
    float* output_data = output.data<float>();
    for (std::size_t index = 0; index < input.numel(); ++index) {
        const float value = input_data[index];
        output_data[index] = value / (1.0f + std::exp(-value));
    }

    return output;
}

Tensor HadamardProduct(const Tensor& left, const Tensor& right) {
    Require(left.dtype() == DType::Float32 && right.dtype() == DType::Float32, "Hadamard inputs must be float32");
    Require(left.shape() == right.shape(), "Hadamard inputs must have matching shape");
    RequireContiguous(left, "Hadamard left input");
    RequireContiguous(right, "Hadamard right input");

    Storage output_storage = Storage::AllocateOwned(left.numel() * sizeof(float));
    Tensor output(std::move(output_storage), DType::Float32, left.shape());

    const float* left_data = left.data<const float>();
    const float* right_data = right.data<const float>();
    float* output_data = output.data<float>();
    for (std::size_t index = 0; index < left.numel(); ++index) {
        output_data[index] = left_data[index] * right_data[index];
    }

    return output;
}
}

QwenMLP::QwenMLP(Linear gate_proj, Linear up_proj, Linear down_proj)
    : gate_proj_(std::move(gate_proj)),
      up_proj_(std::move(up_proj)),
      down_proj_(std::move(down_proj)) {
    Require(gate_proj_.output_dim() == up_proj_.output_dim(), "gate_proj and up_proj output dimensions must match");
    Require(down_proj_.input_dim() == gate_proj_.output_dim(), "down_proj input dimension must match intermediate size");
}

Tensor QwenMLP::Forward(const Tensor& hidden_states) const {
    Require(hidden_states.dtype() == DType::Float32, "mlp hidden_states must be float32");
    Require(hidden_states.dim() >= 1, "mlp hidden_states must have at least one dimension");

    const Tensor gate = gate_proj_.Forward(hidden_states);
    const Tensor up = up_proj_.Forward(hidden_states);
    const Tensor activated = ApplySiLU(gate);
    const Tensor gated = HadamardProduct(activated, up);
    return down_proj_.Forward(gated);
}