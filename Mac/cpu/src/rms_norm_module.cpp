#include "header/rms_norm_module.h"

#include "header/dtype.h"

#include <cmath>
#include <stdexcept>

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
}

RMSNormModule::RMSNormModule(Tensor weight, float epsilon) : weight_(std::move(weight)), epsilon_(epsilon) {
    Require(DTypeIsFloatLike(weight_.dtype()), "rms norm weight must be float32, float16, or bfloat16");
    Require(weight_.dim() == 1, "rms norm weight must have shape [hidden_size]");
    Require(epsilon_ > 0.0f, "rms norm epsilon must be positive");
    RequireContiguous(weight_, "rms norm weight");
}

const Tensor& RMSNormModule::weight() const {
    return weight_;
}

float RMSNormModule::epsilon() const {
    return epsilon_;
}

Tensor RMSNormModule::Forward(const Tensor& input) const {
    Require(input.dtype() == DType::Float32, "rms norm input must be float32");
    Require(input.dim() >= 1, "rms norm input must have at least one dimension");
    Require(input.shape().back() == weight_.shape()[0], "rms norm input last dimension must match weight size");
    RequireContiguous(input, "rms norm input");

    Storage output_storage = Storage::AllocateOwned(input.numel() * sizeof(float));
    Tensor output(std::move(output_storage), DType::Float32, input.shape());

    const std::size_t hidden_size = weight_.shape()[0];
    const std::size_t rows = input.numel() / hidden_size;
    const float* input_data = input.data<const float>();
    const void* weight_data = static_cast<const void*>(weight_.data<const std::byte>());
    float* output_data = output.data<float>();

    for (std::size_t row = 0; row < rows; ++row) {
        const float* input_row = input_data + row * hidden_size;
        float* output_row = output_data + row * hidden_size;

        float mean_square = 0.0f;
        for (std::size_t index = 0; index < hidden_size; ++index) {
            mean_square += input_row[index] * input_row[index];
        }
        mean_square /= static_cast<float>(hidden_size);
        const float scale = 1.0f / std::sqrt(mean_square + epsilon_);

        for (std::size_t index = 0; index < hidden_size; ++index) {
            output_row[index] = input_row[index] * scale * DTypeReadFloat(weight_data, weight_.dtype(), index);
        }
    }

    return output;
}