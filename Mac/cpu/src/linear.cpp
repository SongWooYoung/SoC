#include "header/linear.h"

#include "header/dtype.h"

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

Linear::Linear(Tensor weight) : weight_(std::move(weight)) {
    Require(DTypeIsFloatLike(weight_.dtype()), "linear weight must be float32, float16, or bfloat16");
    Require(weight_.dim() == 2, "linear weight must have shape [out_features, in_features]");
    RequireContiguous(weight_, "linear weight");
}

Linear::Linear(Tensor weight, Tensor bias) : Linear(std::move(weight)) {
    Require(DTypeIsFloatLike(bias.dtype()), "linear bias must be float32, float16, or bfloat16");
    Require(bias.dim() == 1, "linear bias must have shape [out_features]");
    Require(bias.shape()[0] == output_dim(), "linear bias size must match weight output dimension");
    RequireContiguous(bias, "linear bias");
    bias_ = std::move(bias);
}

const Tensor& Linear::weight() const {
    return weight_;
}

const std::optional<Tensor>& Linear::bias() const {
    return bias_;
}

std::size_t Linear::input_dim() const {
    return weight_.shape()[1];
}

std::size_t Linear::output_dim() const {
    return weight_.shape()[0];
}

Tensor Linear::Forward(const Tensor& input) const {
    Require(input.dtype() == DType::Float32, "linear input must be float32");
    Require(input.dim() >= 1, "linear input must have at least one dimension");
    Require(input.shape().back() == input_dim(), "linear input last dimension must match weight input dimension");
    RequireContiguous(input, "linear input");

    std::vector<std::size_t> output_shape = input.shape();
    output_shape.back() = output_dim();

    const std::size_t batch_elements = input.numel() / input_dim();
    Storage output_storage = Storage::AllocateOwned(batch_elements * output_dim() * sizeof(float));
    Tensor output(std::move(output_storage), DType::Float32, output_shape);

    const float* input_data = input.data<const float>();
    const void* weight_data = static_cast<const void*>(weight_.data<const std::byte>());
    const void* bias_data = bias_.has_value() ? static_cast<const void*>(bias_->data<const std::byte>()) : nullptr;
    float* output_data = output.data<float>();

    for (std::size_t row = 0; row < batch_elements; ++row) {
        const float* input_row = input_data + row * input_dim();
        float* output_row = output_data + row * output_dim();

        for (std::size_t out_index = 0; out_index < output_dim(); ++out_index) {
            const std::size_t weight_row_base = out_index * input_dim();
            float value = bias_data == nullptr ? 0.0f : DTypeReadFloat(bias_data, bias_->dtype(), out_index);
            for (std::size_t in_index = 0; in_index < input_dim(); ++in_index) {
                value += input_row[in_index] * DTypeReadFloat(weight_data, weight_.dtype(), weight_row_base + in_index);
            }
            output_row[out_index] = value;
        }
    }

    return output;
}