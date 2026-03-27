#include "header/rope.h"

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
}

RoPE::RoPE(std::size_t head_dim, double rope_theta) : head_dim_(head_dim), rope_theta_(rope_theta) {
    Require(head_dim_ != 0, "rope head_dim must be non-zero");
    Require(head_dim_ % 2 == 0, "rope head_dim must be even");
    Require(rope_theta_ > 0.0, "rope theta must be positive");
}

std::size_t RoPE::head_dim() const {
    return head_dim_;
}

double RoPE::rope_theta() const {
    return rope_theta_;
}

Tensor RoPE::Forward(const Tensor& input, std::size_t position_offset) const {
    Require(input.dtype() == DType::Float32, "rope input must be float32");
    Require(input.dim() == 3 || input.dim() == 4, "rope input must have shape [seq, heads, head_dim] or [batch, seq, heads, head_dim]");
    Require(input.shape().back() == head_dim_, "rope input last dimension must match configured head_dim");
    RequireContiguous(input, "rope input");

    Storage output_storage = Storage::AllocateOwned(input.numel() * sizeof(float));
    Tensor output(std::move(output_storage), DType::Float32, input.shape());

    const float* input_data = input.data<const float>();
    float* output_data = output.data<float>();
    for (std::size_t index = 0; index < input.numel(); ++index) {
        output_data[index] = input_data[index];
    }

    ApplyInPlace(output, position_offset);
    return output;
}

void RoPE::ApplyInPlace(Tensor& input, std::size_t position_offset) const {
    Require(input.dtype() == DType::Float32, "rope input must be float32");
    Require(input.dim() == 3 || input.dim() == 4, "rope input must have shape [seq, heads, head_dim] or [batch, seq, heads, head_dim]");
    Require(input.shape().back() == head_dim_, "rope input last dimension must match configured head_dim");
    RequireContiguous(input, "rope input");

    const std::vector<std::size_t>& shape = input.shape();
    const std::size_t batch_size = input.dim() == 4 ? shape[0] : 1;
    const std::size_t sequence_length = input.dim() == 4 ? shape[1] : shape[0];
    const std::size_t num_heads = input.dim() == 4 ? shape[2] : shape[1];

    float* data = input.data<float>();
    const std::size_t batch_stride = sequence_length * num_heads * head_dim_;
    const std::size_t sequence_stride = num_heads * head_dim_;
    const std::size_t half_dim = head_dim_ / 2;

    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        for (std::size_t position = 0; position < sequence_length; ++position) {
            const double absolute_position = static_cast<double>(position_offset + position);
            for (std::size_t head = 0; head < num_heads; ++head) {
                float* head_base = data + batch * batch_stride + position * sequence_stride + head * head_dim_;
                for (std::size_t pair_index = 0; pair_index < half_dim; ++pair_index) {
                    const double exponent = static_cast<double>(pair_index * 2) / static_cast<double>(head_dim_);
                    const double angle = absolute_position / std::pow(rope_theta_, exponent);
                    const float cosine = static_cast<float>(std::cos(angle));
                    const float sine = static_cast<float>(std::sin(angle));

                    const float first = head_base[pair_index];
                    const float second = head_base[pair_index + half_dim];
                    head_base[pair_index] = first * cosine - second * sine;
                    head_base[pair_index + half_dim] = first * sine + second * cosine;
                }
            }
        }
    }
}