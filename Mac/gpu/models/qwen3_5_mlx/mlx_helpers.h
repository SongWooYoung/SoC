#pragma once

#include <mlx/mlx.h>

#include <cmath>
#include <optional>
#include <string>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace qwen3_5_mlx {
namespace mx = mlx::core;

namespace mlx_helpers {

struct TensorParam {
    mx::array weight = mx::array(0.0f);
    std::optional<mx::array> scales;
    std::optional<mx::array> biases;
    int group_size = 64;
    int bits = 8;
    std::string mode = "affine";

    TensorParam() = default;
    TensorParam(const mx::array& dense_weight) : weight(dense_weight) {}
    TensorParam(mx::array&& dense_weight) : weight(std::move(dense_weight)) {}

    TensorParam& operator=(const mx::array& dense_weight) {
        weight = dense_weight;
        scales.reset();
        biases.reset();
        group_size = 64;
        bits = 8;
        mode = "affine";
        return *this;
    }

    TensorParam& operator=(mx::array&& dense_weight) {
        weight = std::move(dense_weight);
        scales.reset();
        biases.reset();
        group_size = 64;
        bits = 8;
        mode = "affine";
        return *this;
    }

    bool is_quantized() const {
        return scales.has_value();
    }
};

inline mx::array silu(const mx::array& x) {
    return x * mx::sigmoid(x);
}

inline mx::array softplus(const mx::array& x) {
    return mx::log(1.0f + mx::exp(x));
}

inline mx::array swiglu(const mx::array& gate, const mx::array& x) {
    return silu(gate) * x;
}

inline mx::array linear(
    const mx::array& x,
    const mx::array& weight,
    const std::optional<mx::array>& bias = std::nullopt) {
    auto out = mx::matmul(x, mx::transpose(weight));
    if (bias.has_value()) {
        out = out + *bias;
    }
    return out;
}

inline mx::array linear(
    const mx::array& x,
    const TensorParam& weight,
    const std::optional<mx::array>& bias = std::nullopt) {
    auto out = weight.is_quantized()
        ? mx::quantized_matmul(
            x,
            weight.weight,
            *weight.scales,
            weight.biases,
            true,
            weight.group_size,
            weight.bits,
            weight.mode)
        : mx::matmul(x, mx::transpose(weight.weight));
    if (bias.has_value()) {
        out = out + *bias;
    }
    return out;
}

inline mx::array embedding(const mx::array& weight, const mx::array& indices) {
    return mx::take(weight, indices, 0);
}

inline mx::array embedding(const TensorParam& weight, const mx::array& indices) {
    if (!weight.is_quantized()) {
        return mx::take(weight.weight, indices, 0);
    }

    auto gathered_weight = mx::take(weight.weight, indices, 0);
    auto gathered_scales = mx::take(*weight.scales, indices, 0);
    std::optional<mx::array> gathered_biases = std::nullopt;
    if (weight.biases.has_value()) {
        gathered_biases = mx::take(*weight.biases, indices, 0);
    }
    return mx::dequantize(
        gathered_weight,
        gathered_scales,
        gathered_biases,
        weight.group_size,
        weight.bits,
        weight.mode);
}

inline mx::array rotate_half(const mx::array& x) {
    auto parts = mx::split(x, 2, x.ndim() - 1);
    return mx::concatenate({-parts[1], parts[0]}, x.ndim() - 1);
}

inline mx::array compute_g(
    const mx::array& A_log,
    const mx::array& a,
    const mx::array& dt_bias) {
    auto A = mx::exp(mx::astype(A_log, mx::float32));
    return mx::exp(-A * softplus(a + dt_bias));
}

inline mx::array slice_axis(
    const mx::array& x,
    int axis,
    int start,
    int end,
    int stride = 1) {
    int ndim = x.ndim();
    if (axis < 0) {
        axis += ndim;
    }
    mx::Shape starts;
    mx::Shape ends;
    mx::Shape strides;
    starts.reserve(ndim);
    ends.reserve(ndim);
    strides.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
        starts.push_back(i == axis ? start : 0);
        ends.push_back(i == axis ? end : x.shape(i));
        strides.push_back(i == axis ? stride : 1);
    }
    return mx::slice(x, starts, ends, strides);
}

inline mx::array slice_last_dim(const mx::array& x, int start, int end) {
    return slice_axis(x, x.ndim() - 1, start, end);
}

inline mx::array take_last_tokens(const mx::array& x, int token_count) {
    return slice_axis(x, 1, x.shape(1) - token_count, x.shape(1));
}

inline mx::array adapt_last_dim(const mx::array& x, int target_dim) {
    int current_dim = x.shape(x.ndim() - 1);
    if (current_dim == target_dim) {
        return x;
    }
    if (current_dim > target_dim) {
        return slice_last_dim(x, 0, target_dim);
    }

    mx::Shape pad_shape;
    for (int i = 0; i < x.ndim(); ++i) {
        pad_shape.push_back(i == x.ndim() - 1 ? target_dim - current_dim : x.shape(i));
    }
    auto pad = mx::zeros(pad_shape, x.dtype());
    return mx::concatenate({x, pad}, x.ndim() - 1);
}

inline mx::array make_position_ids(int seq_len, int offset, int batch_size = 1) {
    std::vector<int> data(static_cast<size_t>(3 * batch_size * seq_len));
    for (int component = 0; component < 3; ++component) {
        for (int batch = 0; batch < batch_size; ++batch) {
            for (int index = 0; index < seq_len; ++index) {
                data[((component * batch_size + batch) * seq_len) + index] = offset + index;
            }
        }
    }
    return mx::array(data.data(), mx::Shape{3, batch_size, seq_len}, mx::int32);
}

inline std::vector<std::array<int, 3>> to_grid_vector(const std::optional<mx::array>& grid) {
    std::vector<std::array<int, 3>> result;
    if (!grid.has_value()) {
        return result;
    }

    auto grid_i32 = mx::astype(*grid, mx::int32);
    mx::eval(grid_i32);
    const int* data = grid_i32.data<int>();
    int rows = grid_i32.shape(0);
    for (int row = 0; row < rows; ++row) {
        result.push_back({
            data[row * 3 + 0],
            data[row * 3 + 1],
            data[row * 3 + 2],
        });
    }
    return result;
}

} // namespace mlx_helpers
} // namespace qwen3_5_mlx