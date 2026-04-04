#pragma once

#include "mlx_helpers.h"

#include <optional>
#include <tuple>
#include <vector>

namespace qwen3_5_mlx {
namespace mx = mlx::core;

inline std::pair<mx::array, mx::array> gated_delta_step_ops(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    const mx::array& g,
    const mx::array& beta,
    const mx::array& state,
    const std::optional<mx::array>& mask = std::nullopt) {
    if (g.ndim() != 2 && g.ndim() != 3) {
        throw std::invalid_argument("gated_delta_step_ops expects g rank 2 or 3");
    }
    auto decay = g.ndim() == 2
        ? mx::expand_dims(mx::expand_dims(g, -1), -1)
        : mx::expand_dims(g, 2);

    auto old_state = state;
    auto decayed_state = state * decay;
    auto kv_mem = mx::sum(decayed_state * mx::expand_dims(k, 2), -1);
    auto delta = (v - kv_mem) * mx::expand_dims(beta, -1);
    auto new_state = decayed_state +
        mx::expand_dims(k, 2) * mx::expand_dims(delta, -1);
    auto y = mx::sum(new_state * mx::expand_dims(q, 2), -1);

    if (mask.has_value()) {
        auto state_mask = mx::expand_dims(mx::expand_dims(mx::expand_dims(*mask, 1), 2), 3);
        auto y_mask = mx::expand_dims(mx::expand_dims(*mask, 1), 2);
        new_state = mx::where(state_mask, new_state, old_state);
        y = mx::where(y_mask, y, mx::zeros_like(y));
    }

    return {mx::astype(y, q.dtype()), new_state};
}

inline std::pair<mx::array, mx::array> gated_delta_ops(
    mx::array q,
    mx::array k,
    const mx::array& v,
    const mx::array& g,
    const mx::array& beta,
    std::optional<mx::array> state = std::nullopt,
    const std::optional<mx::array>& mask = std::nullopt) {
    int batch = q.shape(0);
    int steps = q.shape(1);
    int key_heads = q.shape(2);
    int key_dim = q.shape(3);
    int value_heads = v.shape(2);
    int value_dim = v.shape(3);

    if (!state.has_value()) {
        state = mx::zeros({batch, value_heads, value_dim, key_dim}, mx::float32);
    }

    if (value_heads % key_heads != 0) {
        throw std::invalid_argument("value heads must be divisible by key heads");
    }
    int repeat_factor = value_heads / key_heads;
    if (repeat_factor > 1) {
        q = mx::repeat(q, repeat_factor, 2);
        k = mx::repeat(k, repeat_factor, 2);
    }

    auto q_steps = mx::split(q, steps, 1);
    auto k_steps = mx::split(k, steps, 1);
    auto v_steps = mx::split(v, steps, 1);
    auto g_steps = mx::split(g, steps, 1);
    auto beta_steps = mx::split(beta, steps, 1);
    std::vector<mx::array> outputs;
    outputs.reserve(steps);

    std::vector<mx::array> mask_steps;
    if (mask.has_value()) {
        mask_steps = mx::split(*mask, steps, 1);
    }

    auto current_state = *state;
    for (int step = 0; step < steps; ++step) {
        auto q_t = mx::reshape(q_steps[step], {batch, value_heads, key_dim});
        auto k_t = mx::reshape(k_steps[step], {batch, value_heads, key_dim});
        auto v_t = mx::reshape(v_steps[step], {batch, value_heads, value_dim});
        auto g_t = mx::reshape(g_steps[step],
            g.ndim() == 4
                ? mx::Shape{batch, value_heads, key_dim}
                : mx::Shape{batch, value_heads});
        auto beta_t = mx::reshape(beta_steps[step], {batch, value_heads});

        std::optional<mx::array> mask_t = std::nullopt;
        if (mask.has_value()) {
            mask_t = mx::reshape(mask_steps[step], {batch});
        }

        auto [y_t, next_state] = gated_delta_step_ops(
            q_t, k_t, v_t, g_t, beta_t, current_state, mask_t);
        outputs.push_back(y_t);
        current_state = next_state;
    }

    return {mx::stack(outputs, 1), current_state};
}

inline std::pair<mx::array, mx::array> gated_delta_update(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    const mx::array& a,
    const mx::array& b,
    const mx::array& A_log,
    const mx::array& dt_bias,
    std::optional<mx::array> state = std::nullopt,
    const std::optional<mx::array>& mask = std::nullopt,
    bool /*use_kernel*/ = true) {
    auto beta = mx::sigmoid(b);
    auto g = mlx_helpers::compute_g(A_log, a, dt_bias);
    return gated_delta_ops(q, k, v, g, beta, state, mask);
}

} // namespace qwen3_5_mlx