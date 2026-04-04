#pragma once

#include "mlx_helpers.h"

#include <mlx/compile.h>
#include <mlx/fast.h>

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace qwen3_5_mlx {
namespace mx = mlx::core;

namespace runtime_options {

enum class GatedDeltaMode {
    Ops,
    CompiledOps,
    MetalKernel,
};

inline GatedDeltaMode gated_delta_mode = GatedDeltaMode::Ops;

inline void set_gated_delta_mode(GatedDeltaMode mode) {
    gated_delta_mode = mode;
}

inline GatedDeltaMode get_gated_delta_mode() {
    return gated_delta_mode;
}

inline const char* gated_delta_mode_name(GatedDeltaMode mode) {
    switch (mode) {
        case GatedDeltaMode::MetalKernel: return "metal_kernel";
        case GatedDeltaMode::CompiledOps: return "compiled_ops";
        case GatedDeltaMode::Ops:
        default: return "ops";
    }
}

} // namespace runtime_options

namespace gated_delta_detail {

inline mx::array compiled_compute_g(
    const mx::array& A_log,
    const mx::array& a,
    const mx::array& dt_bias) {
    static auto compiled = mx::compile(
        [](const std::vector<mx::array>& inputs) {
            auto A = mx::exp(mx::astype(inputs[0], mx::float32));
            auto g = mx::exp(-A * mlx_helpers::softplus(inputs[1] + inputs[2]));
            return std::vector<mx::array>{mx::astype(g, inputs[1].dtype())};
        },
        true);
    return compiled({A_log, a, dt_bias})[0];
}

inline std::pair<mx::array, mx::array> compiled_gated_delta_step_ops(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    const mx::array& g,
    const mx::array& beta,
    const mx::array& state) {
    static auto compiled_scalar = mx::compile(
        [](const std::vector<mx::array>& inputs) {
            auto decay = mx::expand_dims(mx::expand_dims(inputs[3], -1), -1);
            auto decayed_state = inputs[5] * decay;
            auto kv_mem = mx::sum(decayed_state * mx::expand_dims(inputs[1], 2), -1);
            auto delta = (inputs[2] - kv_mem) * mx::expand_dims(inputs[4], -1);
            auto new_state = decayed_state +
                mx::expand_dims(inputs[1], 2) * mx::expand_dims(delta, -1);
            auto y = mx::sum(new_state * mx::expand_dims(inputs[0], 2), -1);
            return std::vector<mx::array>{mx::astype(y, inputs[0].dtype()), new_state};
        });

    static auto compiled_vector = mx::compile(
        [](const std::vector<mx::array>& inputs) {
            auto decay = mx::expand_dims(inputs[3], 2);
            auto decayed_state = inputs[5] * decay;
            auto kv_mem = mx::sum(decayed_state * mx::expand_dims(inputs[1], 2), -1);
            auto delta = (inputs[2] - kv_mem) * mx::expand_dims(inputs[4], -1);
            auto new_state = decayed_state +
                mx::expand_dims(inputs[1], 2) * mx::expand_dims(delta, -1);
            auto y = mx::sum(new_state * mx::expand_dims(inputs[0], 2), -1);
            return std::vector<mx::array>{mx::astype(y, inputs[0].dtype()), new_state};
        });

    auto outputs = (g.ndim() == 2)
        ? compiled_scalar({q, k, v, g, beta, state})
        : compiled_vector({q, k, v, g, beta, state});
    return {outputs[0], outputs[1]};
}

inline std::string make_gated_delta_kernel_source(bool has_mask, bool vectorized) {
    const std::string mask_source = has_mask ? "mask[b_idx * T + t]" : "true";
    const std::string g_comment = vectorized
        ? "// g: [B, T, Hv, Dk]"
        : "// g: [B, T, Hv]";
    const std::string g_setup = vectorized
        ? "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
        : "auto g_ = g + b_idx * T * Hv;";
    const std::string g_access = vectorized ? "g_[s_idx]" : "g_[hv_idx]";
    const std::string g_advance = vectorized ? "g_ += Hv * Dk;" : "g_ += Hv;";

    std::string source = R"(
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

)";
    source += "        " + g_comment + "\n";
    source += "        " + g_setup + "\n";
    source += R"(
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          if ()";
    source += mask_source;
    source += R"() {
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * )";
    source += g_access;
    source += R"(;
              kv_mem += state[i] * k_[s_idx];
            }
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {
              y[dv_idx] = static_cast<InT>(out);
            }
          }
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
)";
    source += "          " + g_advance + "\n";
    source += R"(
          beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<InT>(state[i]);
        }
    )";
    return source;
}

inline mx::fast::CustomKernelFunction& gated_delta_kernel_function(bool has_mask, bool vectorized) {
    static auto scalar = mx::fast::metal_kernel(
        "gated_delta_step",
        {"q", "k", "v", "g", "beta", "state_in", "T"},
        {"y", "state_out"},
        make_gated_delta_kernel_source(false, false));
    static auto scalar_masked = mx::fast::metal_kernel(
        "gated_delta_step_mask",
        {"q", "k", "v", "g", "beta", "state_in", "T", "mask"},
        {"y", "state_out"},
        make_gated_delta_kernel_source(true, false));
    static auto vector = mx::fast::metal_kernel(
        "gated_delta_step_vec",
        {"q", "k", "v", "g", "beta", "state_in", "T"},
        {"y", "state_out"},
        make_gated_delta_kernel_source(false, true));
    static auto vector_masked = mx::fast::metal_kernel(
        "gated_delta_step_vec_mask",
        {"q", "k", "v", "g", "beta", "state_in", "T", "mask"},
        {"y", "state_out"},
        make_gated_delta_kernel_source(true, true));

    if (vectorized) {
        return has_mask ? vector_masked : vector;
    }
    return has_mask ? scalar_masked : scalar;
}

inline bool kernel_supported(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    const mx::array& g,
    const mx::array& beta,
    const std::optional<mx::array>& state,
    bool use_kernel) {
    if (!use_kernel || mx::default_device() != mx::Device::gpu || !mx::metal::is_available()) {
        return false;
    }
    if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4 || beta.ndim() != 3) {
        return false;
    }
    if (g.ndim() != 3 && g.ndim() != 4) {
        return false;
    }
    if (q.shape(3) % 32 != 0) {
        return false;
    }
    if (q.dtype() != k.dtype() || q.dtype() != v.dtype() || q.dtype() != g.dtype() || q.dtype() != beta.dtype()) {
        return false;
    }
    if (state.has_value() && state->dtype() != q.dtype()) {
        return false;
    }
    return true;
}

inline std::pair<mx::array, mx::array> gated_delta_kernel(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    const mx::array& g,
    const mx::array& beta,
    const mx::array& state,
    const std::optional<mx::array>& mask = std::nullopt) {
    const int batch = q.shape(0);
    const int steps = q.shape(1);
    const int key_heads = q.shape(2);
    const int key_dim = q.shape(3);
    const int value_heads = v.shape(2);
    const int value_dim = v.shape(3);

    auto t_steps = mx::array(steps, mx::int32);
    std::vector<mx::array> inputs = {q, k, v, g, beta, state, t_steps};
    if (mask.has_value()) {
        inputs.push_back(*mask);
    }

    auto& kernel = gated_delta_kernel_function(mask.has_value(), g.ndim() == 4);
    auto outputs = kernel(
        inputs,
        {mx::Shape{batch, steps, value_heads, value_dim}, state.shape()},
        {q.dtype(), state.dtype()},
        {32, value_dim, batch * value_heads},
        {32, std::min(value_dim, 4), 1},
        {
            {"InT", q.dtype()},
            {"Dk", key_dim},
            {"Dv", value_dim},
            {"Hk", key_heads},
            {"Hv", value_heads},
        },
        std::nullopt,
        false,
        {});
    return {outputs[0], outputs[1]};
}

} // namespace gated_delta_detail

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

inline std::pair<mx::array, mx::array> gated_delta_ops_compiled(
    mx::array q,
    mx::array k,
    const mx::array& v,
    const mx::array& g,
    const mx::array& beta,
    std::optional<mx::array> state = std::nullopt,
    const std::optional<mx::array>& mask = std::nullopt) {
    if (mask.has_value()) {
        return gated_delta_ops(q, k, v, g, beta, state, mask);
    }

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

        auto [y_t, next_state] = gated_delta_detail::compiled_gated_delta_step_ops(
            q_t, k_t, v_t, g_t, beta_t, current_state);
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
    bool use_kernel = true) {
    auto beta = mx::sigmoid(b);
    auto mode = runtime_options::get_gated_delta_mode();
    auto g = mode != runtime_options::GatedDeltaMode::Ops
        ? gated_delta_detail::compiled_compute_g(A_log, a, dt_bias)
        : mlx_helpers::compute_g(A_log, a, dt_bias);

    if (mode == runtime_options::GatedDeltaMode::MetalKernel &&
        gated_delta_detail::kernel_supported(q, k, v, g, beta, state, use_kernel)) {
        if (!state.has_value()) {
            state = mx::zeros(
                {q.shape(0), v.shape(2), v.shape(3), q.shape(3)},
                q.dtype());
        }
        return gated_delta_detail::gated_delta_kernel(q, k, v, g, beta, *state, mask);
    }

    if (mode == runtime_options::GatedDeltaMode::CompiledOps ||
        mode == runtime_options::GatedDeltaMode::MetalKernel) {
        return gated_delta_ops_compiled(q, k, v, g, beta, state, mask);
    }
    return gated_delta_ops(q, k, v, g, beta, state, mask);
}

} // namespace qwen3_5_mlx