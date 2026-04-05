#include "models/qwen3_5_mlx/language.h"
#include "utils/json.h"

#include <mlx/mlx.h>
#include <mlx/random.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
namespace mx = mlx::core;

namespace {

struct QuantizationConfig {
    int group_size = 64;
    int bits = 8;
    std::string mode = "affine";
};

struct StageStats {
    int calls = 0;
    double dispatch_ms = 0.0;
    double sync_ms = 0.0;
};

using StageMap = std::unordered_map<std::string, StageStats>;

qwen3_5_mlx::runtime_options::GatedDeltaMode gated_delta_mode_from_env() {
    const char* value = std::getenv("QWEN3_5_MLX_GATED_DELTA_MODE");
    if (value != nullptr) {
        const std::string text(value);
        if (text == "compiled_ops") {
            return qwen3_5_mlx::runtime_options::GatedDeltaMode::CompiledOps;
        }
        if (text == "metal_kernel") {
            return qwen3_5_mlx::runtime_options::GatedDeltaMode::MetalKernel;
        }
    }
    return qwen3_5_mlx::runtime_options::GatedDeltaMode::Ops;
}

std::string json_escape(const std::string& s) {
    std::ostringstream out;
    for (char c : s) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << c; break;
        }
    }
    return out.str();
}

std::string json_number(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << value;
    return out.str();
}

QuantizationConfig load_quant_config(const std::string& config_path) {
    QuantizationConfig qc;
    auto root = JsonParser::parse_file(config_path);
    const JsonValue* quant = root.find("quantization");
    if (!quant) {
        quant = root.find("quantization_config");
    }
    if (quant) {
        if (auto* value = quant->find("group_size")) qc.group_size = value->as_int();
        if (auto* value = quant->find("bits")) qc.bits = value->as_int();
        if (auto* value = quant->find("mode")) qc.mode = value->as_string();
    }
    return qc;
}

qwen3_5_mlx::mlx_helpers::TensorParam make_quant_param(
    int out_features,
    int in_features,
    const QuantizationConfig& qc) {
    qwen3_5_mlx::mlx_helpers::TensorParam param;
    int packed = in_features / (32 / qc.bits);
    int scale_cols = in_features / qc.group_size;
    param.weight = mx::random::randint(0, 255, {out_features, packed}, mx::uint32);
    param.scales = mx::random::normal({out_features, scale_cols}, mx::float32);
    param.biases = mx::random::normal({out_features, scale_cols}, mx::float32);
    param.group_size = qc.group_size;
    param.bits = qc.bits;
    param.mode = qc.mode;
    return param;
}

mx::array sync_stage(
    StageMap& stages,
    const std::string& name,
    mx::array value,
    const std::chrono::steady_clock::time_point& dispatch_start) {
    auto dispatch_end = std::chrono::steady_clock::now();
    auto& stats = stages[name];
    stats.calls += 1;
    stats.dispatch_ms += std::chrono::duration<double, std::milli>(dispatch_end - dispatch_start).count();
    auto sync_start = std::chrono::steady_clock::now();
    mx::eval(value);
    mx::synchronize();
    stats.sync_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sync_start).count();
    return value;
}

std::string json_stage_map(const StageMap& stage_map) {
    std::vector<std::string> names;
    names.reserve(stage_map.size());
    for (const auto& [name, _stats] : stage_map) {
        names.push_back(name);
    }
    std::sort(names.begin(), names.end());

    std::ostringstream out;
    out << "{";
    for (size_t i = 0; i < names.size(); ++i) {
        const auto& name = names[i];
        const auto& stats = stage_map.at(name);
        if (i != 0) out << ", ";
        out << "\"" << json_escape(name) << "\": {"
            << "\"calls\": " << stats.calls << ", "
            << "\"dispatch_ms\": " << json_number(stats.dispatch_ms) << ", "
            << "\"sync_ms\": " << json_number(stats.sync_ms) << "}";
    }
    out << "}";
    return out.str();
}

mx::array run_linear_attention_bench(
    qwen3_5_mlx::GatedDeltaNet& linear,
    qwen3_5_mlx::ArraysCache& cache,
    const mx::array& inputs,
    StageMap& stages) {
    using namespace qwen3_5_mlx;

    int batch = inputs.shape(0);
    int seq_len = inputs.shape(1);

    auto t0 = std::chrono::steady_clock::now();
    auto mixed_qkv = sync_stage(stages, "in_proj_qkv", mlx_helpers::linear(inputs, linear.in_proj_qkv_w), t0);
    t0 = std::chrono::steady_clock::now();
    auto z = sync_stage(stages, "in_proj_z", mlx_helpers::linear(inputs, linear.in_proj_z_w), t0);
    z = mx::reshape(z, {batch, seq_len, linear.num_v_heads, linear.head_v_dim});

    t0 = std::chrono::steady_clock::now();
    auto b = sync_stage(stages, "in_proj_b", mlx_helpers::linear(inputs, linear.in_proj_b_w), t0);
    t0 = std::chrono::steady_clock::now();
    auto a = sync_stage(stages, "in_proj_a", mlx_helpers::linear(inputs, linear.in_proj_a_w), t0);

    mx::array conv_state = cache.conv_state.value_or(
        mx::zeros({batch, linear.conv_kernel_size - 1, linear.conv_dim}, inputs.dtype()));
    auto conv_input = mx::concatenate({conv_state, mixed_qkv}, 1);

    t0 = std::chrono::steady_clock::now();
    if (runtime_options::get_linear_cache_mode() == runtime_options::LinearCacheMode::ArraysStyle) {
        int n_keep = linear.conv_kernel_size - 1;
        int token_count = inputs.shape(1);
        if (token_count >= n_keep) {
            cache.conv_state = mlx_helpers::slice_axis(mixed_qkv, 1, token_count - n_keep, token_count);
        } else {
            int keep_old = n_keep - token_count;
            auto old_tail = mlx_helpers::slice_axis(conv_state, 1, conv_state.shape(1) - keep_old, conv_state.shape(1));
            cache.conv_state = mx::concatenate({old_tail, mixed_qkv}, 1);
        }
        cache.advance(token_count);
    } else {
        cache.conv_state = mlx_helpers::take_last_tokens(conv_input, linear.conv_kernel_size - 1);
    }
    sync_stage(stages, "conv_cache_update", *cache.conv_state, t0);

    t0 = std::chrono::steady_clock::now();
    auto conv_out = sync_stage(stages, "conv1d", mlx_helpers::silu(mx::conv1d(conv_input, linear.conv1d_w, 1, 0, 1, linear.conv_dim)), t0);

    auto qkv_parts = mx::split(conv_out, {linear.key_dim, linear.key_dim * 2}, conv_out.ndim() - 1);
    auto q = mx::reshape(qkv_parts[0], {batch, seq_len, linear.num_k_heads, linear.head_k_dim});
    auto k = mx::reshape(qkv_parts[1], {batch, seq_len, linear.num_k_heads, linear.head_k_dim});
    auto v = mx::reshape(qkv_parts[2], {batch, seq_len, linear.num_v_heads, linear.head_v_dim});

    float inv_scale = std::pow(static_cast<float>(linear.head_k_dim), -0.5f);
    t0 = std::chrono::steady_clock::now();
    q = sync_stage(stages, "q_norm", (inv_scale * inv_scale) * mx::fast::rms_norm(q, std::nullopt, 1e-6f), t0);
    t0 = std::chrono::steady_clock::now();
    k = sync_stage(stages, "k_norm", inv_scale * mx::fast::rms_norm(k, std::nullopt, 1e-6f), t0);

    t0 = std::chrono::steady_clock::now();
    auto [out, next_state] = gated_delta_update(q, k, v, a, b, linear.A_log, linear.dt_bias, cache.rec_state, std::nullopt, true);
    auto dispatch_end = std::chrono::steady_clock::now();
    auto& delta_stats = stages["gated_delta_update"];
    delta_stats.calls += 1;
    delta_stats.dispatch_ms += std::chrono::duration<double, std::milli>(dispatch_end - t0).count();
    auto sync_start = std::chrono::steady_clock::now();
    mx::eval(out, next_state);
    mx::synchronize();
    delta_stats.sync_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sync_start).count();

    t0 = std::chrono::steady_clock::now();
    cache.rec_state = next_state;
    sync_stage(stages, "rec_state_update", *cache.rec_state, t0);

    qwen3_5_mlx::RMSNormGated norm{linear.norm_w, linear.norm_eps};
    t0 = std::chrono::steady_clock::now();
    out = sync_stage(stages, "norm_gated", norm.forward(out, z), t0);
    out = mx::reshape(out, {batch, seq_len, linear.value_dim});
    t0 = std::chrono::steady_clock::now();
    return sync_stage(stages, "out_proj", mlx_helpers::linear(out, linear.out_proj_w), t0);
}

void accumulate(StageMap& total, const StageMap& part) {
    for (const auto& [name, stats] : part) {
        auto& dst = total[name];
        dst.calls += stats.calls;
        dst.dispatch_ms += stats.dispatch_ms;
        dst.sync_ms += stats.sync_ms;
    }
}

StageMap bench_mode(
    const Qwen3_5TextConfig& cfg,
    const QuantizationConfig& qc,
    qwen3_5_mlx::runtime_options::LinearCacheMode mode,
    qwen3_5_mlx::runtime_options::GatedDeltaMode gated_delta_mode,
    int prefill_iterations,
    int decode_iterations,
    int prefill_tokens) {
    using namespace qwen3_5_mlx;

    runtime_options::set_linear_cache_mode(mode);
    runtime_options::set_gated_delta_mode(gated_delta_mode);
    GatedDeltaNet linear;
    linear.num_k_heads = cfg.linear_num_key_heads;
    linear.num_v_heads = cfg.linear_num_value_heads;
    linear.head_k_dim = cfg.linear_key_head_dim;
    linear.head_v_dim = cfg.linear_value_head_dim;
    linear.key_dim = linear.num_k_heads * linear.head_k_dim;
    linear.value_dim = linear.num_v_heads * linear.head_v_dim;
    linear.conv_dim = linear.key_dim * 2 + linear.value_dim;
    linear.conv_kernel_size = cfg.linear_conv_kernel_dim;
    linear.norm_eps = cfg.rms_norm_eps;
    linear.conv1d_w = mx::random::normal({linear.conv_dim, linear.conv_kernel_size, 1}, mx::float32);
    linear.in_proj_qkv_w = make_quant_param(linear.conv_dim, cfg.hidden_size, qc);
    linear.in_proj_z_w = make_quant_param(linear.value_dim, cfg.hidden_size, qc);
    linear.in_proj_b_w = make_quant_param(linear.num_v_heads, cfg.hidden_size, qc);
    linear.in_proj_a_w = make_quant_param(linear.num_v_heads, cfg.hidden_size, qc);
    linear.dt_bias = mx::random::normal({linear.num_v_heads}, mx::float32);
    linear.A_log = mx::random::normal({linear.num_v_heads}, mx::float32);
    linear.norm_w = mx::random::normal({linear.head_v_dim}, mx::float32);
    linear.out_proj_w = make_quant_param(cfg.hidden_size, linear.value_dim, qc);

    StageMap total;

    for (int i = 0; i < prefill_iterations; ++i) {
        ArraysCache cache;
        auto inputs = mx::random::normal({1, prefill_tokens, cfg.hidden_size}, mx::float32);
        StageMap iter;
        auto out = run_linear_attention_bench(linear, cache, inputs, iter);
        mx::eval(out);
        mx::synchronize();
        accumulate(total, iter);
    }

    ArraysCache decode_cache;
    auto warmup = mx::random::normal({1, prefill_tokens, cfg.hidden_size}, mx::float32);
    StageMap warmup_stats;
    auto warm = run_linear_attention_bench(linear, decode_cache, warmup, warmup_stats);
    mx::eval(warm);
    mx::synchronize();
    for (int i = 0; i < decode_iterations; ++i) {
        auto inputs = mx::random::normal({1, 1, cfg.hidden_size}, mx::float32);
        StageMap iter;
        auto out = run_linear_attention_bench(linear, decode_cache, inputs, iter);
        mx::eval(out);
        mx::synchronize();
        accumulate(total, iter);
    }

    return total;
}

std::string json_result(const StageMap& stage_map) {
    return json_stage_map(stage_map);
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <config.json> <output.json> [prefill_tokens] [prefill_iterations] [decode_iterations]\n", argv[0]);
        return 1;
    }

    const std::string config_path = argv[1];
    const std::string output_path = argv[2];
    const int prefill_tokens = argc >= 4 ? std::atoi(argv[3]) : 64;
    const int prefill_iterations = argc >= 5 ? std::atoi(argv[4]) : 20;
    const int decode_iterations = argc >= 6 ? std::atoi(argv[5]) : 128;
    const auto gated_delta_mode = gated_delta_mode_from_env();

    const Qwen3_5Config config = Qwen3_5Config::from_file(config_path);
    const QuantizationConfig qc = load_quant_config(config_path);

    auto legacy = bench_mode(config.text_config, qc, qwen3_5_mlx::runtime_options::LinearCacheMode::Legacy, gated_delta_mode, prefill_iterations, decode_iterations, prefill_tokens);
    auto arrays = bench_mode(config.text_config, qc, qwen3_5_mlx::runtime_options::LinearCacheMode::ArraysStyle, gated_delta_mode, prefill_iterations, decode_iterations, prefill_tokens);

    fs::create_directories(fs::path(output_path).parent_path());
    std::ofstream out(output_path);
    out << "{\n";
    out << "  \"gated_delta_mode\": \""
        << qwen3_5_mlx::runtime_options::gated_delta_mode_name(gated_delta_mode)
        << "\",\n";
    out << "  \"prefill_tokens\": " << prefill_tokens << ",\n";
    out << "  \"prefill_iterations\": " << prefill_iterations << ",\n";
    out << "  \"decode_iterations\": " << decode_iterations << ",\n";
    out << "  \"legacy\": " << json_result(legacy) << ",\n";
    out << "  \"arrays\": " << json_result(arrays) << "\n";
    out << "}\n";
    return 0;
}