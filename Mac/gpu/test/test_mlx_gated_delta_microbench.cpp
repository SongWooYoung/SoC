#include "models/qwen3_5_mlx/config.h"
#include "models/qwen3_5_mlx/gated_delta.h"

#include <mlx/mlx.h>
#include <mlx/random.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

namespace fs = std::filesystem;
namespace mx = mlx::core;

namespace {

struct BenchResult {
    double dispatch_ms = 0.0;
    double sync_ms = 0.0;
};

std::string json_number(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << value;
    return out.str();
}

BenchResult bench_mode(
    qwen3_5_mlx::runtime_options::GatedDeltaMode mode,
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    const mx::array& a,
    const mx::array& b,
    const mx::array& A_log,
    const mx::array& dt_bias,
    int iterations) {
    qwen3_5_mlx::runtime_options::set_gated_delta_mode(mode);

    for (int warmup = 0; warmup < 3; ++warmup) {
        auto warm = qwen3_5_mlx::gated_delta_update(q, k, v, a, b, A_log, dt_bias);
        mx::eval(warm.first, warm.second);
        mx::synchronize();
    }

    BenchResult total;
    for (int iter = 0; iter < iterations; ++iter) {
        auto dispatch_start = std::chrono::steady_clock::now();
        auto out = qwen3_5_mlx::gated_delta_update(q, k, v, a, b, A_log, dt_bias);
        total.dispatch_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - dispatch_start).count();

        auto sync_start = std::chrono::steady_clock::now();
        mx::eval(out.first, out.second);
        mx::synchronize();
        total.sync_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - sync_start).count();
    }

    total.dispatch_ms /= iterations;
    total.sync_ms /= iterations;
    return total;
}

void write_result(
    std::ofstream& out,
    const std::string& name,
    const BenchResult& ops,
    const BenchResult& compiled,
    bool trailing_comma) {
    out << "  \"" << name << "\": {\n";
    out << "    \"ops\": {\"dispatch_ms\": " << json_number(ops.dispatch_ms)
        << ", \"sync_ms\": " << json_number(ops.sync_ms) << "},\n";
    out << "    \"compiled_ops\": {\"dispatch_ms\": " << json_number(compiled.dispatch_ms)
        << ", \"sync_ms\": " << json_number(compiled.sync_ms) << "}\n";
    out << "  }";
    if (trailing_comma) {
        out << ",";
    }
    out << "\n";
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <config.json> <output.json> [prefill_tokens] [iterations]\n", argv[0]);
        return 1;
    }

    const std::string config_path = argv[1];
    const std::string output_path = argv[2];
    const int prefill_tokens = argc >= 4 ? std::atoi(argv[3]) : 64;
    const int iterations = argc >= 5 ? std::atoi(argv[4]) : 100;

    const Qwen3_5Config config = Qwen3_5Config::from_file(config_path);
    const auto& text = config.text_config;

    const int batch = 1;
    const int key_heads = text.linear_num_key_heads;
    const int value_heads = text.linear_num_value_heads;
    const int key_dim = text.linear_key_head_dim;
    const int value_dim = text.linear_value_head_dim;

    auto A_log = mx::random::normal({value_heads}, mx::float32);
    auto dt_bias = mx::random::normal({value_heads}, mx::float32);

    auto bench_shape = [&](int steps) {
        auto q = mx::random::normal({batch, steps, key_heads, key_dim}, mx::float32);
        auto k = mx::random::normal({batch, steps, key_heads, key_dim}, mx::float32);
        auto v = mx::random::normal({batch, steps, value_heads, value_dim}, mx::float32);
        auto a = mx::random::normal({batch, steps, value_heads}, mx::float32);
        auto b = mx::random::normal({batch, steps, value_heads}, mx::float32);
        auto ops = bench_mode(
            qwen3_5_mlx::runtime_options::GatedDeltaMode::Ops,
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            iterations);
        auto compiled = bench_mode(
            qwen3_5_mlx::runtime_options::GatedDeltaMode::CompiledOps,
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            iterations);
        return std::pair<BenchResult, BenchResult>{ops, compiled};
    };

    auto [prefill_ops, prefill_compiled] = bench_shape(prefill_tokens);
    auto [decode_ops, decode_compiled] = bench_shape(1);

    fs::create_directories(fs::path(output_path).parent_path());
    std::ofstream out(output_path);
    out << "{\n";
    out << "  \"prefill_tokens\": " << prefill_tokens << ",\n";
    out << "  \"iterations\": " << iterations << ",\n";
    write_result(out, "prefill", prefill_ops, prefill_compiled, true);
    write_result(out, "decode", decode_ops, decode_compiled, false);
    out << "}\n";
    return 0;
}