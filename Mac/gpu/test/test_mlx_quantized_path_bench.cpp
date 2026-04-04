#include "models/qwen3_5_mlx/language.h"
#include "utils/json.h"

#include <mlx/mlx.h>
#include <mlx/random.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>

namespace fs = std::filesystem;
namespace mx = mlx::core;

namespace {

struct QuantizationConfig {
    int group_size = 64;
    int bits = 8;
    std::string mode = "affine";
};

struct BenchResult {
    double dispatch_ms = 0.0;
    double sync_ms = 0.0;
};

std::string json_escape(const std::string& s) {
    std::ostringstream out;
    for (char c : s) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
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

BenchResult bench_op(const mx::array& output) {
    BenchResult result;
    auto sync_start = std::chrono::steady_clock::now();
    mx::eval(output);
    mx::synchronize();
    result.sync_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sync_start).count();
    return result;
}

BenchResult bench_mlp(
    const qwen3_5_mlx::MLP& mlp,
    const mx::array& inputs,
    int iterations) {
    BenchResult total;
    for (int i = 0; i < iterations; ++i) {
        auto dispatch_start = std::chrono::steady_clock::now();
        auto out = mlp.forward(inputs);
        total.dispatch_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - dispatch_start).count();
        auto sync = bench_op(out);
        total.sync_ms += sync.sync_ms;
    }
    total.dispatch_ms /= iterations;
    total.sync_ms /= iterations;
    return total;
}

BenchResult bench_lm_head(
    const qwen3_5_mlx::mlx_helpers::TensorParam& lm_head,
    const mx::array& inputs,
    int iterations) {
    BenchResult total;
    for (int i = 0; i < iterations; ++i) {
        auto dispatch_start = std::chrono::steady_clock::now();
        auto out = qwen3_5_mlx::mlx_helpers::linear(inputs, lm_head);
        total.dispatch_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - dispatch_start).count();
        auto sync = bench_op(out);
        total.sync_ms += sync.sync_ms;
    }
    total.dispatch_ms /= iterations;
    total.sync_ms /= iterations;
    return total;
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
    const int iterations = argc >= 5 ? std::atoi(argv[4]) : 50;

    const Qwen3_5Config config = Qwen3_5Config::from_file(config_path);
    const QuantizationConfig qc = load_quant_config(config_path);

    qwen3_5_mlx::MLP mlp;
    mlp.gate_proj_w = make_quant_param(config.text_config.intermediate_size, config.text_config.hidden_size, qc);
    mlp.up_proj_w = make_quant_param(config.text_config.intermediate_size, config.text_config.hidden_size, qc);
    mlp.down_proj_w = make_quant_param(config.text_config.hidden_size, config.text_config.intermediate_size, qc);
    auto lm_head = make_quant_param(config.text_config.vocab_size, config.text_config.hidden_size, qc);

    auto prefill_input = mx::random::normal({1, prefill_tokens, config.text_config.hidden_size}, mx::float32);
    auto decode_input = mx::random::normal({1, 1, config.text_config.hidden_size}, mx::float32);

    auto mlp_prefill = bench_mlp(mlp, prefill_input, iterations);
    auto mlp_decode = bench_mlp(mlp, decode_input, iterations);
    auto lm_head_prefill = bench_lm_head(lm_head, prefill_input, iterations);
    auto lm_head_decode = bench_lm_head(lm_head, decode_input, iterations);

    fs::create_directories(fs::path(output_path).parent_path());
    std::ofstream out(output_path);
    out << "{\n";
    out << "  \"prefill_tokens\": " << prefill_tokens << ",\n";
    out << "  \"iterations\": " << iterations << ",\n";
    out << "  \"mlp_prefill\": {\"dispatch_ms\": " << json_number(mlp_prefill.dispatch_ms) << ", \"sync_ms\": " << json_number(mlp_prefill.sync_ms) << "},\n";
    out << "  \"mlp_decode\": {\"dispatch_ms\": " << json_number(mlp_decode.dispatch_ms) << ", \"sync_ms\": " << json_number(mlp_decode.sync_ms) << "},\n";
    out << "  \"lm_head_prefill\": {\"dispatch_ms\": " << json_number(lm_head_prefill.dispatch_ms) << ", \"sync_ms\": " << json_number(lm_head_prefill.sync_ms) << "},\n";
    out << "  \"lm_head_decode\": {\"dispatch_ms\": " << json_number(lm_head_decode.dispatch_ms) << ", \"sync_ms\": " << json_number(lm_head_decode.sync_ms) << "}\n";
    out << "}\n";
    return 0;
}