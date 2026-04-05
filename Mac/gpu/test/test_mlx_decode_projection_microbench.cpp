#include "models/qwen3_5_mlx/language.h"
#include "test/mlx_quantized_model_utils.h"

#include <mlx/mlx.h>

#include <chrono>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

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

BenchResult bench_array_op(const std::function<mx::array()>& fn, int warmup, int iterations) {
	for (int index = 0; index < warmup; ++index) {
		auto warm = fn();
		mx::eval(warm);
		mx::synchronize();
	}

	BenchResult total;
	for (int index = 0; index < iterations; ++index) {
		auto dispatch_start = std::chrono::steady_clock::now();
		auto out = fn();
		total.dispatch_ms += std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now() - dispatch_start).count();

		auto sync_start = std::chrono::steady_clock::now();
		mx::eval(out);
		mx::synchronize();
		total.sync_ms += std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now() - sync_start).count();
	}

	total.dispatch_ms /= iterations;
	total.sync_ms /= iterations;
	return total;
}

double mean_sync(const std::vector<BenchResult>& values) {
	if (values.empty()) {
		return 0.0;
	}
	double sum = 0.0;
	for (const auto& value : values) {
		sum += value.sync_ms;
	}
	return sum / static_cast<double>(values.size());
}

double mean_dispatch(const std::vector<BenchResult>& values) {
	if (values.empty()) {
		return 0.0;
	}
	double sum = 0.0;
	for (const auto& value : values) {
		sum += value.dispatch_ms;
	}
	return sum / static_cast<double>(values.size());
}

} // namespace

int main(int argc, char* argv[]) {
	if (argc < 3) {
		std::fprintf(stderr, "Usage: %s <model_dir> <output.json> [token_id] [warmup] [iterations]\n", argv[0]);
		return 1;
	}

	const std::string model_dir = argv[1];
	const std::string output_path = argv[2];
	const int token_id = argc >= 4 ? std::atoi(argv[3]) : 42;
	const int warmup = argc >= 5 ? std::max(1, std::atoi(argv[4])) : 5;
	const int iterations = argc >= 6 ? std::max(1, std::atoi(argv[5])) : 50;

	const std::string config_path = model_dir + "/config.json";
	const auto config = Qwen3_5Config::from_file(config_path);
	const auto quant_config = test_mlx_quantized_utils::load_quant_config(config_path);
	const auto weights = test_mlx_quantized_utils::load_weight_map(model_dir);
	auto language_model = test_mlx_quantized_utils::load_language_model(weights, config, quant_config);

	qwen3_5_mlx::stage_trace::set_enabled(false);

	auto token_input = mx::array(&token_id, mx::Shape{1, 1}, mx::int32);
	auto hidden = qwen3_5_mlx::mlx_helpers::embedding(language_model.model.embed_w, token_input);
	mx::eval(hidden);
	mx::synchronize();

	std::vector<BenchResult> mlp_results;
	mlp_results.reserve(language_model.model.layers.size());

	for (const auto& layer : language_model.model.layers) {
		auto mlp_input = mx::fast::rms_norm(hidden, layer.post_attn_ln_w, layer.ln_eps);
		mx::eval(mlp_input);
		mx::synchronize();
		mlp_results.push_back(bench_array_op([&]() {
			return layer.mlp.forward(mlp_input);
		}, warmup, iterations));
	}

	BenchResult lm_head_result = bench_array_op([&]() {
		if (language_model.config.tie_word_embeddings || !language_model.lm_head_w.has_value()) {
			return qwen3_5_mlx::mlx_helpers::linear(hidden, language_model.model.embed_w);
		}
		return qwen3_5_mlx::mlx_helpers::linear(hidden, *language_model.lm_head_w);
	}, warmup, iterations);

	fs::create_directories(fs::path(output_path).parent_path());
	std::ofstream out(output_path);
	out << "{\n";
	out << "  \"token_id\": " << token_id << ",\n";
	out << "  \"warmup\": " << warmup << ",\n";
	out << "  \"iterations\": " << iterations << ",\n";
	out << "  \"mlp_average\": {\"dispatch_ms\": " << json_number(mean_dispatch(mlp_results))
		<< ", \"sync_ms\": " << json_number(mean_sync(mlp_results)) << "},\n";
	out << "  \"mlp_layers\": [\n";
	for (size_t index = 0; index < mlp_results.size(); ++index) {
		const auto& result = mlp_results[index];
		const auto& layer = language_model.model.layers[index];
		out << "    {\"layer_index\": " << index
			<< ", \"layer_type\": \"" << (layer.is_linear ? "linear" : "full_attention") << "\""
			<< ", \"dispatch_ms\": " << json_number(result.dispatch_ms)
			<< ", \"sync_ms\": " << json_number(result.sync_ms) << "}";
		if (index + 1 != mlp_results.size()) {
			out << ",";
		}
		out << "\n";
	}
	out << "  ],\n";
	out << "  \"lm_head_decode\": {\"dispatch_ms\": " << json_number(lm_head_result.dispatch_ms)
		<< ", \"sync_ms\": " << json_number(lm_head_result.sync_ms) << "}\n";
	out << "}\n";
	return 0;
}