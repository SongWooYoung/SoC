#pragma once

#include "models/qwen3_5_mlx/language.h"
#include "utils/json.h"

#include <mlx/io.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace test_mlx_quantized_utils {
namespace fs = std::filesystem;
namespace mx = mlx::core;

struct QuantizationConfig {
	int group_size = 64;
	int bits = 8;
	std::string mode = "affine";
};

using TensorMap = std::unordered_map<std::string, mx::array>;

inline QuantizationConfig load_quant_config(const std::string& config_path) {
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

inline TensorMap load_weight_map(const std::string& model_dir) {
	std::vector<fs::path> shard_paths;
	for (const auto& entry : fs::directory_iterator(model_dir)) {
		auto path = entry.path();
		if (path.extension() == ".safetensors") {
			shard_paths.push_back(path);
		}
	}
	std::sort(shard_paths.begin(), shard_paths.end());

	TensorMap weights;
	for (const auto& shard_path : shard_paths) {
		auto [arrays, _metadata] = mx::load_safetensors(shard_path.string());
		for (auto& [name, array] : arrays) {
			weights.insert_or_assign(name, std::move(array));
		}
	}
	return weights;
}

inline const mx::array& require_array(
	const TensorMap& weights,
	const std::vector<std::string>& names) {
	for (const auto& name : names) {
		auto it = weights.find(name);
		if (it != weights.end()) {
			return it->second;
		}
	}

	std::ostringstream message;
	message << "Missing tensor. Tried:";
	for (const auto& name : names) {
		message << " " << name;
	}
	throw std::runtime_error(message.str());
}

inline std::optional<mx::array> find_optional_array(
	const TensorMap& weights,
	const std::vector<std::string>& names) {
	for (const auto& name : names) {
		auto it = weights.find(name);
		if (it != weights.end()) {
			return it->second;
		}
	}
	return std::nullopt;
}

inline mx::array load_dense(
	const TensorMap& weights,
	const std::vector<std::string>& names,
	bool reorder_conv = false) {
	auto tensor = require_array(weights, names);
	if (reorder_conv && tensor.ndim() == 3 && tensor.shape(2) != 1) {
		return mx::transpose(tensor, {0, 2, 1});
	}
	return tensor;
}

inline qwen3_5_mlx::mlx_helpers::TensorParam load_param(
	const TensorMap& weights,
	const std::vector<std::string>& base_names,
	const QuantizationConfig& quant_config,
	bool reorder_conv = false) {
	for (const auto& base_name : base_names) {
		auto weight_it = weights.find(base_name + ".weight");
		if (weight_it == weights.end()) {
			continue;
		}

		qwen3_5_mlx::mlx_helpers::TensorParam param;
		param.weight = weight_it->second;
		if (reorder_conv && param.weight.ndim() == 3 && param.weight.shape(2) != 1) {
			param.weight = mx::transpose(param.weight, {0, 2, 1});
		}
		if (auto scales = find_optional_array(weights, {base_name + ".scales"}); scales.has_value()) {
			param.scales = *scales;
			param.biases = find_optional_array(weights, {base_name + ".biases"});
			param.group_size = quant_config.group_size;
			param.bits = quant_config.bits;
			param.mode = quant_config.mode;
		}
		return param;
	}

	std::ostringstream message;
	message << "Missing weight parameter. Tried bases:";
	for (const auto& base_name : base_names) {
		message << " " << base_name;
	}
	throw std::runtime_error(message.str());
}

inline qwen3_5_mlx::LanguageModel load_language_model(
	const TensorMap& weights,
	const Qwen3_5Config& config,
	const QuantizationConfig& quant_config) {
	using namespace qwen3_5_mlx;

	LanguageModel language_model(config.text_config, config);
	auto& text_model = language_model.model;
	const auto& text_cfg = config.text_config;

	text_model.config = text_cfg;
	text_model.norm_eps = text_cfg.rms_norm_eps;
	text_model.ssm_idx = 0;
	text_model.fa_idx = std::max(0, default_full_attention_interval(text_cfg) - 1);
	text_model.embed_w = load_param(
		weights,
		{
			"language_model.model.embed_tokens",
			"model.language_model.embed_tokens",
		},
		quant_config);
	text_model.norm_w = load_dense(
		weights,
		{
			"language_model.model.norm.weight",
			"model.language_model.norm.weight",
		});
	text_model.layers.resize(static_cast<size_t>(text_cfg.num_hidden_layers));

	for (int layer_index = 0; layer_index < text_cfg.num_hidden_layers; ++layer_index) {
		auto& layer = text_model.layers[static_cast<size_t>(layer_index)];
		const std::string mlx_prefix = "language_model.model.layers." + std::to_string(layer_index);
		const std::string raw_prefix = "model.language_model.layers." + std::to_string(layer_index);

		layer.is_linear = is_linear_layer(text_cfg, layer_index);
		layer.ln_eps = text_cfg.rms_norm_eps;
		layer.input_ln_w = load_dense(
			weights,
			{
				mlx_prefix + ".input_layernorm.weight",
				raw_prefix + ".input_layernorm.weight",
			});
		layer.post_attn_ln_w = load_dense(
			weights,
			{
				mlx_prefix + ".post_attention_layernorm.weight",
				raw_prefix + ".post_attention_layernorm.weight",
			});
		layer.mlp.gate_proj_w = load_param(
			weights,
			{
				mlx_prefix + ".mlp.gate_proj",
				raw_prefix + ".mlp.gate_proj",
			},
			quant_config);
		layer.mlp.up_proj_w = load_param(
			weights,
			{
				mlx_prefix + ".mlp.up_proj",
				raw_prefix + ".mlp.up_proj",
			},
			quant_config);
		layer.mlp.down_proj_w = load_param(
			weights,
			{
				mlx_prefix + ".mlp.down_proj",
				raw_prefix + ".mlp.down_proj",
			},
			quant_config);

		if (layer.is_linear) {
			auto& linear = layer.linear_attn;
			linear.num_k_heads = text_cfg.linear_num_key_heads;
			linear.num_v_heads = text_cfg.linear_num_value_heads;
			linear.head_k_dim = text_cfg.linear_key_head_dim;
			linear.head_v_dim = text_cfg.linear_value_head_dim;
			linear.key_dim = linear.num_k_heads * linear.head_k_dim;
			linear.value_dim = linear.num_v_heads * linear.head_v_dim;
			linear.conv_dim = linear.key_dim * 2 + linear.value_dim;
			linear.conv_kernel_size = text_cfg.linear_conv_kernel_dim;
			linear.norm_eps = text_cfg.rms_norm_eps;
			linear.conv1d_w = load_dense(
				weights,
				{
					mlx_prefix + ".linear_attn.conv1d.weight",
					raw_prefix + ".linear_attn.conv1d.weight",
				},
				true);
			linear.in_proj_qkv_w = load_param(
				weights,
				{
					mlx_prefix + ".linear_attn.in_proj_qkv",
					raw_prefix + ".linear_attn.in_proj_qkv",
				},
				quant_config);
			linear.in_proj_z_w = load_param(
				weights,
				{
					mlx_prefix + ".linear_attn.in_proj_z",
					raw_prefix + ".linear_attn.in_proj_z",
				},
				quant_config);
			linear.in_proj_b_w = load_param(
				weights,
				{
					mlx_prefix + ".linear_attn.in_proj_b",
					raw_prefix + ".linear_attn.in_proj_b",
				},
				quant_config);
			linear.in_proj_a_w = load_param(
				weights,
				{
					mlx_prefix + ".linear_attn.in_proj_a",
					raw_prefix + ".linear_attn.in_proj_a",
				},
				quant_config);
			linear.dt_bias = load_dense(
				weights,
				{
					mlx_prefix + ".linear_attn.dt_bias",
					raw_prefix + ".linear_attn.dt_bias",
				});
			linear.A_log = load_dense(
				weights,
				{
					mlx_prefix + ".linear_attn.A_log",
					raw_prefix + ".linear_attn.A_log",
				});
			linear.norm_w = load_dense(
				weights,
				{
					mlx_prefix + ".linear_attn.norm.weight",
					raw_prefix + ".linear_attn.norm.weight",
				});
			linear.out_proj_w = load_param(
				weights,
				{
					mlx_prefix + ".linear_attn.out_proj",
					raw_prefix + ".linear_attn.out_proj",
				},
				quant_config);
		} else {
			auto& attn = layer.self_attn;
			attn.num_heads = text_cfg.num_attention_heads;
			attn.num_kv_heads = text_cfg.num_key_value_heads;
			attn.head_dim = text_cfg.head_dim;
			attn.scale = 1.0f / std::sqrt(static_cast<float>(attn.head_dim));
			attn.norm_eps = text_cfg.rms_norm_eps;
			attn.q_proj_w = load_param(
				weights,
				{
					mlx_prefix + ".self_attn.q_proj",
					raw_prefix + ".self_attn.q_proj",
				},
				quant_config);
			attn.k_proj_w = load_param(
				weights,
				{
					mlx_prefix + ".self_attn.k_proj",
					raw_prefix + ".self_attn.k_proj",
				},
				quant_config);
			attn.v_proj_w = load_param(
				weights,
				{
					mlx_prefix + ".self_attn.v_proj",
					raw_prefix + ".self_attn.v_proj",
				},
				quant_config);
			attn.o_proj_w = load_param(
				weights,
				{
					mlx_prefix + ".self_attn.o_proj",
					raw_prefix + ".self_attn.o_proj",
				},
				quant_config);
			attn.q_norm_w = load_dense(
				weights,
				{
					mlx_prefix + ".self_attn.q_norm.weight",
					raw_prefix + ".self_attn.q_norm.weight",
				});
			attn.k_norm_w = load_dense(
				weights,
				{
					mlx_prefix + ".self_attn.k_norm.weight",
					raw_prefix + ".self_attn.k_norm.weight",
				});
			attn.rope = qwen3_5_mlx::RotaryEmbedding(
				text_cfg.rotary_dim(),
				static_cast<float>(text_cfg.rope_parameters.rope_theta),
				text_cfg.rope_parameters.mrope_section);
		}
	}

	if (weights.count("language_model.lm_head.weight") > 0 || weights.count("model.lm_head.weight") > 0) {
		language_model.lm_head_w = load_param(
			weights,
			{
				"language_model.lm_head",
				"model.lm_head",
				"lm_head",
			},
			quant_config);
	}

	return language_model;
}

} // namespace test_mlx_quantized_utils