#pragma once

#include "config.h"
#include "gated_delta.h"
#include "mlx_helpers.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace qwen3_5_mlx {
namespace mx = mlx::core;

inline mx::array scalar_array(float value = 0.0f) {
	return mx::array(value);
}

inline int default_full_attention_interval(const Qwen3_5TextConfig& cfg) {
	if (!cfg.layer_types.empty()) {
		for (size_t index = 0; index < cfg.layer_types.size(); ++index) {
			if (cfg.layer_types[index] == LayerType::FullAttention) {
				return static_cast<int>(index) + 1;
			}
		}
	}
	return 4;
}

inline bool is_linear_layer(const Qwen3_5TextConfig& cfg, int layer_index) {
	if (!cfg.layer_types.empty() && layer_index < static_cast<int>(cfg.layer_types.size())) {
		return cfg.layer_types[layer_index] == LayerType::LinearAttention;
	}
	return ((layer_index + 1) % default_full_attention_interval(cfg)) != 0;
}

struct RotaryEmbedding {
	int dim = 0;
	float base = 10000.0f;
	std::vector<int> mrope_section = {11, 11, 0};
	mx::array inv_freq = scalar_array();

	RotaryEmbedding() = default;

	RotaryEmbedding(int rotary_dim, float rope_base, std::vector<int> sections)
		: dim(rotary_dim), base(rope_base), mrope_section(std::move(sections)) {
		auto arange = mx::arange(0, dim, 2, mx::float32);
		auto base_arr = mx::array(base, mx::float32);
		inv_freq = 1.0f / mx::power(base_arr, arange / static_cast<float>(dim));
	}

	std::pair<mx::array, mx::array> operator()(mx::Dtype dtype, mx::array position_ids) const {
		if (position_ids.ndim() == 2) {
			position_ids = mx::broadcast_to(
				mx::expand_dims(position_ids, 0),
				{3, position_ids.shape(0), position_ids.shape(1)});
		}

		auto pos_i32 = mx::astype(position_ids, mx::int32);
		auto inv_f32 = mx::astype(inv_freq, mx::float32);
		mx::eval(pos_i32);
		mx::eval(inv_f32);

		int batch = pos_i32.shape(1);
		int seq_len = pos_i32.shape(2);
		int half_dim = inv_f32.shape(0);
		const int* pos_ptr = pos_i32.data<int>();
		const float* inv_ptr = inv_f32.data<float>();

		std::vector<float> merged(static_cast<size_t>(batch * seq_len * half_dim));
		for (int batch_index = 0; batch_index < batch; ++batch_index) {
			for (int token_index = 0; token_index < seq_len; ++token_index) {
				for (int freq_index = 0; freq_index < half_dim; ++freq_index) {
					float value = static_cast<float>(
						pos_ptr[((0 * batch + batch_index) * seq_len) + token_index]) *
						inv_ptr[freq_index];

					if (mrope_section.size() > 1 && freq_index % 3 == 1) {
						value = static_cast<float>(
							pos_ptr[((1 * batch + batch_index) * seq_len) + token_index]) *
							inv_ptr[freq_index];
					} else if (mrope_section.size() > 2 && freq_index % 3 == 2) {
						value = static_cast<float>(
							pos_ptr[((2 * batch + batch_index) * seq_len) + token_index]) *
							inv_ptr[freq_index];
					}

					merged[((batch_index * seq_len) + token_index) * half_dim + freq_index] = value;
				}
			}
		}

		std::vector<float> emb(static_cast<size_t>(batch * seq_len * dim));
		for (int batch_index = 0; batch_index < batch; ++batch_index) {
			for (int token_index = 0; token_index < seq_len; ++token_index) {
				for (int freq_index = 0; freq_index < half_dim; ++freq_index) {
					float value = merged[((batch_index * seq_len) + token_index) * half_dim + freq_index];
					emb[((batch_index * seq_len) + token_index) * dim + freq_index] = value;
					emb[((batch_index * seq_len) + token_index) * dim + half_dim + freq_index] = value;
				}
			}
		}

		auto emb_arr = mx::array(emb.data(), mx::Shape{batch, seq_len, dim}, mx::float32);
		return {mx::astype(mx::cos(emb_arr), dtype), mx::astype(mx::sin(emb_arr), dtype)};
	}
};

inline std::pair<mx::array, mx::array> apply_rotary_pos_emb(
	const mx::array& q,
	const mx::array& k,
	const mx::array& cos,
	const mx::array& sin,
	int unsqueeze_dim = 1) {
	auto cos_e = mx::expand_dims(cos, unsqueeze_dim);
	auto sin_e = mx::expand_dims(sin, unsqueeze_dim);
	int rotary_dim = cos.shape(cos.ndim() - 1);

	auto q_parts = mx::split(q, mx::Shape{rotary_dim}, q.ndim() - 1);
	auto k_parts = mx::split(k, mx::Shape{rotary_dim}, k.ndim() - 1);

	auto q_embed = (q_parts[0] * cos_e) + (mlx_helpers::rotate_half(q_parts[0]) * sin_e);
	auto k_embed = (k_parts[0] * cos_e) + (mlx_helpers::rotate_half(k_parts[0]) * sin_e);

	return {
		mx::concatenate({q_embed, q_parts[1]}, q.ndim() - 1),
		mx::concatenate({k_embed, k_parts[1]}, k.ndim() - 1),
	};
}

struct KVCache {
	std::optional<mx::array> keys;
	std::optional<mx::array> values;
	int offset = 0;

	bool empty() const {
		return !keys.has_value() || !values.has_value();
	}

	std::pair<mx::array, mx::array> update_and_fetch(
		const mx::array& new_keys,
		const mx::array& new_values) {
		if (empty()) {
			keys = new_keys;
			values = new_values;
		} else {
			keys = mx::concatenate({*keys, new_keys}, 2);
			values = mx::concatenate({*values, new_values}, 2);
		}
		offset += new_keys.shape(2);
		return {*keys, *values};
	}
};

struct ArraysCache {
	std::optional<mx::array> conv_state;
	std::optional<mx::array> rec_state;
};

using LayerCache = std::variant<ArraysCache, KVCache>;

struct RMSNormGated {
	mx::array weight = scalar_array(1.0f);
	float eps = 1e-6f;

	mx::array forward(
		const mx::array& hidden_states,
		const std::optional<mx::array>& gate = std::nullopt) const {
		auto out = mx::fast::rms_norm(hidden_states, weight, eps);
		if (gate.has_value()) {
			out = mlx_helpers::swiglu(*gate, out);
		}
		return mx::astype(out, hidden_states.dtype());
	}
};

struct MLP {
	mlx_helpers::TensorParam gate_proj_w;
	mlx_helpers::TensorParam up_proj_w;
	mlx_helpers::TensorParam down_proj_w;

	mx::array forward(const mx::array& x) const {
		auto gate = mlx_helpers::linear(x, gate_proj_w);
		auto up = mlx_helpers::linear(x, up_proj_w);
		return mlx_helpers::linear(mlx_helpers::swiglu(gate, up), down_proj_w);
	}
};

struct Attention {
	int num_heads = 0;
	int num_kv_heads = 0;
	int head_dim = 0;
	float scale = 1.0f;
	float norm_eps = 1e-6f;
	mlx_helpers::TensorParam q_proj_w;
	mlx_helpers::TensorParam k_proj_w;
	mlx_helpers::TensorParam v_proj_w;
	mlx_helpers::TensorParam o_proj_w;
	mx::array q_norm_w = scalar_array(1.0f);
	mx::array k_norm_w = scalar_array(1.0f);
	RotaryEmbedding rope;

	mx::array forward(
		const mx::array& x,
		KVCache& cache,
		const std::optional<mx::array>& mask = std::nullopt,
		const std::optional<mx::array>& position_ids = std::nullopt) const {
		int batch = x.shape(0);
		int seq_len = x.shape(1);

		auto q_proj_out = mlx_helpers::linear(x, q_proj_w);
		q_proj_out = mx::reshape(q_proj_out, {batch, seq_len, num_heads, head_dim * 2});
		auto q_parts = mx::split(q_proj_out, 2, 3);
		auto queries = q_parts[0];
		auto gate = mx::reshape(q_parts[1], {batch, seq_len, num_heads * head_dim});

		auto keys = mx::reshape(
			mlx_helpers::linear(x, k_proj_w),
			{batch, seq_len, num_kv_heads, head_dim});
		auto values = mx::reshape(
			mlx_helpers::linear(x, v_proj_w),
			{batch, seq_len, num_kv_heads, head_dim});

		queries = mx::fast::rms_norm(queries, q_norm_w, norm_eps);
		keys = mx::fast::rms_norm(keys, k_norm_w, norm_eps);

		queries = mx::transpose(queries, {0, 2, 1, 3});
		keys = mx::transpose(keys, {0, 2, 1, 3});
		values = mx::transpose(values, {0, 2, 1, 3});

		auto pos = position_ids.value_or(mlx_helpers::make_position_ids(seq_len, cache.offset, batch));
		auto [cos, sin] = rope(x.dtype(), pos);
		auto [rotated_q, rotated_k] = apply_rotary_pos_emb(queries, keys, cos, sin);

		auto [full_k, full_v] = cache.update_and_fetch(rotated_k, values);

		auto output = mask.has_value()
			? mx::fast::scaled_dot_product_attention(rotated_q, full_k, full_v, scale, "array", *mask)
			: mx::fast::scaled_dot_product_attention(rotated_q, full_k, full_v, scale, "causal");
		output = mx::transpose(output, {0, 2, 1, 3});
		output = mx::reshape(output, {batch, seq_len, num_heads * head_dim});
		return mlx_helpers::linear(output * mx::sigmoid(gate), o_proj_w);
	}
};

struct GatedDeltaNet {
	int num_k_heads = 0;
	int num_v_heads = 0;
	int head_k_dim = 0;
	int head_v_dim = 0;
	int key_dim = 0;
	int value_dim = 0;
	int conv_dim = 0;
	int conv_kernel_size = 4;
	float norm_eps = 1e-6f;

	mx::array conv1d_w = scalar_array();
	mlx_helpers::TensorParam in_proj_qkv_w;
	mlx_helpers::TensorParam in_proj_z_w;
	mlx_helpers::TensorParam in_proj_b_w;
	mlx_helpers::TensorParam in_proj_a_w;
	mx::array dt_bias = scalar_array();
	mx::array A_log = scalar_array();
	mx::array norm_w = scalar_array(1.0f);
	mlx_helpers::TensorParam out_proj_w;

	mx::array forward(
		const mx::array& inputs,
		ArraysCache& cache,
		const std::optional<mx::array>& mask = std::nullopt) const {
		int batch = inputs.shape(0);
		int seq_len = inputs.shape(1);

		auto mixed_qkv = mlx_helpers::linear(inputs, in_proj_qkv_w);
		auto z = mx::reshape(mlx_helpers::linear(inputs, in_proj_z_w),
			{batch, seq_len, num_v_heads, head_v_dim});
		auto b = mlx_helpers::linear(inputs, in_proj_b_w);
		auto a = mlx_helpers::linear(inputs, in_proj_a_w);

		mx::array conv_state = cache.conv_state.value_or(
			mx::zeros({batch, conv_kernel_size - 1, conv_dim}, inputs.dtype()));
		if (mask.has_value()) {
			mixed_qkv = mx::where(
				mx::expand_dims(*mask, -1),
				mixed_qkv,
				mx::zeros_like(mixed_qkv));
		}

		auto conv_input = mx::concatenate({conv_state, mixed_qkv}, 1);
		cache.conv_state = mlx_helpers::take_last_tokens(conv_input, conv_kernel_size - 1);
		auto conv_out = mlx_helpers::silu(
			mx::conv1d(conv_input, conv1d_w, 1, 0, 1, conv_dim));

		auto qkv_parts = mx::split(conv_out, {key_dim, key_dim * 2}, conv_out.ndim() - 1);
		auto q = mx::reshape(qkv_parts[0], {batch, seq_len, num_k_heads, head_k_dim});
		auto k = mx::reshape(qkv_parts[1], {batch, seq_len, num_k_heads, head_k_dim});
		auto v = mx::reshape(qkv_parts[2], {batch, seq_len, num_v_heads, head_v_dim});

		float inv_scale = std::pow(static_cast<float>(head_k_dim), -0.5f);
		q = (inv_scale * inv_scale) * mx::fast::rms_norm(q, std::nullopt, 1e-6f);
		k = inv_scale * mx::fast::rms_norm(k, std::nullopt, 1e-6f);

		auto rec_state = cache.rec_state;
		auto [out, next_state] = gated_delta_update(
			q, k, v, a, b, A_log, dt_bias, rec_state, mask, true);
		cache.rec_state = next_state;

		RMSNormGated norm{norm_w, norm_eps};
		out = norm.forward(out, z);
		out = mx::reshape(out, {batch, seq_len, value_dim});
		return mlx_helpers::linear(out, out_proj_w);
	}
};

struct DecoderLayer {
	bool is_linear = true;
	float ln_eps = 1e-6f;
	mx::array input_ln_w = scalar_array(1.0f);
	mx::array post_attn_ln_w = scalar_array(1.0f);
	GatedDeltaNet linear_attn;
	Attention self_attn;
	MLP mlp;

	mx::array forward(
		const mx::array& x,
		LayerCache& cache,
		const std::optional<mx::array>& mask = std::nullopt,
		const std::optional<mx::array>& position_ids = std::nullopt) const {
		auto normed = mx::fast::rms_norm(x, input_ln_w, ln_eps);
		auto residual = is_linear
			? linear_attn.forward(std::move(normed), std::get<ArraysCache>(cache), mask)
			: self_attn.forward(std::move(normed), std::get<KVCache>(cache), mask, position_ids);
		auto hidden = x + residual;
		auto mlp_out = mlp.forward(mx::fast::rms_norm(hidden, post_attn_ln_w, ln_eps));
		return hidden + mlp_out;
	}
};

struct Qwen3_5Model {
	Qwen3_5TextConfig config;
	mlx_helpers::TensorParam embed_w;
	std::vector<DecoderLayer> layers;
	mx::array norm_w = scalar_array(1.0f);
	float norm_eps = 1e-6f;
	int ssm_idx = 0;
	int fa_idx = 0;

	mx::array forward(
		const mx::array& input_ids,
		std::vector<LayerCache>& cache,
		const std::optional<mx::array>& attention_mask = std::nullopt,
		const std::optional<mx::array>& position_ids = std::nullopt,
		const std::optional<mx::array>& inputs_embeds = std::nullopt) const {
		auto hidden = inputs_embeds.value_or(mlx_helpers::embedding(embed_w, input_ids));
		auto pos = position_ids.value_or(mlx_helpers::make_position_ids(
			hidden.shape(1), 0, hidden.shape(0)));

		for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
			auto layer_mask = layers[layer_index].is_linear ? attention_mask : std::nullopt;
			hidden = layers[layer_index].forward(hidden, cache[layer_index], layer_mask, pos);
		}

		return mx::fast::rms_norm(hidden, norm_w, norm_eps);
	}
};

struct LanguageModel {
	Qwen3_5TextConfig config;
	std::optional<Qwen3_5Config> model_config;
	Qwen3_5Model model;
	std::optional<mlx_helpers::TensorParam> lm_head_w;

	LanguageModel() = default;

	LanguageModel(const Qwen3_5TextConfig& text_config, const std::optional<Qwen3_5Config>& cfg = std::nullopt)
		: config(text_config), model_config(cfg) {
		model.config = text_config;
		model.norm_eps = text_config.rms_norm_eps;
		model.fa_idx = std::max(0, default_full_attention_interval(text_config) - 1);
	}

	static mx::array make_position_ids(int seq_len, int offset, int batch_size = 1) {
		return mlx_helpers::make_position_ids(seq_len, offset, batch_size);
	}

	static std::vector<LayerCache> make_cache(const Qwen3_5TextConfig& cfg) {
		std::vector<LayerCache> cache;
		cache.reserve(cfg.num_hidden_layers);
		for (int layer = 0; layer < cfg.num_hidden_layers; ++layer) {
			bool is_linear = is_linear_layer(cfg, layer);
			if (is_linear) {
				cache.emplace_back(ArraysCache{});
			} else {
				cache.emplace_back(KVCache{});
			}
		}
		return cache;
	}

	std::pair<mx::array, mx::array> get_rope_index(
		const mx::array& input_ids,
		const std::optional<mx::array>& image_grid_thw = std::nullopt,
		const std::optional<mx::array>& video_grid_thw = std::nullopt,
		const std::optional<mx::array>& attention_mask = std::nullopt) const {
		int batch = input_ids.shape(0);
		int seq_len = input_ids.shape(1);
		if (!image_grid_thw.has_value() && !video_grid_thw.has_value()) {
			auto pos = make_position_ids(seq_len, 0, batch);
			auto deltas = mx::zeros({batch, 1}, mx::int32);
			if (!attention_mask.has_value()) {
				return {pos, deltas};
			}

			auto mask_i32 = mx::astype(*attention_mask, mx::int32);
			mx::eval(mask_i32);
			const int* mask_ptr = mask_i32.data<int>();
			std::vector<int> data(static_cast<size_t>(3 * batch * seq_len), 1);
			std::vector<int> delta_data(static_cast<size_t>(batch), 0);
			for (int batch_index = 0; batch_index < batch; ++batch_index) {
				int running = -1;
				int max_pos = 0;
				for (int token_index = 0; token_index < seq_len; ++token_index) {
					if (mask_ptr[batch_index * seq_len + token_index] != 0) {
						++running;
						max_pos = running;
						for (int component = 0; component < 3; ++component) {
							data[((component * batch + batch_index) * seq_len) + token_index] = running;
						}
					}
				}
				delta_data[batch_index] = max_pos + 1 - seq_len;
			}
			return {
				mx::array(data.data(), mx::Shape{3, batch, seq_len}, mx::int32),
				mx::array(delta_data.data(), mx::Shape{batch, 1}, mx::int32),
			};
		}

		auto image_grids = mlx_helpers::to_grid_vector(image_grid_thw);
		auto video_grids = mlx_helpers::to_grid_vector(video_grid_thw);
		auto ids_i32 = mx::astype(input_ids, mx::int32);
		mx::eval(ids_i32);
		const int* ids_ptr = ids_i32.data<int>();

		std::vector<int> mask_values(static_cast<size_t>(batch * seq_len), 1);
		if (attention_mask.has_value()) {
			auto mask_i32 = mx::astype(*attention_mask, mx::int32);
			mx::eval(mask_i32);
			const int* mask_ptr = mask_i32.data<int>();
			std::copy(mask_ptr, mask_ptr + batch * seq_len, mask_values.begin());
		}

		int image_index = 0;
		int video_index = 0;
		std::vector<int> position_data(static_cast<size_t>(3 * batch * seq_len), 1);
		std::vector<int> delta_data(static_cast<size_t>(batch), 0);
		int image_token_id = model_config ? model_config->image_token_id : 248056;
		int video_token_id = model_config ? model_config->video_token_id : 248057;
		int spatial_merge = model_config ? model_config->vision_config.spatial_merge_size : 2;

		for (int batch_index = 0; batch_index < batch; ++batch_index) {
			int cursor = 0;
			int next_position = 0;
			while (cursor < seq_len) {
				while (cursor < seq_len && mask_values[batch_index * seq_len + cursor] == 0) {
					++cursor;
				}
				if (cursor >= seq_len) {
					break;
				}

				int token = ids_ptr[batch_index * seq_len + cursor];
				if (token == image_token_id || token == video_token_id) {
					std::array<int, 3> grid{1, 1, 1};
					if (token == image_token_id && image_index < static_cast<int>(image_grids.size())) {
						grid = image_grids[image_index++];
					} else if (token == video_token_id && video_index < static_cast<int>(video_grids.size())) {
						grid = video_grids[video_index++];
					}

					int llm_t = std::max(1, grid[0]);
					int llm_h = std::max(1, grid[1] / spatial_merge);
					int llm_w = std::max(1, grid[2] / spatial_merge);
					for (int t = 0; t < llm_t && cursor < seq_len; ++t) {
						for (int h = 0; h < llm_h && cursor < seq_len; ++h) {
							for (int w = 0; w < llm_w && cursor < seq_len; ++w) {
								position_data[((0 * batch + batch_index) * seq_len) + cursor] = next_position + t;
								position_data[((1 * batch + batch_index) * seq_len) + cursor] = next_position + h;
								position_data[((2 * batch + batch_index) * seq_len) + cursor] = next_position + w;
								++cursor;
							}
						}
					}
					next_position += std::max({llm_t, llm_h, llm_w});
				} else {
					position_data[((0 * batch + batch_index) * seq_len) + cursor] = next_position;
					position_data[((1 * batch + batch_index) * seq_len) + cursor] = next_position;
					position_data[((2 * batch + batch_index) * seq_len) + cursor] = next_position;
					++cursor;
					++next_position;
				}
			}
			delta_data[batch_index] = next_position - seq_len;
		}

		return {
			mx::array(position_data.data(), mx::Shape{3, batch, seq_len}, mx::int32),
			mx::array(delta_data.data(), mx::Shape{batch, 1}, mx::int32),
		};
	}

	mx::array forward(
		const mx::array& input_ids,
		std::vector<LayerCache>& cache,
		const std::optional<mx::array>& attention_mask = std::nullopt,
		const std::optional<mx::array>& position_ids = std::nullopt,
		const std::optional<mx::array>& inputs_embeds = std::nullopt,
		const std::optional<mx::array>& image_grid_thw = std::nullopt,
		const std::optional<mx::array>& video_grid_thw = std::nullopt) const {
		int offset = 0;
		for (const auto& layer_cache : cache) {
			if (std::holds_alternative<KVCache>(layer_cache)) {
				offset = std::max(offset, std::get<KVCache>(layer_cache).offset);
			}
		}

		auto pos = position_ids;
		if (!pos.has_value()) {
			if (image_grid_thw.has_value() || video_grid_thw.has_value()) {
				pos = get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask).first;
			} else {
				pos = make_position_ids(input_ids.shape(1), offset, input_ids.shape(0));
			}
		}

		auto hidden = model.forward(input_ids, cache, attention_mask, pos, inputs_embeds);
		if (config.tie_word_embeddings || !lm_head_w.has_value()) {
			return mlx_helpers::linear(hidden, model.embed_w);
		}
		return mlx_helpers::linear(hidden, *lm_head_w);
	}
};

struct GenerationSession {
	LanguageModel* language_model = nullptr;
	std::vector<LayerCache> cache;
	int eos_token_id = 248046;

	explicit GenerationSession(LanguageModel& model, int eos = 248046)
		: language_model(&model),
		  cache(LanguageModel::make_cache(model.config)),
		  eos_token_id(eos) {}

	std::vector<int> generate(const std::vector<int>& prompt_tokens, int max_new_tokens) {
		auto input = mx::array(
			prompt_tokens.data(),
			mx::Shape{1, static_cast<int>(prompt_tokens.size())},
			mx::int32);
		auto logits = language_model->forward(input, cache);
		mx::eval(logits);

		auto current = mx::argmax(
			mlx_helpers::slice_axis(logits, 1, logits.shape(1) - 1, logits.shape(1)),
			-1);
		mx::eval(current);
		int token = current.item<int>();
		std::vector<int> output{token};

		for (int step = 0; step < max_new_tokens && token != eos_token_id; ++step) {
			auto decode_input = mx::array(&token, mx::Shape{1, 1}, mx::int32);
			logits = language_model->forward(decode_input, cache);
			mx::eval(logits);
			current = mx::argmax(logits, -1);
			mx::eval(current);
			token = current.item<int>();
			output.push_back(token);
		}

		return output;
	}
};

} // namespace qwen3_5_mlx
