#pragma once

#include "language.h"
#include "vision.h"
#include "weight_loader.h"

#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace qwen3_5_mlx {
namespace mx = mlx::core;

struct InputEmbeddingsFeatures {
	mx::array inputs_embeds = scalar_array();
	std::optional<mx::array> position_ids;
};

struct Model {
	Qwen3_5Config config;
	VisionModel vision_tower;
	LanguageModel language_model;

	Model()
		: vision_tower(Qwen3_5VisionConfig{}),
		  language_model(Qwen3_5TextConfig{}, Qwen3_5Config{}) {}

	explicit Model(const Qwen3_5Config& cfg)
		: config(cfg),
		  vision_tower(cfg.vision_config),
		  language_model(cfg.text_config, cfg) {}

	static std::pair<mx::array, mx::array> merge_input_ids_with_image_features(
		const mx::array& image_features,
		const mx::array& inputs_embeds,
		const mx::array& input_ids,
		int image_token_index,
		int video_token_index) {
		auto embeds_f32 = mx::astype(inputs_embeds, mx::float32);
		auto features_f32 = mx::astype(image_features, mx::float32);
		auto ids_i32 = mx::astype(input_ids, mx::int32);
		mx::eval(embeds_f32);
		mx::eval(features_f32);
		mx::eval(ids_i32);

		int batch = embeds_f32.shape(0);
		int seq_len = embeds_f32.shape(1);
		int hidden = embeds_f32.shape(2);
		int rows = features_f32.shape(0);

		const float* embed_ptr = embeds_f32.data<float>();
		const float* feature_ptr = features_f32.data<float>();
		const int* id_ptr = ids_i32.data<int>();

		std::vector<float> merged(embed_ptr, embed_ptr + static_cast<size_t>(batch * seq_len * hidden));
		std::vector<uint8_t> mask(static_cast<size_t>(batch * seq_len), 0);
		int feature_row = 0;
		for (int batch_index = 0; batch_index < batch; ++batch_index) {
			for (int token_index = 0; token_index < seq_len; ++token_index) {
				int token = id_ptr[batch_index * seq_len + token_index];
				if (token != image_token_index && token != video_token_index) {
					continue;
				}
				if (feature_row >= rows) {
					throw std::invalid_argument("Image features and image token count do not match");
				}

				mask[batch_index * seq_len + token_index] = 1;
				float* dst = merged.data() + ((batch_index * seq_len + token_index) * hidden);
				const float* src = feature_ptr + (feature_row * hidden);
				std::copy(src, src + hidden, dst);
				++feature_row;
			}
		}

		if (feature_row != rows) {
			throw std::invalid_argument("Unused image features remain after merge");
		}

		return {
			mx::astype(mx::array(merged.data(), mx::Shape{batch, seq_len, hidden}, mx::float32), inputs_embeds.dtype()),
			mx::astype(mx::array(mask.data(), mx::Shape{batch, seq_len}, mx::uint8), mx::bool_),
		};
	}

	InputEmbeddingsFeatures get_input_embeddings(
		const mx::array& input_ids,
		const std::optional<mx::array>& pixel_values = std::nullopt,
		const std::optional<mx::array>& image_grid_thw = std::nullopt,
		const std::optional<mx::array>& video_grid_thw = std::nullopt,
		const std::optional<mx::array>& mask = std::nullopt) {
		auto inputs_embeds = mlx_helpers::embedding(language_model.model.embed_w, input_ids);
		if (!pixel_values.has_value()) {
			return {inputs_embeds, std::nullopt};
		}

		auto grid = image_grid_thw.has_value() ? image_grid_thw : video_grid_thw;
		auto [image_features, _] = vision_tower.forward(*pixel_values, grid);
		auto [merged_embeds, __] = merge_input_ids_with_image_features(
			image_features,
			inputs_embeds,
			input_ids,
			config.image_token_id,
			config.video_token_id);

		std::optional<mx::array> position_ids = std::nullopt;
		if (image_grid_thw.has_value() || video_grid_thw.has_value()) {
			position_ids = language_model.get_rope_index(
				input_ids,
				image_grid_thw,
				video_grid_thw,
				mask).first;
		}

		return {merged_embeds, position_ids};
	}

	mx::array forward(
		const mx::array& input_ids,
		std::vector<LayerCache>& cache,
		const std::optional<mx::array>& pixel_values = std::nullopt,
		const std::optional<mx::array>& image_grid_thw = std::nullopt,
		const std::optional<mx::array>& video_grid_thw = std::nullopt,
		const std::optional<mx::array>& mask = std::nullopt,
		const std::optional<mx::array>& position_ids = std::nullopt) {
		auto embeddings = get_input_embeddings(
			input_ids,
			pixel_values,
			image_grid_thw,
			video_grid_thw,
			mask);

		auto rope_ids = position_ids.has_value() ? position_ids : embeddings.position_ids;
		return language_model.forward(
			input_ids,
			cache,
			mask,
			rope_ids,
			embeddings.inputs_embeds,
			image_grid_thw,
			video_grid_thw);
	}

	std::unordered_map<std::string, mx::array> sanitize(
		const std::unordered_map<std::string, mx::array>& weights) const {
		std::unordered_map<std::string, mx::array> sanitized;
		for (const auto& [key, value] : weights) {
			if (key.find("mtp.") != std::string::npos) {
				continue;
			}
			sanitized.insert_or_assign(key, value);
		}
		return sanitized;
	}
};

} // namespace qwen3_5_mlx
