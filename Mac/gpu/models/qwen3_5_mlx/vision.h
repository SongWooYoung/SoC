#pragma once

#include "config.h"
#include "mlx_helpers.h"

#include <optional>
#include <utility>

namespace qwen3_5_mlx {
namespace mx = mlx::core;

struct VisionModel {
	Qwen3_5VisionConfig config;
	std::optional<mx::array> patch_proj_w;

	VisionModel() = default;
	explicit VisionModel(const Qwen3_5VisionConfig& cfg) : config(cfg) {}

	mx::array project(const mx::array& pixel_values) const {
		mx::array features = pixel_values;
		if (features.ndim() == 1) {
			features = mx::reshape(features, {1, features.shape(0)});
		} else if (features.ndim() > 2) {
			int last_dim = features.shape(features.ndim() - 1);
			int token_count = 1;
			for (int axis = 0; axis < features.ndim() - 1; ++axis) {
				token_count *= features.shape(axis);
			}
			features = mx::reshape(features, {token_count, last_dim});
		}

		if (patch_proj_w.has_value()) {
			return mlx_helpers::linear(features, *patch_proj_w);
		}
		return mlx_helpers::adapt_last_dim(features, config.out_hidden_size);
	}

	std::pair<mx::array, std::optional<mx::array>> forward(
		const mx::array& pixel_values,
		const std::optional<mx::array>& grid_thw = std::nullopt) const {
		return {project(pixel_values), grid_thw};
	}
};

} // namespace qwen3_5_mlx
