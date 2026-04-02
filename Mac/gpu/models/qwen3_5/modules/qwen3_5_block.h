#pragma once

#include "models/qwen3_5/modules/qwen3_5_gated_attention.h"
#include "models/qwen3_5/modules/qwen3_5_gated_deltanet.h"
#include "models/qwen3_5/modules/qwen3_5_mlp.h"
#include "models/qwen3_5/qwen3_5_architecture.h"

namespace soc::gpu::models::qwen3_5 {

struct Qwen3_5BlockWeights {
    DeviceTensor input_layernorm_weight;
    Qwen3_5GatedDeltaNetWeights linear;
    Qwen3_5GatedAttentionWeights attention;
    DeviceTensor post_attention_layernorm_weight;
    Qwen3_5MlpWeights mlp;
};

struct Qwen3_5BlockParams {
    Qwen3_5LayerType layer_type = Qwen3_5LayerType::kGatedDeltaNet;
    Qwen3_5GatedDeltaNetParams linear;
    Qwen3_5GatedAttentionParams attention;
    Qwen3_5MlpParams mlp;
};

}  // namespace soc::gpu::models::qwen3_5
