#pragma once

#include <string>
#include <vector>

#include "asset/runtime_assets.h"
#include "models/qwen3_5/qwen3_5_architecture.h"

namespace soc::gpu::models::qwen3_5 {

struct Qwen3_5LayerManifestMetadata {
    std::size_t layer_index = 0;
    Qwen3_5LayerType layer_type = Qwen3_5LayerType::kGatedDeltaNet;
    TensorRecord input_layernorm;
    TensorRecord post_attention_layernorm;
    TensorRecord mlp_gate_proj;
    TensorRecord mlp_up_proj;
    TensorRecord mlp_down_proj;

    TensorRecord self_attn_q_proj;
    TensorRecord self_attn_k_proj;
    TensorRecord self_attn_v_proj;
    TensorRecord self_attn_o_proj;
    TensorRecord self_attn_q_norm;
    TensorRecord self_attn_k_norm;

    TensorRecord linear_attn_norm;
    TensorRecord linear_attn_in_proj_qkv;
    TensorRecord linear_attn_in_proj_z;
    TensorRecord linear_attn_in_proj_a;
    TensorRecord linear_attn_in_proj_b;
    TensorRecord linear_attn_out_proj;
    TensorRecord linear_attn_conv1d_weight;
    TensorRecord linear_attn_a_log;
    TensorRecord linear_attn_dt_bias;
};

struct Qwen3_5ManifestMetadata {
    TensorRecord embed_tokens;
    TensorRecord final_norm;
    TensorRecord lm_head;
    std::vector<Qwen3_5LayerManifestMetadata> layers;
};

bool ResolveManifestMetadata(const ManifestData& manifest,
                             const Qwen3_5ArchitectureSpec& spec,
                             Qwen3_5ManifestMetadata* metadata,
                             std::string* error_message);

}  // namespace soc::gpu::models::qwen3_5
