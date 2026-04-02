#include "models/qwen3_5/qwen3_5_manifest_metadata.h"

#include <initializer_list>
#include <sstream>

namespace soc::gpu::models::qwen3_5 {
namespace {

bool HasShape(const TensorRecord& tensor, std::initializer_list<std::size_t> shape) {
    if (tensor.shape.size() != shape.size()) {
        return false;
    }
    std::size_t index = 0;
    for (const std::size_t dim : shape) {
        if (tensor.shape[index] != dim) {
            return false;
        }
        ++index;
    }
    return true;
}

std::string ShapeString(std::initializer_list<std::size_t> shape) {
    std::ostringstream stream;
    stream << "[";
    std::size_t index = 0;
    for (const std::size_t dim : shape) {
        if (index != 0) {
            stream << ", ";
        }
        stream << dim;
        ++index;
    }
    stream << "]";
    return stream.str();
}

bool RequireShape(const TensorRecord& tensor,
                  std::initializer_list<std::size_t> shape,
                  const std::string& logical_name,
                  std::string* error_message) {
    if (HasShape(tensor, shape)) {
        return true;
    }
    if (error_message != nullptr) {
        *error_message = "unexpected shape for " + logical_name + ": expected " + ShapeString(shape);
    }
    return false;
}

bool RequireRank(const TensorRecord& tensor,
                 const std::size_t rank,
                 const std::string& logical_name,
                 std::string* error_message) {
    if (tensor.shape.size() == rank) {
        return true;
    }
    if (error_message != nullptr) {
        *error_message = "unexpected rank for " + logical_name;
    }
    return false;
}

const TensorRecord* FindFirstTensorAlias(const ManifestData& manifest,
                                         std::initializer_list<std::string> aliases) {
    for (const std::string& alias : aliases) {
        if (const TensorRecord* tensor = manifest.FindTensorIfPresent(alias)) {
            return tensor;
        }
    }
    return nullptr;
}

bool RequireTensor(const ManifestData& manifest,
                   std::initializer_list<std::string> aliases,
                   const std::string& logical_name,
                   TensorRecord* output,
                   std::string* error_message) {
    if (output == nullptr) {
        return false;
    }
    if (const TensorRecord* tensor = FindFirstTensorAlias(manifest, aliases)) {
        *output = *tensor;
        return true;
    }
    if (error_message != nullptr) {
        *error_message = "missing required tensor for " + logical_name;
    }
    return false;
}

std::vector<std::string> LayerPrefixes(const std::size_t layer_index) {
    const std::string suffix = "layers." + std::to_string(layer_index) + ".";
    return {
        "model.language_model." + suffix,
        "model." + suffix,
        suffix,
    };
}

}  // namespace

bool ResolveManifestMetadata(const ManifestData& manifest,
                             const Qwen3_5ArchitectureSpec& spec,
                             Qwen3_5ManifestMetadata* metadata,
                             std::string* error_message) {
    if (metadata == nullptr) {
        if (error_message != nullptr) {
            *error_message = "ResolveManifestMetadata requires a non-null metadata output";
        }
        return false;
    }

    Qwen3_5ManifestMetadata parsed;
    if (!RequireTensor(manifest,
                       {"model.language_model.embed_tokens.weight", "model.embed_tokens.weight", "embed_tokens.weight"},
                       "embed_tokens",
                       &parsed.embed_tokens,
                       error_message) ||
        !RequireTensor(manifest,
                       {"model.language_model.norm.weight", "model.norm.weight", "norm.weight"},
                       "final_norm",
                       &parsed.final_norm,
                       error_message)) {
        return false;
    }

    if (!RequireTensor(manifest,
                       {"lm_head.weight", "model.language_model.lm_head.weight", "model.lm_head.weight"},
                       "lm_head",
                       &parsed.lm_head,
                       nullptr)) {
        if (spec.tie_word_embeddings) {
            parsed.lm_head = parsed.embed_tokens;
        } else {
            if (error_message != nullptr) {
                *error_message = "missing required tensor for lm_head";
            }
            return false;
        }
    }

    if (!RequireShape(parsed.embed_tokens,
                      {spec.vocab_size, spec.hidden_size},
                      "embed_tokens",
                      error_message) ||
        !RequireShape(parsed.final_norm, {spec.hidden_size}, "final_norm", error_message) ||
        !RequireShape(parsed.lm_head, {spec.vocab_size, spec.hidden_size}, "lm_head", error_message)) {
        return false;
    }

    const std::size_t attention_proj_dim = spec.num_attention_heads * spec.attention_head_dim;
    const std::size_t attention_q_proj_dim = attention_proj_dim * 2;
    const std::size_t kv_proj_dim = spec.num_key_value_heads * spec.attention_head_dim;
    const std::size_t linear_key_dim = spec.linear_num_key_heads * spec.linear_key_head_dim;
    const std::size_t linear_value_dim = spec.linear_num_value_heads * spec.linear_value_head_dim;
    const std::size_t linear_qkv_dim = linear_key_dim * 2 + linear_value_dim;

    parsed.layers.reserve(spec.num_hidden_layers);
    for (std::size_t layer_index = 0; layer_index < spec.num_hidden_layers; ++layer_index) {
        const std::vector<std::string> prefixes = LayerPrefixes(layer_index);
        const auto alias = [&](const std::string& suffix) {
            std::initializer_list<std::string> dummy{};
            (void)dummy;
            return std::vector<std::string>{
                prefixes[0] + suffix,
                prefixes[1] + suffix,
                prefixes[2] + suffix,
            };
        };

        Qwen3_5LayerManifestMetadata layer;
        layer.layer_index = layer_index;
        layer.layer_type = spec.layer_types[layer_index];

        auto require_layer_tensor = [&](const std::string& logical_name,
                                        const std::string& suffix,
                                        TensorRecord* output) -> bool {
            const std::vector<std::string> aliases = alias(suffix);
            return RequireTensor(manifest,
                                 {aliases[0], aliases[1], aliases[2]},
                                 logical_name,
                                 output,
                                 error_message);
        };

        if (!require_layer_tensor("input_layernorm", "input_layernorm.weight", &layer.input_layernorm) ||
            !require_layer_tensor("post_attention_layernorm",
                                  "post_attention_layernorm.weight",
                                  &layer.post_attention_layernorm) ||
            !require_layer_tensor("mlp.gate_proj", "mlp.gate_proj.weight", &layer.mlp_gate_proj) ||
            !require_layer_tensor("mlp.up_proj", "mlp.up_proj.weight", &layer.mlp_up_proj) ||
            !require_layer_tensor("mlp.down_proj", "mlp.down_proj.weight", &layer.mlp_down_proj)) {
            return false;
        }

        if (!RequireShape(layer.input_layernorm, {spec.hidden_size}, "input_layernorm", error_message) ||
            !RequireShape(layer.post_attention_layernorm,
                          {spec.hidden_size},
                          "post_attention_layernorm",
                          error_message) ||
            !RequireShape(layer.mlp_gate_proj,
                          {spec.intermediate_size, spec.hidden_size},
                          "mlp.gate_proj",
                          error_message) ||
            !RequireShape(layer.mlp_up_proj,
                          {spec.intermediate_size, spec.hidden_size},
                          "mlp.up_proj",
                          error_message) ||
            !RequireShape(layer.mlp_down_proj,
                          {spec.hidden_size, spec.intermediate_size},
                          "mlp.down_proj",
                          error_message)) {
            return false;
        }

        if (layer.layer_type == Qwen3_5LayerType::kGatedAttention) {
            if (!require_layer_tensor("self_attn.q_proj", "self_attn.q_proj.weight", &layer.self_attn_q_proj) ||
                !require_layer_tensor("self_attn.k_proj", "self_attn.k_proj.weight", &layer.self_attn_k_proj) ||
                !require_layer_tensor("self_attn.v_proj", "self_attn.v_proj.weight", &layer.self_attn_v_proj) ||
                !require_layer_tensor("self_attn.o_proj", "self_attn.o_proj.weight", &layer.self_attn_o_proj) ||
                !require_layer_tensor("self_attn.q_norm", "self_attn.q_norm.weight", &layer.self_attn_q_norm) ||
                !require_layer_tensor("self_attn.k_norm", "self_attn.k_norm.weight", &layer.self_attn_k_norm)) {
                return false;
            }

            if (!RequireShape(layer.self_attn_q_proj,
                              {attention_q_proj_dim, spec.hidden_size},
                              "self_attn.q_proj",
                              error_message) ||
                !RequireShape(layer.self_attn_k_proj,
                              {kv_proj_dim, spec.hidden_size},
                              "self_attn.k_proj",
                              error_message) ||
                !RequireShape(layer.self_attn_v_proj,
                              {kv_proj_dim, spec.hidden_size},
                              "self_attn.v_proj",
                              error_message) ||
                !RequireShape(layer.self_attn_o_proj,
                              {spec.hidden_size, attention_proj_dim},
                              "self_attn.o_proj",
                              error_message) ||
                !RequireShape(layer.self_attn_q_norm, {spec.attention_head_dim}, "self_attn.q_norm", error_message) ||
                !RequireShape(layer.self_attn_k_norm, {spec.attention_head_dim}, "self_attn.k_norm", error_message)) {
                return false;
            }
        } else {
            if (!require_layer_tensor("linear_attn.norm", "linear_attn.norm.weight", &layer.linear_attn_norm) ||
                !require_layer_tensor("linear_attn.in_proj_qkv",
                                      "linear_attn.in_proj_qkv.weight",
                                      &layer.linear_attn_in_proj_qkv) ||
                !require_layer_tensor("linear_attn.in_proj_z",
                                      "linear_attn.in_proj_z.weight",
                                      &layer.linear_attn_in_proj_z) ||
                !require_layer_tensor("linear_attn.in_proj_a",
                                      "linear_attn.in_proj_a.weight",
                                      &layer.linear_attn_in_proj_a) ||
                !require_layer_tensor("linear_attn.in_proj_b",
                                      "linear_attn.in_proj_b.weight",
                                      &layer.linear_attn_in_proj_b) ||
                !require_layer_tensor("linear_attn.out_proj",
                                      "linear_attn.out_proj.weight",
                                      &layer.linear_attn_out_proj) ||
                !require_layer_tensor("linear_attn.conv1d.weight",
                                      "linear_attn.conv1d.weight",
                                      &layer.linear_attn_conv1d_weight) ||
                !require_layer_tensor("linear_attn.A_log", "linear_attn.A_log", &layer.linear_attn_a_log) ||
                !require_layer_tensor("linear_attn.dt_bias", "linear_attn.dt_bias", &layer.linear_attn_dt_bias)) {
                return false;
            }

            if (!RequireShape(layer.linear_attn_norm,
                              {spec.linear_value_head_dim},
                              "linear_attn.norm",
                              error_message) ||
                !RequireShape(layer.linear_attn_in_proj_qkv,
                              {linear_qkv_dim, spec.hidden_size},
                              "linear_attn.in_proj_qkv",
                              error_message) ||
                !RequireShape(layer.linear_attn_in_proj_z,
                              {linear_value_dim, spec.hidden_size},
                              "linear_attn.in_proj_z",
                              error_message) ||
                !RequireShape(layer.linear_attn_in_proj_a,
                              {spec.linear_num_value_heads, spec.hidden_size},
                              "linear_attn.in_proj_a",
                              error_message) ||
                !RequireShape(layer.linear_attn_in_proj_b,
                              {spec.linear_num_value_heads, spec.hidden_size},
                              "linear_attn.in_proj_b",
                              error_message) ||
                !RequireShape(layer.linear_attn_out_proj,
                              {spec.hidden_size, linear_value_dim},
                              "linear_attn.out_proj",
                              error_message) ||
                !RequireShape(layer.linear_attn_conv1d_weight,
                              {linear_qkv_dim, 1, spec.linear_conv_kernel_dim},
                              "linear_attn.conv1d.weight",
                              error_message) ||
                !RequireShape(layer.linear_attn_a_log,
                              {spec.linear_num_value_heads},
                              "linear_attn.A_log",
                              error_message) ||
                !RequireShape(layer.linear_attn_dt_bias,
                              {spec.linear_num_value_heads},
                              "linear_attn.dt_bias",
                              error_message)) {
                return false;
            }

            if (!RequireRank(layer.linear_attn_conv1d_weight, 3, "linear_attn.conv1d.weight", error_message)) {
                return false;
            }
        }

        parsed.layers.push_back(std::move(layer));
    }

    *metadata = std::move(parsed);
    return true;
}

}  // namespace soc::gpu::models::qwen3_5
