#include "header/qwen_model_loader.h"

#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "header/qwen_attention.h"
#include "header/qwen_block.h"
#include "header/qwen_mlp.h"
#include "header/runtime_assets.h"
#include "header/startup_validator.h"
#include "header/weight_loader.h"

namespace {
Tensor RequireFloatingWeight(const ManifestData& manifest, const std::string& name, const std::vector<std::size_t>& expected_shape) {
    const Tensor tensor = WeightLoader::LoadByName(manifest, name);
    if (!DTypeIsFloatLike(tensor.dtype())) {
        throw std::runtime_error("model loader requires floating-point weights for: " + name);
    }
    if (tensor.shape() != expected_shape) {
        throw std::runtime_error("model loader encountered unexpected shape for: " + name);
    }
    return tensor;
}

std::string LayerTensorName(std::size_t layer_index, const char* suffix) {
    std::ostringstream name;
    name << "model.layers." << layer_index << "." << suffix;
    return name.str();
}
}

QwenCausalLM QwenModelLoader::LoadModel(const ManifestData& manifest) {
    const Qwen3Config config = ValidateAndReadConfig(manifest);

    const std::size_t hidden_size = config.hidden_size;
    const std::size_t intermediate_size = config.intermediate_size;
    const std::size_t q_projection_size = config.num_attention_heads * config.head_dim;
    const std::size_t kv_projection_size = config.num_key_value_heads * config.head_dim;

    const Embedding embedding(RequireFloatingWeight(
        manifest,
        "model.embed_tokens.weight",
        {config.vocab_size, hidden_size}));

    std::vector<QwenBlock> blocks;
    blocks.reserve(config.num_hidden_layers);

    for (std::size_t layer_index = 0; layer_index < config.num_hidden_layers; ++layer_index) {
        const RMSNormModule input_norm(RequireFloatingWeight(
            manifest,
            LayerTensorName(layer_index, "input_layernorm.weight"),
            {hidden_size}),
            static_cast<float>(config.rms_norm_eps));

        const QwenAttention attention(
            Linear(RequireFloatingWeight(manifest, LayerTensorName(layer_index, "self_attn.q_proj.weight"), {q_projection_size, hidden_size})),
            Linear(RequireFloatingWeight(manifest, LayerTensorName(layer_index, "self_attn.k_proj.weight"), {kv_projection_size, hidden_size})),
            Linear(RequireFloatingWeight(manifest, LayerTensorName(layer_index, "self_attn.v_proj.weight"), {kv_projection_size, hidden_size})),
            Linear(RequireFloatingWeight(manifest, LayerTensorName(layer_index, "self_attn.o_proj.weight"), {hidden_size, q_projection_size})),
            RoPE(config.head_dim, config.rope_theta),
            GroupedQueryAttention(config.num_attention_heads, config.num_key_value_heads, config.head_dim, true));

        const RMSNormModule post_attention_norm(RequireFloatingWeight(
            manifest,
            LayerTensorName(layer_index, "post_attention_layernorm.weight"),
            {hidden_size}),
            static_cast<float>(config.rms_norm_eps));

        const QwenMLP mlp(
            Linear(RequireFloatingWeight(manifest, LayerTensorName(layer_index, "mlp.gate_proj.weight"), {intermediate_size, hidden_size})),
            Linear(RequireFloatingWeight(manifest, LayerTensorName(layer_index, "mlp.up_proj.weight"), {intermediate_size, hidden_size})),
            Linear(RequireFloatingWeight(manifest, LayerTensorName(layer_index, "mlp.down_proj.weight"), {hidden_size, intermediate_size})));

        blocks.emplace_back(input_norm, attention, post_attention_norm, mlp);
    }

    const RMSNormModule final_norm(RequireFloatingWeight(manifest, "model.norm.weight", {hidden_size}), static_cast<float>(config.rms_norm_eps));

    if (config.tie_word_embeddings) {
        return QwenCausalLM(embedding, std::move(blocks), final_norm);
    }

    return QwenCausalLM(
        embedding,
        std::move(blocks),
        final_norm,
        Linear(RequireFloatingWeight(manifest, "lm_head.weight", {config.vocab_size, hidden_size})));
}

QwenCausalLM QwenModelLoader::LoadModelFromFile(const std::string& manifest_path) {
    return LoadModel(ManifestLoader::LoadFromFile(manifest_path));
}

Qwen3Config QwenModelLoader::ValidateAndReadConfig(const ManifestData& manifest) {
    return StartupValidator::ValidateManifest(manifest);
}