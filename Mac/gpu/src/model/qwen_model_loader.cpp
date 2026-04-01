#include "model/qwen_model_loader.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <sstream>

#include "buffer/metal_buffer.h"

namespace soc::gpu {
namespace {

float HalfToFloat(std::uint16_t value) {
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000u) << 16;
    std::uint32_t exponent = static_cast<std::uint32_t>((value >> 10) & 0x1fu);
    std::uint32_t mantissa = static_cast<std::uint32_t>(value & 0x03ffu);
    std::uint32_t bits = 0;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 127 - 15 + 1;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x03ffu;
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }

    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

bool ReadOptionalBool(const JsonValue& object, const char* key, bool fallback) {
    if (!object.is_object() || !object.contains(key) || object.at(key).is_null()) {
        return fallback;
    }
    return object.at(key).as_bool();
}

std::size_t ReadRequiredSize(const JsonValue& object, const char* key) {
    return static_cast<std::size_t>(object.at(key).as_int64());
}

double ReadRequiredNumber(const JsonValue& object, const char* key) {
    return object.at(key).as_number();
}

bool ShouldPreserveFloat16Storage(const std::string& name,
                                  const std::vector<std::size_t>& expected_shape,
                                  const std::string& dtype) {
    const char* env_value = std::getenv("SOC_GPU_ENABLE_EXPERIMENTAL_F16_WEIGHTS");
    if (env_value == nullptr || std::string(env_value) != "1") {
        return false;
    }
    const auto ends_with = [](const std::string& value, const char* suffix) {
        const std::size_t suffix_size = std::char_traits<char>::length(suffix);
        return value.size() >= suffix_size &&
               value.compare(value.size() - suffix_size, suffix_size, suffix) == 0;
    };
    if (dtype != "float16" || expected_shape.size() != 2) {
        return false;
    }
    return name == "model.embed_tokens.weight" ||
           name == "lm_head.weight" ||
           ends_with(name, ".self_attn.q_proj.weight") ||
           ends_with(name, ".self_attn.k_proj.weight") ||
           ends_with(name, ".self_attn.v_proj.weight") ||
           ends_with(name, ".self_attn.o_proj.weight") ||
           ends_with(name, ".mlp.gate_proj.weight") ||
           ends_with(name, ".mlp.up_proj.weight") ||
           ends_with(name, ".mlp.down_proj.weight");
}

bool LoadFloatingWeight(const MetalContext& context,
                        const ManifestData& manifest,
                        const std::string& name,
                        const std::vector<std::size_t>& expected_shape,
                        DeviceTensor* tensor,
                        std::string* error_message) {
    const TensorRecord& record = manifest.FindTensor(name);
    if (record.shape != expected_shape) {
        if (error_message != nullptr) {
            *error_message = "unexpected tensor shape for " + name;
        }
        return false;
    }

    std::vector<char> bytes;
    if (!TensorFileLoader::LoadBytes(record, &bytes, error_message)) {
        return false;
    }

    const bool preserve_float16_storage = ShouldPreserveFloat16Storage(name, expected_shape, record.dtype);
    if (preserve_float16_storage) {
        auto buffer = MetalBuffer::CreateInitializedForTensorClass(context,
                                                                   bytes.data(),
                                                                   bytes.size(),
                                                                   name,
                                                                   TensorClass::kStaticWeight,
                                                                   error_message);
        if (buffer == nullptr) {
            return false;
        }
        *tensor = DeviceTensor(buffer, 0, TensorDesc::CreateContiguous(DataType::kFloat16, expected_shape));
        return true;
    }

    std::vector<float> converted(record.byte_size / (record.dtype == "float16" ? sizeof(std::uint16_t) : sizeof(float)), 0.0f);
    if (record.dtype == "float16") {
        const auto* half_data = reinterpret_cast<const std::uint16_t*>(bytes.data());
        for (std::size_t index = 0; index < converted.size(); ++index) {
            converted[index] = HalfToFloat(half_data[index]);
        }
    } else if (record.dtype == "float32") {
        std::memcpy(converted.data(), bytes.data(), bytes.size());
    } else {
        if (error_message != nullptr) {
            *error_message = "unsupported floating tensor dtype for " + name + ": " + record.dtype;
        }
        return false;
    }

    auto buffer = MetalBuffer::CreateInitializedForTensorClass(context,
                                                               converted.data(),
                                                               sizeof(float) * converted.size(),
                                                               name,
                                                               TensorClass::kStaticWeight,
                                                               error_message);
    if (buffer == nullptr) {
        return false;
    }
    *tensor = DeviceTensor(buffer, 0, TensorDesc::CreateContiguous(DataType::kFloat32, expected_shape));
    return true;
}

bool LoadRawWeight(const MetalContext& context,
                   const ManifestData& manifest,
                   const std::string& name,
                   const std::vector<std::size_t>& expected_shape,
                   DataType expected_type,
                   DeviceTensor* tensor,
                   std::string* error_message) {
    const TensorRecord* record = manifest.FindTensorIfPresent(name);
    if (record == nullptr) {
        return false;
    }
    if (record->shape != expected_shape || ParseDataTypeString(record->dtype) != expected_type) {
        if (error_message != nullptr) {
            *error_message = "unexpected tensor shape or dtype for " + name;
        }
        return false;
    }

    std::vector<char> bytes;
    if (!TensorFileLoader::LoadBytes(*record, &bytes, error_message)) {
        return false;
    }
    auto buffer = MetalBuffer::CreateInitializedForTensorClass(context,
                                                               bytes.data(),
                                                               bytes.size(),
                                                               name,
                                                               TensorClass::kStaticWeight,
                                                               error_message);
    if (buffer == nullptr) {
        return false;
    }
    *tensor = DeviceTensor(buffer, 0, TensorDesc::CreateContiguous(expected_type, expected_shape));
    return true;
}

std::string LayerTensorName(std::size_t layer_index, const char* suffix) {
    std::ostringstream name;
    name << "model.layers." << layer_index << "." << suffix;
    return name.str();
}

std::string QuantizedTensorName(const std::string& weight_name, const char* suffix) {
    constexpr char kWeightSuffix[] = ".weight";
    const std::size_t suffix_size = sizeof(kWeightSuffix) - 1;
    if (weight_name.size() < suffix_size ||
        weight_name.compare(weight_name.size() - suffix_size, suffix_size, kWeightSuffix) != 0) {
        return weight_name + "." + suffix;
    }
    return weight_name.substr(0, weight_name.size() - suffix_size) + "." + suffix;
}

bool LoadOptionalQuantizedWeight(const MetalContext& context,
                                 const ManifestData& manifest,
                                 const std::string& weight_name,
                                 const std::vector<std::size_t>& weight_shape,
                                 AffineQmmWeight* weight,
                                 std::string* error_message) {
    if (weight == nullptr || weight_shape.size() != 2 || weight_shape[1] % 128 != 0) {
        return false;
    }
    AffineQmmWeight loaded;
    if (LoadRawWeight(context,
                      manifest,
                      QuantizedTensorName(weight_name, "qweight"),
                      {weight_shape[0], weight_shape[1] / 8},
                      DataType::kUInt32,
                      &loaded.qweight,
                      error_message) &&
        LoadFloatingWeight(context,
                           manifest,
                           QuantizedTensorName(weight_name, "scales"),
                           {weight_shape[0], weight_shape[1] / 128},
                           &loaded.scales,
                           error_message) &&
        LoadFloatingWeight(context,
                           manifest,
                           QuantizedTensorName(weight_name, "qbiases"),
                           {weight_shape[0], weight_shape[1] / 128},
                           &loaded.qbiases,
                           error_message)) {
        loaded.bits = 4;
        loaded.group_size = 128;
        *weight = std::move(loaded);
        return true;
    }
    return false;
}

}  // namespace

bool QwenModelLoader::LoadModel(const MetalContext& context,
                                const ManifestData& manifest,
                                QwenCausalLM* model,
                                std::string* error_message) {
    if (model == nullptr) {
        if (error_message != nullptr) {
            *error_message = "QwenModelLoader requires a non-null model output";
        }
        return false;
    }
    if (!manifest.config.is_object()) {
        if (error_message != nullptr) {
            *error_message = "manifest config must be an object";
        }
        return false;
    }

    QwenCausalLMParams params;
    params.hidden_size = ReadRequiredSize(manifest.config, "hidden_size");
    params.intermediate_size = ReadRequiredSize(manifest.config, "intermediate_size");
    params.num_attention_heads = ReadRequiredSize(manifest.config, "num_attention_heads");
    params.num_key_value_heads = ReadRequiredSize(manifest.config, "num_key_value_heads");
    params.num_hidden_layers = ReadRequiredSize(manifest.config, "num_hidden_layers");
    params.vocab_size = ReadRequiredSize(manifest.config, "vocab_size");
    params.max_position_embeddings = ReadRequiredSize(manifest.config, "max_position_embeddings");
    params.rope_theta = static_cast<float>(ReadRequiredNumber(manifest.config, "rope_theta"));
    params.rms_norm_eps = static_cast<float>(ReadRequiredNumber(manifest.config, "rms_norm_eps"));
    params.head_dim = manifest.config.contains("head_dim")
        ? static_cast<std::size_t>(manifest.config.at("head_dim").as_int64())
        : params.hidden_size / params.num_attention_heads;

    QwenCausalLMWeights weights;
    weights.tie_word_embeddings = ReadOptionalBool(manifest.config, "tie_word_embeddings", true);
    if (!LoadFloatingWeight(context,
                            manifest,
                            "model.embed_tokens.weight",
                            {params.vocab_size, params.hidden_size},
                            &weights.embed_tokens_weight,
                            error_message)) {
        return false;
    }
    if (!LoadFloatingWeight(context,
                            manifest,
                            "model.norm.weight",
                            {params.hidden_size},
                            &weights.final_norm_weight,
                            error_message)) {
        return false;
    }
    if (weights.tie_word_embeddings) {
        weights.lm_head_weight = weights.embed_tokens_weight;
    } else if (!LoadFloatingWeight(context,
                                   manifest,
                                   "lm_head.weight",
                                   {params.vocab_size, params.hidden_size},
                                   &weights.lm_head_weight,
                                   error_message)) {
        return false;
    }
    if (LoadRawWeight(context,
                      manifest,
                      "lm_head.qweight",
                      {params.vocab_size, params.hidden_size / 8},
                      DataType::kUInt32,
                      &weights.lm_head_q4_weight.qweight,
                      error_message) &&
        LoadFloatingWeight(context,
                           manifest,
                           "lm_head.scales",
                           {params.vocab_size, params.hidden_size / 128},
                           &weights.lm_head_q4_weight.scales,
                           error_message) &&
        LoadFloatingWeight(context,
                           manifest,
                           "lm_head.qbiases",
                           {params.vocab_size, params.hidden_size / 128},
                           &weights.lm_head_q4_weight.qbiases,
                           error_message)) {
        weights.lm_head_q4_weight.bits = 4;
        weights.lm_head_q4_weight.group_size = 128;
        weights.has_quantized_lm_head = true;
    }

    const std::size_t q_projection_size = params.num_attention_heads * params.head_dim;
    const std::size_t kv_projection_size = params.num_key_value_heads * params.head_dim;
    weights.blocks.resize(params.num_hidden_layers);
    for (std::size_t layer_index = 0; layer_index < params.num_hidden_layers; ++layer_index) {
        QwenBlockWeights& block = weights.blocks[layer_index];
        const std::string q_proj_name = LayerTensorName(layer_index, "self_attn.q_proj.weight");
        const std::string k_proj_name = LayerTensorName(layer_index, "self_attn.k_proj.weight");
        const std::string v_proj_name = LayerTensorName(layer_index, "self_attn.v_proj.weight");
        const std::string o_proj_name = LayerTensorName(layer_index, "self_attn.o_proj.weight");
        const std::string gate_proj_name = LayerTensorName(layer_index, "mlp.gate_proj.weight");
        const std::string up_proj_name = LayerTensorName(layer_index, "mlp.up_proj.weight");
        const std::string down_proj_name = LayerTensorName(layer_index, "mlp.down_proj.weight");
        if (!LoadFloatingWeight(context, manifest, LayerTensorName(layer_index, "input_layernorm.weight"), {params.hidden_size}, &block.input_layernorm_weight, error_message) ||
            !LoadFloatingWeight(context, manifest, q_proj_name, {q_projection_size, params.hidden_size}, &block.attention.q_proj_weight, error_message) ||
            !LoadFloatingWeight(context, manifest, k_proj_name, {kv_projection_size, params.hidden_size}, &block.attention.k_proj_weight, error_message) ||
            !LoadFloatingWeight(context, manifest, v_proj_name, {kv_projection_size, params.hidden_size}, &block.attention.v_proj_weight, error_message) ||
            !LoadFloatingWeight(context, manifest, o_proj_name, {params.hidden_size, q_projection_size}, &block.attention.o_proj_weight, error_message) ||
            !LoadFloatingWeight(context, manifest, LayerTensorName(layer_index, "self_attn.q_norm.weight"), {params.head_dim}, &block.attention.q_norm_weight, error_message) ||
            !LoadFloatingWeight(context, manifest, LayerTensorName(layer_index, "self_attn.k_norm.weight"), {params.head_dim}, &block.attention.k_norm_weight, error_message) ||
            !LoadFloatingWeight(context, manifest, LayerTensorName(layer_index, "post_attention_layernorm.weight"), {params.hidden_size}, &block.post_attention_layernorm_weight, error_message) ||
            !LoadFloatingWeight(context, manifest, gate_proj_name, {params.intermediate_size, params.hidden_size}, &block.mlp.gate_proj_weight, error_message) ||
            !LoadFloatingWeight(context, manifest, up_proj_name, {params.intermediate_size, params.hidden_size}, &block.mlp.up_proj_weight, error_message) ||
            !LoadFloatingWeight(context, manifest, down_proj_name, {params.hidden_size, params.intermediate_size}, &block.mlp.down_proj_weight, error_message)) {
            return false;
        }
        LoadOptionalQuantizedWeight(context, manifest, q_proj_name, {q_projection_size, params.hidden_size}, &block.attention.q_proj_q4_weight, error_message);
        LoadOptionalQuantizedWeight(context, manifest, k_proj_name, {kv_projection_size, params.hidden_size}, &block.attention.k_proj_q4_weight, error_message);
        LoadOptionalQuantizedWeight(context, manifest, v_proj_name, {kv_projection_size, params.hidden_size}, &block.attention.v_proj_q4_weight, error_message);
        LoadOptionalQuantizedWeight(context, manifest, o_proj_name, {params.hidden_size, q_projection_size}, &block.attention.o_proj_q4_weight, error_message);
        LoadOptionalQuantizedWeight(context, manifest, gate_proj_name, {params.intermediate_size, params.hidden_size}, &block.mlp.gate_proj_q4_weight, error_message);
        LoadOptionalQuantizedWeight(context, manifest, up_proj_name, {params.intermediate_size, params.hidden_size}, &block.mlp.up_proj_q4_weight, error_message);
        LoadOptionalQuantizedWeight(context, manifest, down_proj_name, {params.hidden_size, params.intermediate_size}, &block.mlp.down_proj_q4_weight, error_message);
    }

    *model = QwenCausalLM(std::move(weights), params);
    return true;
}

bool QwenModelLoader::LoadModelFromFile(const MetalContext& context,
                                        const std::string& manifest_path,
                                        QwenCausalLM* model,
                                        std::string* error_message) {
    return LoadModel(context, ManifestLoader::LoadFromFile(manifest_path), model, error_message);
}

}  // namespace soc::gpu
