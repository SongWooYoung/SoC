#include "models/qwen3_5/qwen3_5_loader_adapter.h"

#include "asset/runtime_assets.h"
#include "buffer/metal_buffer.h"
#include "header/dtype.h"
#include "header/tensor.h"
#include "header/storage.h"

#include <cstdint>
#include <cstring>

namespace soc::gpu::models::qwen3_5 {
namespace {

const JsonValue* FindObjectField(const JsonValue& object, const char* key) {
    return object.is_object() ? object.find(key) : nullptr;
}

std::size_t ReadSizeField(const JsonValue& object,
                          const char* key,
                          const std::size_t fallback) {
    const JsonValue* value = FindObjectField(object, key);
    if (value == nullptr || value->is_null()) {
        return fallback;
    }
    return static_cast<std::size_t>(value->as_int64());
}

bool ReadBoolField(const JsonValue& object, const char* key, const bool fallback) {
    const JsonValue* value = FindObjectField(object, key);
    if (value == nullptr || value->is_null()) {
        return fallback;
    }
    return value->as_bool();
}

double ReadNumberField(const JsonValue& object, const char* key, const double fallback) {
    const JsonValue* value = FindObjectField(object, key);
    if (value == nullptr || value->is_null()) {
        return fallback;
    }
    return value->as_number();
}

std::string ReadStringField(const JsonValue& object,
                            const char* key,
                            const std::string& fallback) {
    const JsonValue* value = FindObjectField(object, key);
    if (value == nullptr || value->is_null()) {
        return fallback;
    }
    return value->as_string();
}

bool ParseLayerTypes(const JsonValue& array_value,
                     std::vector<Qwen3_5LayerType>* layer_types,
                     std::string* error_message) {
    if (layer_types == nullptr) {
        return false;
    }
    if (!array_value.is_array()) {
        if (error_message != nullptr) {
            *error_message = "Qwen3.5 layer_types must be an array";
        }
        return false;
    }

    layer_types->clear();
    for (const JsonValue& value : array_value.as_array()) {
        const std::string layer_type = value.as_string();
        if (layer_type == "linear_attention") {
            layer_types->push_back(Qwen3_5LayerType::kGatedDeltaNet);
        } else if (layer_type == "full_attention") {
            layer_types->push_back(Qwen3_5LayerType::kGatedAttention);
        } else {
            if (error_message != nullptr) {
                *error_message = "unsupported Qwen3.5 layer type: " + layer_type;
            }
            return false;
        }
    }
    return true;
}

const JsonValue* ResolveTextConfig(const ManifestData& manifest) {
    const JsonValue& config = manifest.config;
    if (!config.is_object()) {
        return nullptr;
    }
    const JsonValue* text_config = config.find("text_config");
    if (text_config != nullptr && text_config->is_object()) {
        return text_config;
    }
    return &config;
}

bool LooksLikeQwen3_5Config(const JsonValue& root, const JsonValue& text_config) {
    return ReadStringField(root, "model_type", "") == "qwen3_5" ||
           ReadStringField(text_config, "model_type", "") == "qwen3_5_text";
}

bool LoadDeviceWeight(const MetalContext& context,
                      const TensorRecord& record,
                      DeviceTensor* tensor,
                      std::string* error_message) {
    if (tensor == nullptr) {
        if (error_message != nullptr) {
            *error_message = "LoadDeviceWeight requires a non-null tensor output";
        }
        return false;
    }

    std::vector<char> bytes;
    if (!TensorFileLoader::LoadBytes(record, &bytes, error_message)) {
        return false;
    }
    auto buffer = MetalBuffer::CreateInitializedForTensorClass(context,
                                                               bytes.data(),
                                                               bytes.size(),
                                                               record.name,
                                                               TensorClass::kStaticWeight,
                                                               error_message);
    if (buffer == nullptr) {
        return false;
    }
    *tensor = DeviceTensor(buffer,
                           0,
                           TensorDesc::CreateContiguous(ParseDataTypeString(record.dtype), record.shape));
    return true;
}

bool LoadDeviceWeightAsFloat32(const MetalContext& context,
                               const TensorRecord& record,
                               DeviceTensor* tensor,
                               std::string* error_message) {
    if (tensor == nullptr) {
        if (error_message != nullptr) {
            *error_message = "LoadDeviceWeightAsFloat32 requires a non-null tensor output";
        }
        return false;
    }

    std::vector<char> bytes;
    if (!TensorFileLoader::LoadBytes(record, &bytes, error_message)) {
        return false;
    }

    const DType source_dtype = DTypeFromString(record.dtype);
    std::size_t element_count = 1;
    for (std::size_t dim : record.shape) {
        element_count *= dim;
    }
    std::vector<float> values(element_count, 0.0f);
    const void* raw = bytes.data();
    for (std::size_t index = 0; index < element_count; ++index) {
        values[index] = 1.0f + DTypeReadFloat(raw, source_dtype, index);
    }

    auto buffer = MetalBuffer::CreateInitializedForTensorClass(context,
                                                               values.data(),
                                                               values.size() * sizeof(float),
                                                               record.name + "_f32",
                                                               TensorClass::kStaticWeight,
                                                               error_message);
    if (buffer == nullptr) {
        return false;
    }
    *tensor = DeviceTensor(buffer, 0, TensorDesc::CreateContiguous(DataType::kFloat32, record.shape));
    return true;
}

bool LoadDeviceWeight(const MetalContext& context,
                      const ManifestData& manifest,
                      const TensorRecord& record,
                      DeviceTensor* tensor,
                      std::string* error_message) {
    (void)manifest;
    return LoadDeviceWeight(context, record, tensor, error_message);
}

bool LoadHostWeight(const TensorRecord& record, Tensor* tensor, std::string* error_message) {
    if (tensor == nullptr) {
        if (error_message != nullptr) {
            *error_message = "LoadHostWeight requires a non-null tensor output";
        }
        return false;
    }
    try {
        const DType dtype = DTypeFromString(record.dtype);
        Storage storage = Storage::MapReadOnly(record.file);
        *tensor = Tensor(std::move(storage),
                         dtype,
                         record.shape,
                         Tensor::ComputeContiguousStrides(record.shape),
                         record.file_offset);
        return true;
    } catch (const std::exception& error) {
        if (error_message != nullptr) {
            *error_message = error.what();
        }
        return false;
    }
}

}  // namespace

bool ResolveArchitectureSpec(const ManifestData& manifest,
                             Qwen3_5ArchitectureSpec* spec,
                             std::string* error_message) {
    if (spec == nullptr) {
        if (error_message != nullptr) {
            *error_message = "ResolveArchitectureSpec requires a non-null spec output";
        }
        return false;
    }

    const JsonValue* text_config = ResolveTextConfig(manifest);
    if (text_config == nullptr || !LooksLikeQwen3_5Config(manifest.config, *text_config)) {
        if (error_message != nullptr) {
            *error_message = "manifest does not contain a recognizable Qwen3.5 text_config";
        }
        return false;
    }

    Qwen3_5ArchitectureSpec parsed = BuildQwen3_5_9BReferenceSpec();
    parsed.top_level_architecture = manifest.config.contains("architectures") &&
                                            manifest.config.at("architectures").is_array() &&
                                            !manifest.config.at("architectures").as_array().empty()
        ? manifest.config.at("architectures").as_array().front().as_string()
        : parsed.top_level_architecture;
    parsed.text_model_type = ReadStringField(*text_config, "model_type", parsed.text_model_type);
    parsed.vocab_size = ReadSizeField(*text_config, "vocab_size", parsed.vocab_size);
    parsed.hidden_size = ReadSizeField(*text_config, "hidden_size", parsed.hidden_size);
    parsed.intermediate_size = ReadSizeField(*text_config, "intermediate_size", parsed.intermediate_size);
    parsed.num_hidden_layers = ReadSizeField(*text_config, "num_hidden_layers", parsed.num_hidden_layers);
    parsed.num_attention_heads = ReadSizeField(*text_config, "num_attention_heads", parsed.num_attention_heads);
    parsed.num_key_value_heads = ReadSizeField(*text_config, "num_key_value_heads", parsed.num_key_value_heads);
    parsed.attention_head_dim = ReadSizeField(*text_config, "head_dim", parsed.attention_head_dim);
    parsed.linear_num_key_heads = ReadSizeField(*text_config, "linear_num_key_heads", parsed.linear_num_key_heads);
    parsed.linear_num_value_heads = ReadSizeField(*text_config, "linear_num_value_heads", parsed.linear_num_value_heads);
    parsed.linear_key_head_dim = ReadSizeField(*text_config, "linear_key_head_dim", parsed.linear_key_head_dim);
    parsed.linear_value_head_dim = ReadSizeField(*text_config, "linear_value_head_dim", parsed.linear_value_head_dim);
    parsed.linear_conv_kernel_dim = ReadSizeField(*text_config, "linear_conv_kernel_dim", parsed.linear_conv_kernel_dim);
    parsed.max_position_embeddings = ReadSizeField(*text_config, "max_position_embeddings", parsed.max_position_embeddings);
    parsed.full_attention_interval = ReadSizeField(*text_config, "full_attention_interval", parsed.full_attention_interval);
    parsed.mtp_num_hidden_layers = ReadSizeField(*text_config, "mtp_num_hidden_layers", parsed.mtp_num_hidden_layers);
    parsed.rms_norm_eps = static_cast<float>(ReadNumberField(*text_config, "rms_norm_eps", parsed.rms_norm_eps));
    parsed.attn_output_gate = ReadBoolField(*text_config, "attn_output_gate", parsed.attn_output_gate);
    parsed.tie_word_embeddings = ReadBoolField(manifest.config, "tie_word_embeddings", parsed.tie_word_embeddings);
    parsed.weights_dtype = ReadStringField(*text_config, "dtype", parsed.weights_dtype);
    parsed.recurrent_state_dtype = ReadStringField(*text_config, "mamba_ssm_dtype", parsed.recurrent_state_dtype);

    if (text_config->contains("layer_types")) {
        if (!ParseLayerTypes(text_config->at("layer_types"), &parsed.layer_types, error_message)) {
            return false;
        }
    }

    const JsonValue* rope_parameters = text_config->find("rope_parameters");
    if (rope_parameters != nullptr && rope_parameters->is_object()) {
        parsed.rope.mrope_interleaved =
            ReadBoolField(*rope_parameters, "mrope_interleaved", parsed.rope.mrope_interleaved);
        parsed.rope.rope_type = ReadStringField(*rope_parameters, "rope_type", parsed.rope.rope_type);
        parsed.rope.rope_theta = ReadNumberField(*rope_parameters, "rope_theta", parsed.rope.rope_theta);
        parsed.rope.partial_rotary_factor =
            ReadNumberField(*rope_parameters, "partial_rotary_factor", parsed.rope.partial_rotary_factor);
        const JsonValue* mrope_section = rope_parameters->find("mrope_section");
        if (mrope_section != nullptr && mrope_section->is_array()) {
            parsed.rope.mrope_section.clear();
            for (const JsonValue& value : mrope_section->as_array()) {
                parsed.rope.mrope_section.push_back(static_cast<std::size_t>(value.as_int64()));
            }
        }
        parsed.rotary_dim = static_cast<std::size_t>(
            parsed.attention_head_dim * parsed.rope.partial_rotary_factor);
    }

    if (parsed.layer_types.size() != parsed.num_hidden_layers) {
        if (error_message != nullptr) {
            *error_message = "Qwen3.5 layer_types count does not match num_hidden_layers";
        }
        return false;
    }

    *spec = std::move(parsed);
    return true;
}

bool ResolveArchitectureSpecFromFile(const std::string& manifest_path,
                                     Qwen3_5ArchitectureSpec* spec,
                                     std::string* error_message) {
    const ManifestData manifest = ManifestLoader::LoadFromFile(manifest_path);
    return ResolveArchitectureSpec(manifest, spec, error_message);
}

bool ResolveManifestMetadataFromFile(const std::string& manifest_path,
                                     const Qwen3_5ArchitectureSpec& spec,
                                     Qwen3_5ManifestMetadata* metadata,
                                     std::string* error_message) {
    const ManifestData manifest = ManifestLoader::LoadFromFile(manifest_path);
    return ResolveManifestMetadata(manifest, spec, metadata, error_message);
}

bool LoadGpuModel(const MetalContext& context,
                  const ManifestData& manifest,
                  Qwen3_5Runner* runner,
                  std::string* error_message) {
    if (runner == nullptr) {
        if (error_message != nullptr) {
            *error_message = "LoadGpuModel requires a non-null runner output";
        }
        return false;
    }

    Qwen3_5ArchitectureSpec spec;
    if (!ResolveArchitectureSpec(manifest, &spec, error_message)) {
        return false;
    }
    Qwen3_5ManifestMetadata metadata;
    if (!ResolveManifestMetadata(manifest, spec, &metadata, error_message)) {
        return false;
    }

    Qwen3_5Weights weights;
    Qwen3_5HostWeights host_weights;
    weights.tie_word_embeddings = spec.tie_word_embeddings;
    host_weights.tie_word_embeddings = spec.tie_word_embeddings;
    if (!LoadDeviceWeight(context, manifest, metadata.embed_tokens, &weights.embed_tokens_weight, error_message) ||
        !LoadHostWeight(metadata.embed_tokens, &host_weights.embed_tokens_weight, error_message) ||
        !LoadDeviceWeightAsFloat32(context, metadata.final_norm, &weights.final_norm_weight, error_message) ||
        !LoadHostWeight(metadata.final_norm, &host_weights.final_norm_weight, error_message)) {
        return false;
    }
    if (spec.tie_word_embeddings) {
        weights.lm_head_weight = weights.embed_tokens_weight;
        host_weights.lm_head_weight = host_weights.embed_tokens_weight;
    } else if (!LoadDeviceWeight(context, manifest, metadata.lm_head, &weights.lm_head_weight, error_message) ||
               !LoadHostWeight(metadata.lm_head, &host_weights.lm_head_weight, error_message)) {
        return false;
    }

    weights.blocks.resize(metadata.layers.size());
    host_weights.blocks.resize(metadata.layers.size());
    for (std::size_t layer_index = 0; layer_index < metadata.layers.size(); ++layer_index) {
        const Qwen3_5LayerManifestMetadata& layer_metadata = metadata.layers[layer_index];
        Qwen3_5BlockWeights& block = weights.blocks[layer_index];
        Qwen3_5HostBlockWeights& host_block = host_weights.blocks[layer_index];
        if (!LoadDeviceWeightAsFloat32(context, layer_metadata.input_layernorm, &block.input_layernorm_weight, error_message) ||
            !LoadHostWeight(layer_metadata.input_layernorm, &host_block.input_layernorm_weight, error_message) ||
            !LoadDeviceWeightAsFloat32(context, layer_metadata.post_attention_layernorm, &block.post_attention_layernorm_weight, error_message) ||
            !LoadHostWeight(layer_metadata.post_attention_layernorm, &host_block.post_attention_layernorm_weight, error_message) ||
            !LoadDeviceWeight(context, manifest, layer_metadata.mlp_gate_proj, &block.mlp.gate_proj_weight, error_message) ||
            !LoadHostWeight(layer_metadata.mlp_gate_proj, &host_block.mlp.gate_proj_weight, error_message) ||
            !LoadDeviceWeight(context, manifest, layer_metadata.mlp_up_proj, &block.mlp.up_proj_weight, error_message) ||
            !LoadHostWeight(layer_metadata.mlp_up_proj, &host_block.mlp.up_proj_weight, error_message) ||
            !LoadDeviceWeight(context, manifest, layer_metadata.mlp_down_proj, &block.mlp.down_proj_weight, error_message)) {
            return false;
        }
        if (!LoadHostWeight(layer_metadata.mlp_down_proj, &host_block.mlp.down_proj_weight, error_message)) {
            return false;
        }

        if (layer_metadata.layer_type == Qwen3_5LayerType::kGatedAttention) {
            if (!LoadDeviceWeight(context, manifest, layer_metadata.self_attn_q_proj, &block.attention.q_proj_weight, error_message) ||
                !LoadHostWeight(layer_metadata.self_attn_q_proj, &host_block.attention.q_proj_weight, error_message) ||
                !LoadDeviceWeight(context, manifest, layer_metadata.self_attn_k_proj, &block.attention.k_proj_weight, error_message) ||
                !LoadHostWeight(layer_metadata.self_attn_k_proj, &host_block.attention.k_proj_weight, error_message) ||
                !LoadDeviceWeight(context, manifest, layer_metadata.self_attn_v_proj, &block.attention.v_proj_weight, error_message) ||
                !LoadHostWeight(layer_metadata.self_attn_v_proj, &host_block.attention.v_proj_weight, error_message) ||
                !LoadDeviceWeight(context, manifest, layer_metadata.self_attn_o_proj, &block.attention.o_proj_weight, error_message) ||
                !LoadHostWeight(layer_metadata.self_attn_o_proj, &host_block.attention.o_proj_weight, error_message) ||
                !LoadDeviceWeightAsFloat32(context, layer_metadata.self_attn_q_norm, &block.attention.q_norm_weight, error_message) ||
                !LoadHostWeight(layer_metadata.self_attn_q_norm, &host_block.attention.q_norm_weight, error_message) ||
                !LoadDeviceWeightAsFloat32(context, layer_metadata.self_attn_k_norm, &block.attention.k_norm_weight, error_message)) {
                return false;
            }
            if (!LoadHostWeight(layer_metadata.self_attn_k_norm, &host_block.attention.k_norm_weight, error_message)) {
                return false;
            }
        } else {
            if (!LoadDeviceWeightAsFloat32(context, layer_metadata.linear_attn_norm, &block.linear.norm_weight, error_message) ||
                !LoadHostWeight(layer_metadata.linear_attn_norm, &host_block.linear.norm_weight, error_message) ||
                !LoadDeviceWeight(context, manifest, layer_metadata.linear_attn_in_proj_qkv, &block.linear.in_proj_qkv_weight, error_message) ||
                !LoadHostWeight(layer_metadata.linear_attn_in_proj_qkv, &host_block.linear.in_proj_qkv_weight, error_message) ||
                !LoadDeviceWeight(context, manifest, layer_metadata.linear_attn_in_proj_z, &block.linear.in_proj_z_weight, error_message) ||
                !LoadHostWeight(layer_metadata.linear_attn_in_proj_z, &host_block.linear.in_proj_z_weight, error_message) ||
                !LoadDeviceWeight(context, manifest, layer_metadata.linear_attn_in_proj_a, &block.linear.in_proj_a_weight, error_message) ||
                !LoadHostWeight(layer_metadata.linear_attn_in_proj_a, &host_block.linear.in_proj_a_weight, error_message) ||
                !LoadDeviceWeight(context, manifest, layer_metadata.linear_attn_in_proj_b, &block.linear.in_proj_b_weight, error_message) ||
                !LoadHostWeight(layer_metadata.linear_attn_in_proj_b, &host_block.linear.in_proj_b_weight, error_message) ||
                !LoadDeviceWeight(context, manifest, layer_metadata.linear_attn_out_proj, &block.linear.out_proj_weight, error_message) ||
                !LoadHostWeight(layer_metadata.linear_attn_out_proj, &host_block.linear.out_proj_weight, error_message) ||
                !LoadDeviceWeight(context, manifest, layer_metadata.linear_attn_conv1d_weight, &block.linear.conv1d_weight, error_message) ||
                !LoadHostWeight(layer_metadata.linear_attn_conv1d_weight, &host_block.linear.conv1d_weight, error_message) ||
                !LoadDeviceWeight(context, manifest, layer_metadata.linear_attn_a_log, &block.linear.a_log, error_message) ||
                !LoadHostWeight(layer_metadata.linear_attn_a_log, &host_block.linear.a_log, error_message) ||
                !LoadDeviceWeight(context, manifest, layer_metadata.linear_attn_dt_bias, &block.linear.dt_bias, error_message)) {
                return false;
            }
            if (!LoadHostWeight(layer_metadata.linear_attn_dt_bias, &host_block.linear.dt_bias, error_message)) {
                return false;
            }
        }
    }

    *runner = Qwen3_5Runner(spec, std::move(weights), std::move(host_weights), BuildStateLayout(spec));
    return true;
}

bool LoadGpuModelFromFile(const MetalContext& context,
                          const std::string& manifest_path,
                          Qwen3_5Runner* runner,
                          std::string* error_message) {
    const ManifestData manifest = ManifestLoader::LoadFromFile(manifest_path);
    return LoadGpuModel(context, manifest, runner, error_message);
}

bool PrepareGpuModelNotImplemented(const MetalContext&,
                                   const ManifestData&,
                                   std::string* error_message) {
    if (error_message != nullptr) {
        *error_message = "Qwen3.5 GPU forward is not implemented yet";
    }
    return false;
}

}  // namespace soc::gpu::models::qwen3_5
