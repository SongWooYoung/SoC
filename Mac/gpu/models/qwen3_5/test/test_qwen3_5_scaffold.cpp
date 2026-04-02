#include <iostream>
#include <string>
#include <vector>

#include "asset/json_value.h"
#include "models/qwen3_5/qwen3_5_architecture.h"
#include "models/qwen3_5/qwen3_5_loader_adapter.h"
#include "models/qwen3_5/qwen3_5_manifest_metadata.h"
#include "models/qwen3_5/qwen3_5_runner.h"
#include "models/qwen3_5/qwen3_5_state_layout.h"

int main() {
    soc::gpu::models::qwen3_5::Qwen3_5ArchitectureSpec spec =
        soc::gpu::models::qwen3_5::BuildQwen3_5_9BReferenceSpec();
    if (spec.num_hidden_layers != 32 || spec.layer_types.size() != 32) {
        std::cerr << "qwen3.5 scaffold spec is invalid\n";
        return 1;
    }
    const soc::gpu::models::qwen3_5::Qwen3_5Runner runner(spec);
    if (runner.num_layers() != 32 || runner.hidden_size() != 4096 || runner.num_key_value_heads() != 4) {
        std::cerr << "qwen3.5 runner metadata surface is invalid\n";
        return 1;
    }

    const soc::gpu::models::qwen3_5::Qwen3_5StateLayout state_layout =
        soc::gpu::models::qwen3_5::BuildStateLayout(spec);
    if (state_layout.deltanet_layers.size() != 24 ||
        state_layout.full_attention_layer_count != 8 ||
        state_layout.recurrent_state_dtype != "float32" ||
        state_layout.recurrent_state_element_bytes != sizeof(float) ||
        state_layout.total_recurrent_state_bytes == 0) {
        std::cerr << "qwen3.5 state layout is invalid\n";
        return 1;
    }

    soc::gpu::ManifestData manifest;
    manifest.model_id = "Qwen/Qwen3.5-9B";
    manifest.config = soc::gpu::JsonValue(soc::gpu::JsonValue::Object{
        {"model_type", soc::gpu::JsonValue(std::string("qwen3_5"))},
        {"architectures", soc::gpu::JsonValue(soc::gpu::JsonValue::Array{
            soc::gpu::JsonValue(std::string("Qwen3_5ForConditionalGeneration"))
        })},
        {"tie_word_embeddings", soc::gpu::JsonValue(false)},
        {"text_config", soc::gpu::JsonValue(soc::gpu::JsonValue::Object{
            {"model_type", soc::gpu::JsonValue(std::string("qwen3_5_text"))},
            {"dtype", soc::gpu::JsonValue(std::string("bfloat16"))},
            {"mamba_ssm_dtype", soc::gpu::JsonValue(std::string("float32"))},
            {"hidden_size", soc::gpu::JsonValue(4096.0)},
            {"intermediate_size", soc::gpu::JsonValue(12288.0)},
            {"num_hidden_layers", soc::gpu::JsonValue(32.0)},
            {"num_attention_heads", soc::gpu::JsonValue(16.0)},
            {"num_key_value_heads", soc::gpu::JsonValue(4.0)},
            {"head_dim", soc::gpu::JsonValue(256.0)},
            {"linear_num_key_heads", soc::gpu::JsonValue(16.0)},
            {"linear_num_value_heads", soc::gpu::JsonValue(32.0)},
            {"linear_key_head_dim", soc::gpu::JsonValue(128.0)},
            {"linear_value_head_dim", soc::gpu::JsonValue(128.0)},
            {"linear_conv_kernel_dim", soc::gpu::JsonValue(4.0)},
            {"max_position_embeddings", soc::gpu::JsonValue(262144.0)},
            {"full_attention_interval", soc::gpu::JsonValue(4.0)},
            {"mtp_num_hidden_layers", soc::gpu::JsonValue(1.0)},
            {"attn_output_gate", soc::gpu::JsonValue(true)},
            {"rms_norm_eps", soc::gpu::JsonValue(1.0e-6)},
            {"vocab_size", soc::gpu::JsonValue(248320.0)},
            {"layer_types", soc::gpu::JsonValue(soc::gpu::JsonValue::Array{
                soc::gpu::JsonValue(std::string("linear_attention")),
                soc::gpu::JsonValue(std::string("linear_attention")),
                soc::gpu::JsonValue(std::string("linear_attention")),
                soc::gpu::JsonValue(std::string("full_attention")),
                soc::gpu::JsonValue(std::string("linear_attention")),
                soc::gpu::JsonValue(std::string("linear_attention")),
                soc::gpu::JsonValue(std::string("linear_attention")),
                soc::gpu::JsonValue(std::string("full_attention")),
                soc::gpu::JsonValue(std::string("linear_attention")),
                soc::gpu::JsonValue(std::string("linear_attention")),
                soc::gpu::JsonValue(std::string("linear_attention")),
                soc::gpu::JsonValue(std::string("full_attention")),
                soc::gpu::JsonValue(std::string("linear_attention")),
                soc::gpu::JsonValue(std::string("linear_attention")),
                soc::gpu::JsonValue(std::string("linear_attention")),
                soc::gpu::JsonValue(std::string("full_attention")),
                soc::gpu::JsonValue(std::string("linear_attention")),
                soc::gpu::JsonValue(std::string("linear_attention")),
                soc::gpu::JsonValue(std::string("linear_attention")),
                soc::gpu::JsonValue(std::string("full_attention")),
                soc::gpu::JsonValue(std::string("linear_attention")),
                soc::gpu::JsonValue(std::string("linear_attention")),
                soc::gpu::JsonValue(std::string("linear_attention")),
                soc::gpu::JsonValue(std::string("full_attention")),
                soc::gpu::JsonValue(std::string("linear_attention")),
                soc::gpu::JsonValue(std::string("linear_attention")),
                soc::gpu::JsonValue(std::string("linear_attention")),
                soc::gpu::JsonValue(std::string("full_attention")),
                soc::gpu::JsonValue(std::string("linear_attention")),
                soc::gpu::JsonValue(std::string("linear_attention")),
                soc::gpu::JsonValue(std::string("linear_attention")),
                soc::gpu::JsonValue(std::string("full_attention"))
            })},
            {"rope_parameters", soc::gpu::JsonValue(soc::gpu::JsonValue::Object{
                {"mrope_interleaved", soc::gpu::JsonValue(true)},
                {"rope_type", soc::gpu::JsonValue(std::string("default"))},
                {"rope_theta", soc::gpu::JsonValue(10000000.0)},
                {"partial_rotary_factor", soc::gpu::JsonValue(0.25)},
                {"mrope_section", soc::gpu::JsonValue(soc::gpu::JsonValue::Array{
                    soc::gpu::JsonValue(11.0),
                    soc::gpu::JsonValue(11.0),
                    soc::gpu::JsonValue(10.0)
                })}
            })}
        })}
    });

    auto add_tensor = [&](const std::string& name, const std::vector<std::size_t>& shape, const std::string& dtype = "float16") {
        std::size_t element_count = 1;
        for (const std::size_t dim : shape) {
            element_count *= dim;
        }
        const std::size_t element_bytes = dtype == "float32" ? sizeof(float) : 2;
        soc::gpu::TensorRecord record;
        record.name = name;
        record.file = "weights/" + name + ".bin";
        record.dtype = dtype;
        record.shape = shape;
        record.file_offset = 0;
        record.byte_size = element_count * element_bytes;
        record.source_shard = "model.safetensors";
        manifest.tensors.push_back(std::move(record));
    };

    add_tensor("model.language_model.embed_tokens.weight", {spec.vocab_size, spec.hidden_size});
    add_tensor("model.language_model.norm.weight", {spec.hidden_size});
    add_tensor("lm_head.weight", {spec.vocab_size, spec.hidden_size});

    const std::size_t attention_proj_dim = spec.num_attention_heads * spec.attention_head_dim;
    const std::size_t attention_q_proj_dim = attention_proj_dim * 2;
    const std::size_t kv_proj_dim = spec.num_key_value_heads * spec.attention_head_dim;
    const std::size_t linear_key_dim = spec.linear_num_key_heads * spec.linear_key_head_dim;
    const std::size_t linear_value_dim = spec.linear_num_value_heads * spec.linear_value_head_dim;
    const std::size_t linear_qkv_dim = linear_key_dim * 2 + linear_value_dim;

    for (std::size_t layer_index = 0; layer_index < spec.layer_types.size(); ++layer_index) {
        const std::string prefix = "model.language_model.layers." + std::to_string(layer_index) + ".";
        add_tensor(prefix + "input_layernorm.weight", {spec.hidden_size});
        add_tensor(prefix + "post_attention_layernorm.weight", {spec.hidden_size});
        add_tensor(prefix + "mlp.gate_proj.weight", {spec.intermediate_size, spec.hidden_size});
        add_tensor(prefix + "mlp.up_proj.weight", {spec.intermediate_size, spec.hidden_size});
        add_tensor(prefix + "mlp.down_proj.weight", {spec.hidden_size, spec.intermediate_size});

        if (spec.layer_types[layer_index] == soc::gpu::models::qwen3_5::Qwen3_5LayerType::kGatedAttention) {
            add_tensor(prefix + "self_attn.q_proj.weight", {attention_q_proj_dim, spec.hidden_size});
            add_tensor(prefix + "self_attn.k_proj.weight", {kv_proj_dim, spec.hidden_size});
            add_tensor(prefix + "self_attn.v_proj.weight", {kv_proj_dim, spec.hidden_size});
            add_tensor(prefix + "self_attn.o_proj.weight", {spec.hidden_size, attention_proj_dim});
            add_tensor(prefix + "self_attn.q_norm.weight", {spec.attention_head_dim}, "float32");
            add_tensor(prefix + "self_attn.k_norm.weight", {spec.attention_head_dim}, "float32");
        } else {
            add_tensor(prefix + "linear_attn.norm.weight", {spec.linear_value_head_dim}, "float32");
            add_tensor(prefix + "linear_attn.in_proj_qkv.weight", {linear_qkv_dim, spec.hidden_size});
            add_tensor(prefix + "linear_attn.in_proj_z.weight", {linear_value_dim, spec.hidden_size});
            add_tensor(prefix + "linear_attn.in_proj_a.weight", {spec.linear_num_value_heads, spec.hidden_size});
            add_tensor(prefix + "linear_attn.in_proj_b.weight", {spec.linear_num_value_heads, spec.hidden_size});
            add_tensor(prefix + "linear_attn.out_proj.weight", {spec.hidden_size, linear_value_dim});
            add_tensor(prefix + "linear_attn.conv1d.weight", {linear_qkv_dim, 1, spec.linear_conv_kernel_dim});
            add_tensor(prefix + "linear_attn.A_log", {spec.linear_num_value_heads}, spec.recurrent_state_dtype);
            add_tensor(prefix + "linear_attn.dt_bias", {spec.linear_num_value_heads}, spec.recurrent_state_dtype);
        }
    }

    std::string error_message;
    soc::gpu::models::qwen3_5::Qwen3_5ArchitectureSpec parsed_spec;
    if (!soc::gpu::models::qwen3_5::ResolveArchitectureSpec(manifest, &parsed_spec, &error_message)) {
        std::cerr << "failed to parse qwen3.5 manifest spec: " << error_message << '\n';
        return 1;
    }
    if (parsed_spec.max_position_embeddings != 262144 ||
        parsed_spec.rotary_dim != 64 ||
        parsed_spec.layer_types[3] != soc::gpu::models::qwen3_5::Qwen3_5LayerType::kGatedAttention) {
        std::cerr << "parsed qwen3.5 manifest spec is incorrect\n";
        return 1;
    }
    const soc::gpu::models::qwen3_5::Qwen3_5StateLayout parsed_state_layout =
        soc::gpu::models::qwen3_5::BuildStateLayout(parsed_spec);
    if (parsed_state_layout.recurrent_state_dtype != "float32" ||
        parsed_state_layout.recurrent_state_element_bytes != sizeof(float) ||
        parsed_state_layout.deltanet_layers.empty() ||
        parsed_state_layout.deltanet_layers.front().state_element_bytes != sizeof(float)) {
        std::cerr << "parsed qwen3.5 state layout is incorrect\n";
        return 1;
    }
    soc::gpu::models::qwen3_5::Qwen3_5ManifestMetadata metadata;
    if (!soc::gpu::models::qwen3_5::ResolveManifestMetadata(manifest, parsed_spec, &metadata, &error_message)) {
        std::cerr << "failed to resolve qwen3.5 manifest metadata: " << error_message << '\n';
        return 1;
    }
    if (metadata.layers.size() != 32 ||
        metadata.layers[0].linear_attn_in_proj_qkv.name != "model.language_model.layers.0.linear_attn.in_proj_qkv.weight" ||
        metadata.layers[3].self_attn_q_proj.name != "model.language_model.layers.3.self_attn.q_proj.weight") {
        std::cerr << "resolved qwen3.5 manifest metadata is incorrect\n";
        return 1;
    }

    std::cout << "test_qwen3_5_scaffold passed\n";
    return 0;
}
