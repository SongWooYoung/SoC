#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cstdint>
#include <vector>

#include "header/dtype.h"
#include "header/json_parser.h"
#include "header/qwen_model_loader.h"
#include "header/runtime_assets.h"

namespace {
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void write_binary_file(const std::filesystem::path& path, const std::vector<float>& values) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("failed to create binary test file");
    }
    stream.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
}

void write_binary_file_u16(const std::filesystem::path& path, const std::vector<std::uint16_t>& values) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("failed to create binary test file");
    }
    stream.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(std::uint16_t)));
}

TensorRecord WriteTensor(
    const std::filesystem::path& weights_dir,
    const std::string& name,
    const std::vector<std::size_t>& shape,
    const std::vector<float>& values,
    const std::string& filename) {
    const std::filesystem::path path = weights_dir / filename;
    write_binary_file(path, values);
    return TensorRecord{name, path.string(), "float32", shape, values.size() * sizeof(float), "model-00001-of-00001.safetensors"};
}

TensorRecord WriteTensorFloat16(
    const std::filesystem::path& weights_dir,
    const std::string& name,
    const std::vector<std::size_t>& shape,
    const std::vector<float>& values,
    const std::string& filename) {
    std::vector<std::uint16_t> encoded(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        encoded[index] = Float32ToFloat16(values[index]);
    }
    const std::filesystem::path path = weights_dir / filename;
    write_binary_file_u16(path, encoded);
    return TensorRecord{name, path.string(), "float16", shape, values.size() * sizeof(std::uint16_t), "model-00001-of-00001.safetensors"};
}

TensorRecord WriteTensorBFloat16(
    const std::filesystem::path& weights_dir,
    const std::string& name,
    const std::vector<std::size_t>& shape,
    const std::vector<float>& values,
    const std::string& filename) {
    std::vector<std::uint16_t> encoded(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        encoded[index] = Float32ToBFloat16(values[index]);
    }
    const std::filesystem::path path = weights_dir / filename;
    write_binary_file_u16(path, encoded);
    return TensorRecord{name, path.string(), "bfloat16", shape, values.size() * sizeof(std::uint16_t), "model-00001-of-00001.safetensors"};
}

ManifestData BuildLoaderManifest(const std::filesystem::path& weights_dir, const std::string& export_dtype) {
    ManifestData manifest;
    manifest.format = "soc.cpp.llm_export";
    manifest.format_version = 2;
    manifest.model_id = "Qwen/Qwen3-0.6B-test";
    manifest.export_dtype = export_dtype;
    manifest.tokenizer_runtime_file = "tokenizer/tokenizer_runtime.json";
    manifest.config = JsonParser::Parse(
        "{"
        "\"model_type\":\"qwen3\","
        "\"hidden_size\":2,"
        "\"intermediate_size\":2,"
        "\"num_hidden_layers\":1,"
        "\"num_attention_heads\":1,"
        "\"num_key_value_heads\":1,"
        "\"head_dim\":2,"
        "\"rms_norm_eps\":1e-6,"
        "\"rope_theta\":10000,"
        "\"tie_word_embeddings\":true,"
        "\"torch_dtype\":\"" + export_dtype + "\" ,"
        "\"vocab_size\":2"
        "}"
    );

    if (export_dtype == "float32") {
        manifest.tensors = {
            WriteTensor(weights_dir, "model.embed_tokens.weight", {2, 2}, {1.0f, 0.0f, 0.0f, 1.0f}, "embed.bin"),
            WriteTensor(weights_dir, "model.layers.0.input_layernorm.weight", {2}, {1.0f, 1.0f}, "in_norm.bin"),
            WriteTensor(weights_dir, "model.layers.0.self_attn.q_proj.weight", {2, 2}, {0.0f, 0.0f, 0.0f, 0.0f}, "q.bin"),
            WriteTensor(weights_dir, "model.layers.0.self_attn.k_proj.weight", {2, 2}, {0.0f, 0.0f, 0.0f, 0.0f}, "k.bin"),
            WriteTensor(weights_dir, "model.layers.0.self_attn.v_proj.weight", {2, 2}, {0.0f, 0.0f, 0.0f, 0.0f}, "v.bin"),
            WriteTensor(weights_dir, "model.layers.0.self_attn.o_proj.weight", {2, 2}, {0.0f, 0.0f, 0.0f, 0.0f}, "o.bin"),
            WriteTensor(weights_dir, "model.layers.0.post_attention_layernorm.weight", {2}, {1.0f, 1.0f}, "post_norm.bin"),
            WriteTensor(weights_dir, "model.layers.0.mlp.gate_proj.weight", {2, 2}, {0.0f, 0.0f, 0.0f, 0.0f}, "gate.bin"),
            WriteTensor(weights_dir, "model.layers.0.mlp.up_proj.weight", {2, 2}, {0.0f, 0.0f, 0.0f, 0.0f}, "up.bin"),
            WriteTensor(weights_dir, "model.layers.0.mlp.down_proj.weight", {2, 2}, {0.0f, 0.0f, 0.0f, 0.0f}, "down.bin"),
            WriteTensor(weights_dir, "model.norm.weight", {2}, {1.0f, 1.0f}, "final_norm.bin"),
        };
    } else {
        manifest.tensors = {
            WriteTensorFloat16(weights_dir, "model.embed_tokens.weight", {2, 2}, {1.0f, 0.0f, 0.0f, 1.0f}, "embed.bin"),
            WriteTensorBFloat16(weights_dir, "model.layers.0.input_layernorm.weight", {2}, {1.0f, 1.0f}, "in_norm.bin"),
            WriteTensorFloat16(weights_dir, "model.layers.0.self_attn.q_proj.weight", {2, 2}, {0.0f, 0.0f, 0.0f, 0.0f}, "q.bin"),
            WriteTensorFloat16(weights_dir, "model.layers.0.self_attn.k_proj.weight", {2, 2}, {0.0f, 0.0f, 0.0f, 0.0f}, "k.bin"),
            WriteTensorFloat16(weights_dir, "model.layers.0.self_attn.v_proj.weight", {2, 2}, {0.0f, 0.0f, 0.0f, 0.0f}, "v.bin"),
            WriteTensorFloat16(weights_dir, "model.layers.0.self_attn.o_proj.weight", {2, 2}, {0.0f, 0.0f, 0.0f, 0.0f}, "o.bin"),
            WriteTensorBFloat16(weights_dir, "model.layers.0.post_attention_layernorm.weight", {2}, {1.0f, 1.0f}, "post_norm.bin"),
            WriteTensorFloat16(weights_dir, "model.layers.0.mlp.gate_proj.weight", {2, 2}, {0.0f, 0.0f, 0.0f, 0.0f}, "gate.bin"),
            WriteTensorFloat16(weights_dir, "model.layers.0.mlp.up_proj.weight", {2, 2}, {0.0f, 0.0f, 0.0f, 0.0f}, "up.bin"),
            WriteTensorFloat16(weights_dir, "model.layers.0.mlp.down_proj.weight", {2, 2}, {0.0f, 0.0f, 0.0f, 0.0f}, "down.bin"),
            WriteTensorBFloat16(weights_dir, "model.norm.weight", {2}, {1.0f, 1.0f}, "final_norm.bin"),
        };
    }

    return manifest;
}

void test_qwen_model_loader_smoke(const std::string& export_dtype) {
    std::cout << "=== Testing Qwen Model Loader Smoke (" << export_dtype << ") ===" << std::endl;
    const std::filesystem::path temp_dir = std::filesystem::temp_directory_path() / ("soc_qwen_model_loader_test_" + export_dtype);
    const std::filesystem::path weights_dir = temp_dir / "weights";
    std::filesystem::create_directories(weights_dir);

    const ManifestData manifest = BuildLoaderManifest(weights_dir, export_dtype);
    const QwenCausalLM model = QwenModelLoader::LoadModel(manifest);

    const int32_t token_ids_raw[2] = {0, 1};
    const Tensor token_ids(Storage::FromOwnedCopy(token_ids_raw, sizeof(token_ids_raw)), DType::Int32, {1, 2});
    const Tensor logits = model.ForwardLogits(token_ids, 0);

    require(logits.shape() == std::vector<std::size_t>({1, 2, 2}), "loaded model must produce logits with shape [batch, seq, vocab]");
    const float* logits_data = logits.data<const float>();
    require(logits_data[0] > logits_data[1], "token 0 hidden state should favor vocab row 0 under tied embeddings");
    require(logits_data[3] > logits_data[2], "token 1 hidden state should favor vocab row 1 under tied embeddings");

    std::filesystem::remove_all(temp_dir);
}
}

int main() {
    test_qwen_model_loader_smoke("float32");
    test_qwen_model_loader_smoke("float16");

    std::cout << "=== Model Loader Tests Completed ===" << std::endl;
    return 0;
}