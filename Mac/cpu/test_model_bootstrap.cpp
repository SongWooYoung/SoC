#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "header/json_parser.h"
#include "header/qwen3_config.h"
#include "header/runtime_assets.h"
#include "header/startup_validator.h"
#include "header/tokenizer_runtime.h"

namespace {
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

ManifestData BuildManifest() {
    ManifestData manifest;
    manifest.format = "soc.cpp.llm_export";
    manifest.format_version = 2;
    manifest.model_id = "Qwen/Qwen3-0.6B";
    manifest.export_dtype = "float16";
    manifest.tokenizer_runtime_file = "tokenizer/tokenizer_runtime.json";
    manifest.config = JsonParser::Parse(
        "{"
        "\"model_type\":\"qwen3\","
        "\"hidden_size\":1024,"
        "\"intermediate_size\":3072,"
        "\"num_hidden_layers\":28,"
        "\"num_attention_heads\":16,"
        "\"num_key_value_heads\":8,"
        "\"head_dim\":128,"
        "\"rms_norm_eps\":1e-6,"
        "\"rope_theta\":1000000,"
        "\"tie_word_embeddings\":true,"
        "\"torch_dtype\":\"bfloat16\","
        "\"vocab_size\":16"
        "}"
    );
    manifest.tensors.push_back(TensorRecord{
        "model.embed_tokens.weight",
        "weights/embed.bin",
        "float16",
        {16, 1024},
        16 * 1024 * 2,
        "model-00001-of-00001.safetensors",
    });
    return manifest;
}

TokenizerRuntimeData BuildTokenizerRuntimeData() {
    TokenizerRuntimeData runtime;
    runtime.format = "soc.cpp.tokenizer_runtime";
    runtime.format_version = 1;
    runtime.tokenizer_class = "Qwen2TokenizerFast";
    runtime.vocab_size = 16;
    runtime.model_max_length = 32768;
    runtime.chat_template = "qwen3-template";
    runtime.template_runtime = TemplateRuntimeData{
        "qwen3",
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "You are a helpful assistant.",
    };
    runtime.added_tokens = {
        {10, "<|im_start|>", true, false, false, false, false},
        {11, "<|im_end|>", true, false, false, false, false},
        {12, "<think>", true, false, false, false, false},
        {13, "</think>", true, false, false, false, false},
    };
    runtime.vocab = {
        {0, "system\n"},
        {1, "assistant\n"},
        {2, "user\n"},
        {3, "You are a helpful assistant."},
        {4, "Hello"},
        {5, " world"},
        {6, "Say hello."},
        {7, "Now think first."},
        {8, "\n"},
        {9, "Hello."},
    };
    return runtime;
}

void test_qwen3_config_parse_and_validate() {
    std::cout << "=== Testing Qwen3 Config Parse ===" << std::endl;
    const Qwen3Config config = Qwen3Config::FromJson(BuildManifest().config);
    require(config.model_type == "qwen3", "config parser must preserve model_type");
    require(config.num_hidden_layers == 28, "config parser must read layer count");
    require(config.HiddenSizePerAttentionHead() == 64, "config must derive hidden size per attention head");
    require(config.tie_word_embeddings, "config parser must read tied embedding setting");
}

void test_startup_validator() {
    std::cout << "=== Testing Startup Validator ===" << std::endl;
    const ManifestData manifest = BuildManifest();
    const TokenizerRuntimeData tokenizer_runtime = BuildTokenizerRuntimeData();

    const Qwen3Config config = StartupValidator::ValidateManifest(manifest);
    require(config.vocab_size == 16, "startup validator must return parsed config");
    StartupValidator::ValidateTokenizerRuntime(tokenizer_runtime);
    StartupValidator::ValidateManifestAndTokenizer(manifest, tokenizer_runtime);

    bool threw_for_vocab_mismatch = false;
    try {
        TokenizerRuntimeData mismatched = tokenizer_runtime;
        mismatched.vocab_size = 17;
        StartupValidator::ValidateManifestAndTokenizer(manifest, mismatched);
    } catch (const std::runtime_error&) {
        threw_for_vocab_mismatch = true;
    }
    require(threw_for_vocab_mismatch, "startup validator must reject manifest/tokenizer vocab mismatches");
}

void test_tokenizer_runtime_minimal_encode_decode() {
    std::cout << "=== Testing Minimal Tokenizer Runtime ===" << std::endl;
    const TokenizerRuntime tokenizer(BuildTokenizerRuntimeData());

    const std::string serialized = "<|im_start|>user\nSay hello.<|im_end|>\n<|im_start|>assistant\nHello world<|im_end|>\n";
    const std::vector<int> encoded = tokenizer.Encode(serialized);
    require(!encoded.empty(), "tokenizer runtime must encode serialized prompt text");
    require(tokenizer.Decode(encoded) == serialized, "tokenizer runtime decode must reconstruct serialized prompt");
    require(tokenizer.TokenToId("<|im_start|>") == 10, "tokenizer runtime must map token content to id");
    require(tokenizer.IdToToken(5) == " world", "tokenizer runtime must map token id to content");

    bool threw_for_unknown = false;
    try {
        static_cast<void>(tokenizer.Encode("unencodable-token"));
    } catch (const std::runtime_error&) {
        threw_for_unknown = true;
    }
    require(threw_for_unknown, "tokenizer runtime must reject unsupported text in minimal mode");
}
}

int main() {
    test_qwen3_config_parse_and_validate();
    test_startup_validator();
    test_tokenizer_runtime_minimal_encode_decode();

    std::cout << "=== Model Bootstrap Tests Completed ===" << std::endl;
    return 0;
}