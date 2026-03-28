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

std::vector<std::string> BuildByteToUnicodeMap() {
    std::vector<int> byte_values;
    for (int value = static_cast<int>('!'); value <= static_cast<int>('~'); ++value) {
        byte_values.push_back(value);
    }
    for (int value = 161; value <= 172; ++value) {
        byte_values.push_back(value);
    }
    for (int value = 174; value <= 255; ++value) {
        byte_values.push_back(value);
    }

    std::vector<int> code_points = byte_values;
    int additional = 0;
    for (int byte_value = 0; byte_value < 256; ++byte_value) {
        bool present = false;
        for (int existing : byte_values) {
            if (existing == byte_value) {
                present = true;
                break;
            }
        }
        if (present) {
            continue;
        }
        byte_values.push_back(byte_value);
        code_points.push_back(256 + additional);
        ++additional;
    }

    std::vector<std::string> mapping(256);
    for (std::size_t index = 0; index < byte_values.size(); ++index) {
        const int code_point = code_points[index];
        std::string utf8;
        if (code_point <= 0x7F) {
            utf8.push_back(static_cast<char>(code_point));
        } else if (code_point <= 0x7FF) {
            utf8.push_back(static_cast<char>(0xC0 | ((code_point >> 6) & 0x1F)));
            utf8.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
        } else {
            utf8.push_back(static_cast<char>(0xE0 | ((code_point >> 12) & 0x0F)));
            utf8.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
            utf8.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
        }
        mapping[byte_values[index]] = utf8;
    }

    return mapping;
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

void test_tokenizer_runtime_bpe_fallback() {
    std::cout << "=== Testing Tokenizer Runtime BPE Fallback ===" << std::endl;
    TokenizerRuntimeData runtime = BuildTokenizerRuntimeData();
    runtime.added_tokens.clear();
    runtime.vocab = {
        {0, "h"},
        {1, "e"},
        {2, "l"},
        {3, "o"},
        {4, " "},
        {5, "w"},
        {6, "r"},
        {7, "d"},
        {8, "he"},
        {9, "ll"},
        {10, "hell"},
        {11, "hello"},
        {12, " world"},
    };
    runtime.vocab_size = 13;
    runtime.bpe_model.enabled = true;
    runtime.bpe_model.type = "bpe";
    runtime.bpe_model.merges = {
        {"h", "e"},
        {"l", "l"},
        {"he", "ll"},
        {"hell", "o"},
        {" ", "w"},
        {" w", "o"},
        {" wo", "r"},
        {" wor", "l"},
        {" worl", "d"},
    };

    const TokenizerRuntime tokenizer(runtime);
    const std::vector<int> encoded = tokenizer.Encode("hello world");
    require(encoded.size() == 2, "tokenizer runtime bpe fallback must merge prompt into expected tokens");
    require(encoded[0] == 11 && encoded[1] == 12, "tokenizer runtime bpe fallback must emit merged token ids");
    require(tokenizer.Decode(encoded) == "hello world", "tokenizer runtime bpe fallback decode must reconstruct text");
}

void test_tokenizer_runtime_byte_level_prefix_space() {
    std::cout << "=== Testing Tokenizer Runtime Byte-Level Prefix Space ===" << std::endl;
    TokenizerRuntimeData runtime = BuildTokenizerRuntimeData();
    runtime.added_tokens.clear();
    runtime.vocab = {
        {0, "Ġh"},
        {1, "e"},
        {2, "l"},
        {3, "o"},
        {4, "Ġw"},
        {5, "r"},
        {6, "d"},
        {7, "Ġhe"},
        {8, "ll"},
        {9, "Ġhell"},
        {10, "Ġhello"},
        {11, "Ġwo"},
        {12, "Ġwor"},
        {13, "Ġworl"},
        {14, "Ġworld"},
    };
    runtime.vocab_size = 15;
    runtime.bpe_model.enabled = true;
    runtime.bpe_model.type = "bpe";
    runtime.bpe_model.merges = {
        {"Ġ", "h"},
        {"Ġh", "e"},
        {"l", "l"},
        {"Ġhe", "ll"},
        {"Ġhell", "o"},
        {"Ġ", "w"},
        {"Ġw", "o"},
        {"Ġwo", "r"},
        {"Ġwor", "l"},
        {"Ġworl", "d"},
    };
    runtime.pre_tokenizer.enabled = true;
    runtime.pre_tokenizer.type = "ByteLevel";
    runtime.pre_tokenizer.byte_level.enabled = true;
    runtime.pre_tokenizer.byte_level.add_prefix_space = true;
    runtime.pre_tokenizer.byte_level.use_regex = true;
        runtime.decoder.enabled = true;
        runtime.decoder.type = "ByteLevel";
        runtime.decoder.byte_level.enabled = true;
        runtime.decoder.byte_level.byte_to_unicode = BuildByteToUnicodeMap();

    const TokenizerRuntime tokenizer(runtime);
    const std::vector<int> encoded = tokenizer.Encode("hello world");
    require(encoded.size() == 2, "byte-level pre-tokenizer must keep word boundaries before BPE");
    require(encoded[0] == 10 && encoded[1] == 14,
            "byte-level pre-tokenizer must add prefix space to the first piece");
    require(tokenizer.Decode(encoded) == " hello world",
            "decoded byte-level tokens must reflect stored leading-space token content");
}
}

int main() {
    test_qwen3_config_parse_and_validate();
    test_startup_validator();
    test_tokenizer_runtime_minimal_encode_decode();
    test_tokenizer_runtime_bpe_fallback();
    test_tokenizer_runtime_byte_level_prefix_space();

    std::cout << "=== Model Bootstrap Tests Completed ===" << std::endl;
    return 0;
}