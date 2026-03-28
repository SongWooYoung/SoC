#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "header/json_parser.h"
#include "header/runtime_pipeline.h"
#include "header/runtime_assets.h"

namespace {
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void write_text_file(const std::filesystem::path& path, const std::string& contents) {
    std::ofstream stream(path);
    if (!stream) {
        throw std::runtime_error("failed to create test file");
    }
    stream << contents;
}

void test_json_parser() {
    std::cout << "=== Testing JSON Parser ===" << std::endl;
    const JsonValue root = JsonParser::Parse(
        "{\"name\":\"qwen\",\"version\":2,\"flags\":[true,false],\"meta\":{\"rope_theta\":1000000}}"
    );

    require(root.at("name").as_string() == "qwen", "json parser must read string fields");
    require(root.at("version").as_int() == 2, "json parser must read integer fields");
    require(root.at("flags").as_array().size() == 2, "json parser must read array fields");
    require(root.at("meta").at("rope_theta").as_int() == 1000000, "json parser must read nested objects");
}

void test_manifest_loader() {
    std::cout << "=== Testing Manifest Loader ===" << std::endl;
    const std::filesystem::path temp_dir = std::filesystem::temp_directory_path() / "soc_manifest_loader_test";
    std::filesystem::create_directories(temp_dir / "weights");
    std::filesystem::create_directories(temp_dir / "tokenizer");

    write_text_file(
        temp_dir / "manifest.json",
        "{\n"
        "  \"format\": \"soc.cpp.llm_export\",\n"
        "  \"format_version\": 2,\n"
        "  \"model_id\": \"Qwen/Qwen3-0.6B\",\n"
        "  \"source_dir\": \"/tmp/source\",\n"
        "  \"export_dtype\": \"float16\",\n"
        "  \"tokenizer_runtime_file\": \"tokenizer/tokenizer_runtime.json\",\n"
        "  \"config\": {\"hidden_size\": 1024},\n"
        "  \"generation_config\": {\"max_new_tokens\": 32},\n"
        "  \"tensors\": [\n"
        "    {\n"
        "      \"name\": \"model.embed_tokens.weight\",\n"
        "      \"file\": \"weights/embed.bin\",\n"
        "      \"dtype\": \"float16\",\n"
        "      \"shape\": [151936, 1024],\n"
        "      \"byte_size\": 311164928,\n"
        "      \"source_shard\": \"model-00001-of-00002.safetensors\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
    );

    const ManifestData manifest = ManifestLoader::LoadFromFile((temp_dir / "manifest.json").string());
    require(manifest.format == "soc.cpp.llm_export", "manifest loader must read export format");
    require(manifest.format_version == 2, "manifest loader must read format version");
    require(manifest.tensors.size() == 1, "manifest loader must read tensor records");
    require(manifest.tensors[0].shape.size() == 2, "manifest loader must read tensor shapes");
    require(manifest.tensors[0].file.find("weights/embed.bin") != std::string::npos, "manifest loader must resolve tensor files");
    require(manifest.tokenizer_runtime_file.find("tokenizer/tokenizer_runtime.json") != std::string::npos,
            "manifest loader must resolve tokenizer runtime path");

    std::filesystem::remove_all(temp_dir);
}

void test_tokenizer_runtime_loader() {
    std::cout << "=== Testing Tokenizer Runtime Loader ===" << std::endl;
    const std::filesystem::path temp_dir = std::filesystem::temp_directory_path() / "soc_tokenizer_runtime_test";
    std::filesystem::create_directories(temp_dir);

    write_text_file(
        temp_dir / "tokenizer_runtime.json",
        "{\n"
        "  \"format\": \"soc.cpp.tokenizer_runtime\",\n"
        "  \"format_version\": 1,\n"
        "  \"tokenizer_class\": \"Qwen2TokenizerFast\",\n"
        "  \"vocab_size\": 151936,\n"
        "  \"model_max_length\": 131072,\n"
        "  \"special_tokens_map\": {\"bos_token\": \"<|endoftext|>\"},\n"
        "  \"special_token_ids\": {\"bos_token\": 151643},\n"
        "  \"added_tokens\": [\n"
        "    {\"id\": 151644, \"content\": \"<|im_start|>\", \"special\": true, \"single_word\": false, \"lstrip\": false, \"rstrip\": false, \"normalized\": false}\n"
        "  ],\n"
        "  \"vocab\": [\n"
        "    {\"id\": 0, \"token\": \"!\"},\n"
        "    {\"id\": 1, \"token\": \"\\\"\"}\n"
        "  ],\n"
        "  \"bpe_model\": {\n"
        "    \"type\": \"bpe\",\n"
        "    \"unk_token\": null,\n"
        "    \"continuing_subword_prefix\": \"\",\n"
        "    \"end_of_word_suffix\": \"\",\n"
        "    \"merges\": [\n"
        "      {\"left\": \"h\", \"right\": \"e\"},\n"
        "      {\"left\": \"he\", \"right\": \"llo\"}\n"
        "    ]\n"
        "  },\n"
        "  \"pre_tokenizer\": {\n"
        "    \"type\": \"ByteLevel\",\n"
        "    \"byte_level\": {\n"
        "      \"enabled\": true,\n"
        "      \"add_prefix_space\": true,\n"
        "      \"use_regex\": true\n"
        "    }\n"
        "  },\n"
        "  \"decoder\": {\n"
        "    \"type\": \"Sequence\",\n"
        "    \"byte_level\": {\n"
        "      \"enabled\": true,\n"
        "      \"add_prefix_space\": false,\n"
        "      \"trim_offsets\": false,\n"
        "      \"use_regex\": true,\n"
        "      \"byte_to_unicode\": [\"!\", \"\\\"\"]\n"
        "    },\n"
        "    \"bpe\": {\n"
        "      \"enabled\": true,\n"
        "      \"suffix\": \"</w>\"\n"
        "    }\n"
        "  },\n"
        "  \"chat_template\": \"dummy-template\",\n"
        "  \"template_runtime\": {\n"
        "    \"type\": \"qwen3\",\n"
        "    \"im_start\": \"<|im_start|>\",\n"
        "    \"im_end\": \"<|im_end|>\",\n"
        "    \"think_start\": \"<think>\",\n"
        "    \"think_end\": \"</think>\",\n"
        "    \"default_system_prompt\": \"You are a helpful assistant.\"\n"
        "  },\n"
        "  \"copied_files\": [\"tokenizer.json\"]\n"
        "}\n"
    );

    const TokenizerRuntimeData runtime = TokenizerRuntimeLoader::LoadFromFile((temp_dir / "tokenizer_runtime.json").string());
    require(runtime.format == "soc.cpp.tokenizer_runtime", "tokenizer runtime loader must read format");
    require(runtime.vocab_size == 151936, "tokenizer runtime loader must read vocab size");
    require(runtime.template_runtime.im_start == "<|im_start|>", "tokenizer runtime loader must read template runtime");
    require(runtime.added_tokens.size() == 1, "tokenizer runtime loader must read added tokens");
    require(runtime.vocab.size() == 2, "tokenizer runtime loader must read vocab entries");
    require(runtime.bpe_model.enabled, "tokenizer runtime loader must read optional bpe model");
    require(runtime.bpe_model.merges.size() == 2, "tokenizer runtime loader must read bpe merges");
    require(runtime.pre_tokenizer.enabled, "tokenizer runtime loader must read pre_tokenizer metadata");
    require(runtime.pre_tokenizer.byte_level.add_prefix_space,
            "tokenizer runtime loader must read byte-level add_prefix_space");
        require(runtime.decoder.enabled, "tokenizer runtime loader must read decoder metadata");
        require(runtime.decoder.bpe.suffix == "</w>", "tokenizer runtime loader must read bpe decoder suffix");
        require(runtime.decoder.byte_level.byte_to_unicode.size() == 2,
            "tokenizer runtime loader must read byte-to-unicode mappings");

    std::filesystem::remove_all(temp_dir);
}

void test_qwen3_template_builder() {
    std::cout << "=== Testing Qwen3 Template Builder ===" << std::endl;
    const TemplateRuntimeData template_runtime{
        "qwen3",
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "You are a helpful assistant.",
    };
    const Qwen3TemplateBuilder builder(template_runtime);

    const std::vector<ChatMessage> messages = {
        {"user", "Say hello."},
        {"assistant", "Hello."},
        {"user", "Now think first."},
    };

    const std::string prompt = builder.BuildPrompt(messages, true, true);
    require(prompt.find("<|im_start|>system\nYou are a helpful assistant.<|im_end|>") != std::string::npos,
            "template builder must inject default system prompt");
    require(prompt.find("<|im_start|>user\nSay hello.<|im_end|>") != std::string::npos,
            "template builder must serialize user messages");
    require(prompt.find("<|im_start|>assistant\n<think>\n") != std::string::npos,
            "template builder must open assistant generation with thinking tag");

    const std::string no_think_prompt = builder.BuildPrompt(messages, true, false, "Custom system prompt.");
    require(no_think_prompt.find("Custom system prompt.") != std::string::npos,
            "template builder must allow overriding the default system prompt");
    require(no_think_prompt.find("<think>") == std::string::npos,
            "template builder must omit think tag when thinking is disabled");
}

void test_runtime_pipeline_helpers() {
    std::cout << "=== Testing Runtime Pipeline Helpers ===" << std::endl;

    ManifestData manifest;
    manifest.generation_config = JsonParser::Parse(
        "{"
        "\"max_new_tokens\":12,"
        "\"temperature\":0.7,"
        "\"top_k\":4,"
        "\"eos_token_id\":[151645],"
        "\"max_length\":1024"
        "}"
    );

    const RuntimeGenerationOptions options = RuntimePipeline::GenerationOptionsFromManifest(manifest);
    require(options.max_new_tokens == 12, "runtime pipeline must load max_new_tokens from generation_config");
    require(options.sampler.top_k == 4, "runtime pipeline must load top_k from generation_config");
    require(options.eos_token_id == 151645, "runtime pipeline must accept eos_token_id arrays from generation_config");
    require(options.max_sequence_length == 1024, "runtime pipeline must map max_length to sequence length");

    TokenizerRuntimeData tokenizer_runtime;
    tokenizer_runtime.template_runtime = TemplateRuntimeData{
        "qwen3",
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "You are a helpful assistant.",
    };

    RuntimePromptOptions prompt_options;
    prompt_options.apply_chat_template = true;
    prompt_options.enable_thinking = false;
    prompt_options.system_prompt = "System override.";
    const std::string prepared_prompt = RuntimePipeline::PreparePrompt(tokenizer_runtime, "Hello", prompt_options);
    require(prepared_prompt.find("System override.") != std::string::npos,
            "runtime pipeline must pass through system prompt overrides");
    require(prepared_prompt.find("<|im_start|>user\nHello<|im_end|>") != std::string::npos,
            "runtime pipeline must wrap prompt as a user chat turn");
    require(prepared_prompt.find("<think>") == std::string::npos,
            "runtime pipeline must omit thinking tag when disabled");
}
}

int main() {
    test_json_parser();
    test_manifest_loader();
    test_tokenizer_runtime_loader();
    test_qwen3_template_builder();
    test_runtime_pipeline_helpers();

    std::cout << "=== Runtime Asset Tests Completed ===" << std::endl;
    return 0;
}