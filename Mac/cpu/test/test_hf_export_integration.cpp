#include <cstdlib>
#include <cctype>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "header/runtime_pipeline.h"

namespace {
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string ReadEnvOrDefault(const char* name, const char* fallback) {
    if (const char* value = std::getenv(name)) {
        return value;
    }
    return fallback;
}

bool ReadEnvFlag(const char* name, bool fallback) {
    if (const char* value = std::getenv(name)) {
        return std::string(value) != "0";
    }
    return fallback;
}

std::string TrimAsciiWhitespace(std::string value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front())) != 0) {
        value.erase(value.begin());
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())) != 0) {
        value.pop_back();
    }
    return value;
}

std::vector<int> ParseIntListEnv(const char* name) {
    const char* raw_value = std::getenv(name);
    if (raw_value == nullptr) {
        return {};
    }

    std::vector<int> values;
    std::string current;
    const std::string text = raw_value;
    for (char ch : text) {
        if (ch == ',') {
            const std::string trimmed = TrimAsciiWhitespace(current);
            if (!trimmed.empty()) {
                values.push_back(std::stoi(trimmed));
            }
            current.clear();
        } else {
            current.push_back(ch);
        }
    }

    const std::string trimmed = TrimAsciiWhitespace(current);
    if (!trimmed.empty()) {
        values.push_back(std::stoi(trimmed));
    }
    return values;
}

void RequireTokenIdsEqual(const std::vector<int>& actual, const std::vector<int>& expected, const char* message) {
    require(actual.size() == expected.size(), message);
    for (std::size_t index = 0; index < actual.size(); ++index) {
        require(actual[index] == expected[index], message);
    }
}

void test_real_export_bundle_integration() {
    std::cout << "=== Testing Real HF Export Bundle Integration ===" << std::endl;

    const char* manifest_env = std::getenv("SOC_HF_MANIFEST");
    if (manifest_env == nullptr || std::string(manifest_env).empty()) {
        std::cout << "skipped: set SOC_HF_MANIFEST to an exported manifest.json path" << std::endl;
        return;
    }

    const RuntimeBundle bundle = RuntimePipeline::LoadBundle(manifest_env);
    RuntimeGenerationOptions options = RuntimePipeline::GenerationOptionsFromManifest(bundle.manifest);
    options.max_new_tokens = static_cast<std::size_t>(std::stoull(ReadEnvOrDefault("SOC_HF_MAX_NEW_TOKENS", "1")));

    RuntimePromptOptions prompt_options;
    prompt_options.apply_chat_template = ReadEnvFlag("SOC_HF_APPLY_CHAT_TEMPLATE", true);
    prompt_options.enable_thinking = ReadEnvFlag("SOC_HF_ENABLE_THINKING", false);
    prompt_options.system_prompt = ReadEnvOrDefault("SOC_HF_SYSTEM_PROMPT", "");

    const std::string raw_prompt = ReadEnvOrDefault("SOC_HF_PROMPT", "Hello");
    const std::string prepared_prompt = RuntimePipeline::PreparePrompt(bundle.tokenizer_runtime, raw_prompt, prompt_options);

    GenerationSession session = RuntimePipeline::CreateSession(bundle, options);
    const GenerationResult result = session.Generate(prepared_prompt, options.max_new_tokens, options.eos_token_id);

    require(!result.prompt_token_ids.empty(), "integration smoke must tokenize the prepared prompt");
    require(result.generated_token_ids.size() <= options.max_new_tokens,
            "integration smoke must honor max_new_tokens");
    require(result.prompt_token_ids.size() + result.generated_token_ids.size() <= options.max_sequence_length,
            "integration smoke must stay within configured sequence length");

    if (const char* expected_prompt = std::getenv("SOC_HF_EXPECT_PROMPT_TOKEN_IDS")) {
        static_cast<void>(expected_prompt);
        RequireTokenIdsEqual(
            result.prompt_token_ids,
            ParseIntListEnv("SOC_HF_EXPECT_PROMPT_TOKEN_IDS"),
            "integration test prompt token ids must match expected values");
    }
    if (const char* expected_generated = std::getenv("SOC_HF_EXPECT_GENERATED_TOKEN_IDS")) {
        static_cast<void>(expected_generated);
        RequireTokenIdsEqual(
            result.generated_token_ids,
            ParseIntListEnv("SOC_HF_EXPECT_GENERATED_TOKEN_IDS"),
            "integration test generated token ids must match expected values");
    }
    if (const char* expected_text = std::getenv("SOC_HF_EXPECT_GENERATED_TEXT")) {
        require(result.generated_text == expected_text,
                "integration test generated text must match expected value");
    }
}
}

int main() {
    test_real_export_bundle_integration();

    std::cout << "=== HF Export Integration Test Completed ===" << std::endl;
    return 0;
}