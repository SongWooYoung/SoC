#include "header/runtime_pipeline.h"

#include "header/startup_validator.h"

namespace {
int ReadOptionalInt(const JsonValue& object, const char* key, int fallback) {
    if (!object.is_object() || !object.contains(key)) {
        return fallback;
    }
    const JsonValue& value = object.at(key);
    if (value.is_array()) {
        const JsonValue::Array& values = value.as_array();
        if (values.empty()) {
            return fallback;
        }
        return values.front().as_int();
    }
    return value.as_int();
}

std::size_t ReadOptionalSize(const JsonValue& object, const char* key, std::size_t fallback) {
    if (!object.is_object() || !object.contains(key)) {
        return fallback;
    }
    return static_cast<std::size_t>(object.at(key).as_int64());
}

float ReadOptionalFloat(const JsonValue& object, const char* key, float fallback) {
    if (!object.is_object() || !object.contains(key)) {
        return fallback;
    }
    return static_cast<float>(object.at(key).as_number());
}
}

RuntimeBundle RuntimePipeline::LoadBundle(const std::string& manifest_path) {
    const ManifestData manifest = ManifestLoader::LoadFromFile(manifest_path);
    const TokenizerRuntimeData tokenizer_runtime = TokenizerRuntimeLoader::LoadFromFile(manifest.tokenizer_runtime_file);
    StartupValidator::ValidateManifestAndTokenizer(manifest, tokenizer_runtime);
    return RuntimeBundle{manifest, tokenizer_runtime, QwenModelLoader::LoadModel(manifest)};
}

RuntimeGenerationOptions RuntimePipeline::GenerationOptionsFromManifest(const ManifestData& manifest) {
    RuntimeGenerationOptions options;
    if (!manifest.generation_config.is_object()) {
        return options;
    }

    options.max_new_tokens = ReadOptionalSize(manifest.generation_config, "max_new_tokens", options.max_new_tokens);
    options.sampler.temperature = ReadOptionalFloat(manifest.generation_config, "temperature", options.sampler.temperature);
    options.sampler.top_k = ReadOptionalSize(manifest.generation_config, "top_k", options.sampler.top_k);
    options.eos_token_id = ReadOptionalInt(manifest.generation_config, "eos_token_id", options.eos_token_id);
    options.max_sequence_length = ReadOptionalSize(manifest.generation_config, "max_length", options.max_sequence_length);
    return options;
}

std::string RuntimePipeline::PreparePrompt(
    const TokenizerRuntimeData& tokenizer_runtime,
    const std::string& prompt,
    const RuntimePromptOptions& options) {
    if (!options.apply_chat_template) {
        return prompt;
    }

    const Qwen3TemplateBuilder builder(tokenizer_runtime.template_runtime);
    return builder.BuildPrompt(
        std::vector<ChatMessage>{{"user", prompt}},
        options.add_generation_prompt,
        options.enable_thinking,
        options.system_prompt);
}

GenerationSession RuntimePipeline::CreateSession(const RuntimeBundle& bundle, const RuntimeGenerationOptions& options) {
    return GenerationSession(
        bundle.model,
        TokenizerRuntime(bundle.tokenizer_runtime),
        Sampler(options.sampler),
        options.max_sequence_length);
}

GenerationResult RuntimePipeline::Generate(const std::string& manifest_path, const std::string& prompt, const RuntimeGenerationOptions& options) {
    const RuntimeBundle bundle = LoadBundle(manifest_path);
    GenerationSession session = CreateSession(bundle, options);
    return session.Generate(prompt, options.max_new_tokens, options.eos_token_id);
}