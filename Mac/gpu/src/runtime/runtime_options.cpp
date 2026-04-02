#include "runtime/runtime_options.h"

#include "header/runtime_pipeline.h"

namespace soc::gpu {

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

}  // namespace

RuntimeGenerationOptions ResolveGenerationOptions(const InferCliOptions& cli, const ::ManifestData& manifest) {
    RuntimeGenerationOptions resolved;
    if (manifest.generation_config.is_object()) {
        resolved.max_new_tokens = ReadOptionalSize(manifest.generation_config, "max_new_tokens", resolved.max_new_tokens);
        resolved.sampler.temperature = ReadOptionalFloat(manifest.generation_config, "temperature", resolved.sampler.temperature);
        resolved.sampler.top_k = ReadOptionalSize(manifest.generation_config, "top_k", resolved.sampler.top_k);
        resolved.eos_token_id = ReadOptionalInt(manifest.generation_config, "eos_token_id", resolved.eos_token_id);
        resolved.max_sequence_length = ReadOptionalSize(manifest.generation_config, "max_length", resolved.max_sequence_length);
    }

    if (cli.override_max_new_tokens) {
        resolved.max_new_tokens = cli.generation.max_new_tokens;
    }
    if (cli.override_temperature) {
        resolved.sampler.temperature = cli.generation.sampler.temperature;
    }
    if (cli.override_top_k) {
        resolved.sampler.top_k = cli.generation.sampler.top_k;
    }
    if (cli.override_eos_token_id) {
        resolved.eos_token_id = cli.generation.eos_token_id;
    }
    if (cli.override_max_sequence_length) {
        resolved.max_sequence_length = cli.generation.max_sequence_length;
    }
    return resolved;
}

::SamplerConfig ToCpuSamplerConfig(const RuntimeSamplerOptions& options) {
    ::SamplerConfig sampler;
    sampler.temperature = options.temperature;
    sampler.top_k = options.top_k;
    return sampler;
}

::RuntimeGenerationOptions ToCpuRuntimeGenerationOptions(const RuntimeGenerationOptions& options) {
    ::RuntimeGenerationOptions converted;
    converted.max_new_tokens = options.max_new_tokens;
    converted.sampler = ToCpuSamplerConfig(options.sampler);
    converted.eos_token_id = options.eos_token_id;
    converted.max_sequence_length = options.max_sequence_length;
    return converted;
}

::RuntimePromptOptions ToCpuRuntimePromptOptions(const RuntimePromptOptions& options) {
    ::RuntimePromptOptions converted;
    converted.apply_chat_template = options.apply_chat_template;
    converted.add_generation_prompt = options.add_generation_prompt;
    converted.enable_thinking = options.enable_thinking;
    converted.system_prompt = options.system_prompt;
    return converted;
}

}  // namespace soc::gpu
