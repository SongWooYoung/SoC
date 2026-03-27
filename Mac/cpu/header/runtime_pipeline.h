#ifndef RUNTIME_PIPELINE_H
#define RUNTIME_PIPELINE_H

#include <cstddef>
#include <string>

#include "header/generation_session.h"
#include "header/qwen_model_loader.h"
#include "header/runtime_assets.h"
#include "header/sampler.h"
#include "header/tokenizer_runtime.h"

struct RuntimeGenerationOptions {
    std::size_t max_new_tokens = 32;
    SamplerConfig sampler;
    int eos_token_id = -1;
    std::size_t max_sequence_length = 256;
};

struct RuntimeBundle {
    ManifestData manifest;
    TokenizerRuntimeData tokenizer_runtime;
    QwenCausalLM model;
};

class RuntimePipeline {
public:
    static RuntimeBundle LoadBundle(const std::string& manifest_path);
    static GenerationSession CreateSession(const RuntimeBundle& bundle, const RuntimeGenerationOptions& options = {});
    static GenerationResult Generate(const std::string& manifest_path, const std::string& prompt, const RuntimeGenerationOptions& options = {});
};

#endif