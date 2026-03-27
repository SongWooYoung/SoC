#ifndef QWEN_MODEL_LOADER_H
#define QWEN_MODEL_LOADER_H

#include <string>

#include "header/qwen_causal_lm.h"
#include "header/qwen3_config.h"
#include "header/runtime_assets.h"

class QwenModelLoader {
public:
    static QwenCausalLM LoadModel(const ManifestData& manifest);
    static QwenCausalLM LoadModelFromFile(const std::string& manifest_path);

private:
    static Qwen3Config ValidateAndReadConfig(const ManifestData& manifest);
};

#endif