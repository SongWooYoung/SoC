#pragma once

#include <string>

#include "asset/runtime_assets.h"
#include "model/qwen_causal_lm.h"

namespace soc::gpu {

class MetalContext;

class QwenModelLoader {
public:
    static bool LoadModel(const MetalContext& context,
                          const ManifestData& manifest,
                          QwenCausalLM* model,
                          std::string* error_message);
    static bool LoadModelFromFile(const MetalContext& context,
                                  const std::string& manifest_path,
                                  QwenCausalLM* model,
                                  std::string* error_message);
};

}  // namespace soc::gpu