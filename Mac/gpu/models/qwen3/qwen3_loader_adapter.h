#pragma once

#include <string>

#include "asset/runtime_assets.h"
#include "model/qwen_causal_lm.h"

namespace soc::gpu {
class MetalContext;
}

namespace soc::gpu::models::qwen3 {

bool LoadGpuModel(const MetalContext& context,
                  const ManifestData& manifest,
                  QwenCausalLM* model,
                  std::string* error_message);

bool LoadGpuModelFromFile(const MetalContext& context,
                          const std::string& manifest_path,
                          QwenCausalLM* model,
                          std::string* error_message);

}  // namespace soc::gpu::models::qwen3
