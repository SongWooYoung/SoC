#include "models/qwen3/qwen3_loader_adapter.h"

#include "model/qwen_model_loader.h"

namespace soc::gpu::models::qwen3 {

bool LoadGpuModel(const MetalContext& context,
                  const ManifestData& manifest,
                  QwenCausalLM* model,
                  std::string* error_message) {
    return QwenModelLoader::LoadModel(context, manifest, model, error_message);
}

bool LoadGpuModelFromFile(const MetalContext& context,
                          const std::string& manifest_path,
                          QwenCausalLM* model,
                          std::string* error_message) {
    return QwenModelLoader::LoadModelFromFile(context, manifest_path, model, error_message);
}

}  // namespace soc::gpu::models::qwen3
