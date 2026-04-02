#include "models/qwen3/qwen3_registry.h"

namespace soc::gpu::models::qwen3 {

bool IsQwen3Manifest(const ManifestData& manifest) {
    if (manifest.model_id.rfind("Qwen/Qwen3-", 0) == 0) {
        return true;
    }
    if (manifest.config.is_object() &&
        manifest.config.contains("model_type") &&
        manifest.config.at("model_type").is_string() &&
        manifest.config.at("model_type").as_string() == "qwen3") {
        return true;
    }
    return false;
}

}  // namespace soc::gpu::models::qwen3
