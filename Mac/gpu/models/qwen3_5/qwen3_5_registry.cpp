#include "models/qwen3_5/qwen3_5_registry.h"

namespace soc::gpu::models::qwen3_5 {

bool IsQwen3_5Manifest(const ManifestData& manifest) {
    if (manifest.model_id.rfind("Qwen/Qwen3.5-", 0) == 0) {
        return true;
    }
    if (manifest.config.is_object() &&
        manifest.config.contains("model_type") &&
        manifest.config.at("model_type").is_string() &&
        manifest.config.at("model_type").as_string() == "qwen3_5") {
        return true;
    }
    return false;
}

}  // namespace soc::gpu::models::qwen3_5
