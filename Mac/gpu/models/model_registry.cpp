#include "models/model_registry.h"

#include "models/qwen3/qwen3_registry.h"
#include "models/qwen3_5/qwen3_5_registry.h"

namespace soc::gpu {
namespace {

ModelSelection MakeSelection(const ModelArchitecture architecture, const bool resolved_from_cli) {
    return {architecture,
            ModelArchitectureName(architecture),
            ModelArchitectureDisplayName(architecture),
            resolved_from_cli};
}

}  // namespace

const char* ModelArchitectureName(const ModelArchitecture architecture) {
    switch (architecture) {
        case ModelArchitecture::kQwen3:
            return "qwen3";
        case ModelArchitecture::kQwen3_5:
            return "qwen3_5";
        case ModelArchitecture::kUnknown:
            break;
    }
    return "unknown";
}

const char* ModelArchitectureDisplayName(const ModelArchitecture architecture) {
    switch (architecture) {
        case ModelArchitecture::kQwen3:
            return "Qwen3";
        case ModelArchitecture::kQwen3_5:
            return "Qwen3.5";
        case ModelArchitecture::kUnknown:
            break;
    }
    return "Unknown";
}

bool ResolveModelSelection(const ManifestData& manifest,
                           const std::string& requested_model_type,
                           ModelSelection* selection,
                           std::string* error_message) {
    if (selection == nullptr) {
        if (error_message != nullptr) {
            *error_message = "ResolveModelSelection requires a non-null selection output";
        }
        return false;
    }

    if (!requested_model_type.empty()) {
        if (requested_model_type == "qwen3") {
            *selection = MakeSelection(ModelArchitecture::kQwen3, true);
            return true;
        }
        if (requested_model_type == "qwen3_5") {
            *selection = MakeSelection(ModelArchitecture::kQwen3_5, true);
            return true;
        }
        if (error_message != nullptr) {
            *error_message = "unsupported --model-type: " + requested_model_type;
        }
        return false;
    }

    if (soc::gpu::models::qwen3_5::IsQwen3_5Manifest(manifest)) {
        *selection = MakeSelection(ModelArchitecture::kQwen3_5, false);
        return true;
    }
    if (soc::gpu::models::qwen3::IsQwen3Manifest(manifest)) {
        *selection = MakeSelection(ModelArchitecture::kQwen3, false);
        return true;
    }

    if (error_message != nullptr) {
        *error_message = "failed to resolve model type from manifest; pass --model-type explicitly";
    }
    return false;
}

}  // namespace soc::gpu
