#pragma once

#include <string>

#include "asset/runtime_assets.h"

namespace soc::gpu {

enum class ModelArchitecture {
    kUnknown,
    kQwen3,
    kQwen3_5,
};

struct ModelSelection {
    ModelArchitecture architecture = ModelArchitecture::kUnknown;
    std::string registry_name;
    std::string display_name;
    bool resolved_from_cli = false;
};

const char* ModelArchitectureName(ModelArchitecture architecture);
const char* ModelArchitectureDisplayName(ModelArchitecture architecture);

bool ResolveModelSelection(const ManifestData& manifest,
                           const std::string& requested_model_type,
                           ModelSelection* selection,
                           std::string* error_message);

}  // namespace soc::gpu
