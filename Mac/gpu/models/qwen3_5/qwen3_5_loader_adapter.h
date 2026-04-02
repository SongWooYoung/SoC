#pragma once

#include <string>

#include "asset/runtime_assets.h"
#include "models/qwen3_5/qwen3_5_architecture.h"
#include "models/qwen3_5/qwen3_5_manifest_metadata.h"
#include "models/qwen3_5/qwen3_5_runner.h"

namespace soc::gpu {
class MetalContext;
}

namespace soc::gpu::models::qwen3_5 {

bool ResolveArchitectureSpec(const ManifestData& manifest,
                             Qwen3_5ArchitectureSpec* spec,
                             std::string* error_message);

bool ResolveArchitectureSpecFromFile(const std::string& manifest_path,
                                     Qwen3_5ArchitectureSpec* spec,
                                     std::string* error_message);

bool ResolveManifestMetadataFromFile(const std::string& manifest_path,
                                     const Qwen3_5ArchitectureSpec& spec,
                                     Qwen3_5ManifestMetadata* metadata,
                                     std::string* error_message);

bool LoadGpuModel(const MetalContext& context,
                  const ManifestData& manifest,
                  Qwen3_5Runner* runner,
                  std::string* error_message);

bool LoadGpuModelFromFile(const MetalContext& context,
                          const std::string& manifest_path,
                          Qwen3_5Runner* runner,
                          std::string* error_message);

}  // namespace soc::gpu::models::qwen3_5
