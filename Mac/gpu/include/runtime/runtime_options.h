#pragma once

#include <cstddef>
#include <string>

#include "metal/metal_context.h"

struct ManifestData;
struct RuntimeGenerationOptions;
struct RuntimePromptOptions;
struct SamplerConfig;

namespace soc::gpu {

struct RuntimeSamplerOptions {
    float temperature = 0.0f;
    std::size_t top_k = 0;
};

struct RuntimeGenerationOptions {
    std::size_t max_new_tokens = 32;
    RuntimeSamplerOptions sampler;
    int eos_token_id = -1;
    std::size_t max_sequence_length = 256;
};

struct RuntimePromptOptions {
    bool apply_chat_template = false;
    bool add_generation_prompt = true;
    bool enable_thinking = true;
    std::string system_prompt;
};

struct InferCliOptions {
    std::string manifest_path = "../../models/cpp/qwen3-0.6b/manifest.json";
    std::string model_type;
    std::string metallib_path = "build/shaders/gpu.metallib";
    std::string shader_source_path = "shaders/gpu_kernels.metal";
    std::string prompt;
    std::string prompt_file;
    std::string output_file;
    std::string prompt_cache_artifact_save;
    std::string prompt_cache_artifact_load;
    std::string qwen3_5_boundary_probe_output;
    RuntimeGenerationOptions generation;
    RuntimePromptOptions prompt_options;
    std::size_t temporary_arena_bytes = 1ull << 26;
    bool override_max_new_tokens = false;
    bool override_temperature = false;
    bool override_top_k = false;
    bool override_eos_token_id = false;
    bool override_max_sequence_length = false;
    bool override_enable_thinking = false;
    bool json_output = false;
    bool validate_only = false;
    int layer = -1;
    bool verbose = false;
    MetalProfilingMode profiling_mode = MetalProfilingMode::kSummary;
};

RuntimeGenerationOptions ResolveGenerationOptions(const InferCliOptions& cli, const ::ManifestData& manifest);
::RuntimePromptOptions ToCpuRuntimePromptOptions(const RuntimePromptOptions& options);
::RuntimeGenerationOptions ToCpuRuntimeGenerationOptions(const RuntimeGenerationOptions& options);
::SamplerConfig ToCpuSamplerConfig(const RuntimeSamplerOptions& options);

}  // namespace soc::gpu
