#pragma once

#include <cstdint>
#include <string>

namespace soc::gpu {

struct BootstrapOptions {
    std::string metallib_path = "build/shaders/gpu.metallib";
    std::string shader_source_path = "shaders/gpu_kernels.metal";
    std::uint32_t input_value = 7;
};

struct BootstrapReport {
    std::string device_name;
    bool is_apple_silicon_gpu = false;
    bool has_unified_memory = false;
    bool supports_simdgroup_matrix = false;
    std::uint64_t recommended_max_working_set_size = 0;
    std::uint32_t max_threads_per_threadgroup = 0;
    std::uint32_t thread_execution_width = 0;
    std::uint32_t input_value = 0;
    std::uint32_t output_value = 0;
};

bool RunMetalBootstrap(const BootstrapOptions& options,
                       BootstrapReport* report,
                       std::string* error_message);

}  // namespace soc::gpu