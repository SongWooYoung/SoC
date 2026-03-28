#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace soc::gpu {

struct MetalDeviceInfo {
    std::string name;
    bool is_apple_silicon_gpu = false;
    bool has_unified_memory = false;
    bool supports_simdgroup_matrix = false;
    std::uint64_t recommended_max_working_set_size = 0;
    std::uint32_t max_threads_per_threadgroup = 0;
    std::uint32_t thread_execution_width = 0;
};

struct MetalProfilingSnapshot {
    double gpu_ms = 0.0;
    std::size_t command_buffer_count = 0;
};

class MetalContext {
public:
    static std::unique_ptr<MetalContext> CreateDefault(const std::string& metallib_path,
                                                       const std::string& shader_source_path,
                                                       std::string* error_message);

    ~MetalContext();

    const MetalDeviceInfo& GetDeviceInfo() const;
    const void* GetNativeDevice() const;
    const void* GetNativeCommandQueue() const;
    const void* GetNativeLibrary() const;
    void ResetProfiling() const;
    MetalProfilingSnapshot GetProfilingSnapshot() const;
    bool FinalizeCommandBuffer(const void* command_buffer_handle,
                               const std::string& error_prefix,
                               std::string* error_message) const;
    bool RunBootstrapKernel(std::uint32_t input_value,
                            std::uint32_t* output_value,
                            std::string* error_message) const;

private:
    struct Impl;

    explicit MetalContext(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

}  // namespace soc::gpu