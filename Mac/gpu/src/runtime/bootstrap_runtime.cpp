#include "runtime/bootstrap_runtime.h"

#include "metal/metal_context.h"

namespace soc::gpu {

bool RunMetalBootstrap(const BootstrapOptions& options,
                       BootstrapReport* report,
                       std::string* error_message) {
    if (report == nullptr) {
        if (error_message != nullptr) {
            *error_message = "Bootstrap report must not be null";
        }
        return false;
    }

    auto context = MetalContext::CreateDefault(options.metallib_path, options.shader_source_path, error_message);
    if (context == nullptr) {
        return false;
    }

    std::uint32_t output_value = 0;
    if (!context->RunBootstrapKernel(options.input_value, &output_value, error_message)) {
        return false;
    }

    const MetalDeviceInfo& info = context->GetDeviceInfo();
    report->device_name = info.name;
    report->is_apple_silicon_gpu = info.is_apple_silicon_gpu;
    report->has_unified_memory = info.has_unified_memory;
    report->supports_simdgroup_matrix = info.supports_simdgroup_matrix;
    report->recommended_max_working_set_size = info.recommended_max_working_set_size;
    report->max_threads_per_threadgroup = info.max_threads_per_threadgroup;
    report->thread_execution_width = info.thread_execution_width;
    report->input_value = options.input_value;
    report->output_value = output_value;
    return true;
}

}  // namespace soc::gpu