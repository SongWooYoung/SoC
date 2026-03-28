#include <cstdlib>

#include <iostream>
#include <string>

#include "runtime/bootstrap_runtime.h"

namespace {

bool ParseArgs(int argc, char** argv, soc::gpu::BootstrapOptions* options, std::string* error_message) {
    for (int index = 1; index < argc; ++index) {
        std::string argument = argv[index];
        if (argument == "--metallib") {
            if (index + 1 >= argc) {
                *error_message = "missing value for --metallib";
                return false;
            }
            options->metallib_path = argv[++index];
            continue;
        }
        if (argument == "--input") {
            if (index + 1 >= argc) {
                *error_message = "missing value for --input";
                return false;
            }
            options->input_value = static_cast<std::uint32_t>(std::strtoul(argv[++index], nullptr, 10));
            continue;
        }
        if (argument == "--shader-source") {
            if (index + 1 >= argc) {
                *error_message = "missing value for --shader-source";
                return false;
            }
            options->shader_source_path = argv[++index];
            continue;
        }

        *error_message = "unknown argument: " + argument;
        return false;
    }

    return true;
}

}  // namespace

int main(int argc, char** argv) {
    soc::gpu::BootstrapOptions options;
    std::string error_message;
    if (!ParseArgs(argc, argv, &options, &error_message)) {
        std::cerr << "argument error: " << error_message << '\n';
        return 1;
    }

    soc::gpu::BootstrapReport report;
    if (!soc::gpu::RunMetalBootstrap(options, &report, &error_message)) {
        std::cerr << "bootstrap failed: " << error_message << '\n';
        return 1;
    }

    std::cout << "Metal bootstrap succeeded\n";
    std::cout << "device_name=" << report.device_name << '\n';
    std::cout << "is_apple_silicon_gpu=" << (report.is_apple_silicon_gpu ? "true" : "false") << '\n';
    std::cout << "has_unified_memory=" << (report.has_unified_memory ? "true" : "false") << '\n';
    std::cout << "supports_simdgroup_matrix=" << (report.supports_simdgroup_matrix ? "true" : "false") << '\n';
    std::cout << "recommended_max_working_set_size=" << report.recommended_max_working_set_size << '\n';
    std::cout << "max_threads_per_threadgroup=" << report.max_threads_per_threadgroup << '\n';
    std::cout << "thread_execution_width=" << report.thread_execution_width << '\n';
    std::cout << "input_value=" << report.input_value << '\n';
    std::cout << "output_value=" << report.output_value << '\n';
    return 0;
}