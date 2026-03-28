#include <cstdlib>

#include <iostream>
#include <string>

#include "runtime/bootstrap_runtime.h"

int main() {
    soc::gpu::BootstrapOptions options;
    options.input_value = 41;

    const char* metallib_env = std::getenv("SOC_GPU_METALLIB");
    if (metallib_env != nullptr && metallib_env[0] != '\0') {
        options.metallib_path = metallib_env;
    }

    soc::gpu::BootstrapReport report;
    std::string error_message;
    if (!soc::gpu::RunMetalBootstrap(options, &report, &error_message)) {
        std::cerr << "bootstrap runtime failed: " << error_message << '\n';
        return 1;
    }

    if (report.output_value != options.input_value + 1) {
        std::cerr << "unexpected shader output: " << report.output_value << '\n';
        return 1;
    }

    if (report.device_name.empty()) {
        std::cerr << "device name must not be empty\n";
        return 1;
    }

    std::cout << "test_metal_bootstrap passed\n";
    return 0;
}