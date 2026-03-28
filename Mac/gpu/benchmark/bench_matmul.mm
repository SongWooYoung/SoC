#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "buffer/buffer_arena.h"
#include "buffer/metal_buffer.h"
#include "kernel/pipeline_cache.h"
#include "metal/metal_context.h"
#include "op/linear_op.h"
#include "tensor/device_tensor.h"
#include "tensor/tensor_desc.h"

namespace {

using Clock = std::chrono::steady_clock;

struct PolicyCandidate {
    std::uint32_t threadgroup_width;
    std::uint32_t threadgroup_height;
};

struct BenchmarkScenario {
    std::string name;
    std::size_t rows;
    std::size_t inner_dim;
    std::size_t cols;
    bool decode_mode;
    bool use_bias;
    bool add_residual;
    bool enable_silu;
    int warmup_iterations;
    int timed_iterations;
    std::vector<PolicyCandidate> candidates;
};

struct BenchmarkResult {
    PolicyCandidate candidate;
    double average_milliseconds = 0.0;
    double gflops = 0.0;
    float max_abs_error = 0.0f;
};

float ComputeValue(std::size_t index, float scale) {
    const float centered = static_cast<float>((index % 29) - 14);
    return centered * scale;
}

std::vector<float> MakeData(std::size_t count, float scale) {
    std::vector<float> values(count, 0.0f);
    for (std::size_t index = 0; index < count; ++index) {
        values[index] = ComputeValue(index, scale);
    }
    return values;
}

std::vector<float> ComputeReference(const BenchmarkScenario& scenario,
                                    const std::vector<float>& lhs,
                                    const std::vector<float>& rhs,
                                    const std::vector<float>& bias,
                                    const std::vector<float>& residual) {
    std::vector<float> output(scenario.rows * scenario.cols, 0.0f);
    for (std::size_t row = 0; row < scenario.rows; ++row) {
        for (std::size_t col = 0; col < scenario.cols; ++col) {
            float accumulator = scenario.use_bias ? bias[col] : 0.0f;
            for (std::size_t inner = 0; inner < scenario.inner_dim; ++inner) {
                accumulator += lhs[row * scenario.inner_dim + inner] * rhs[inner * scenario.cols + col];
            }
            if (scenario.add_residual) {
                accumulator += residual[row * scenario.cols + col];
            }
            if (scenario.enable_silu) {
                accumulator = accumulator / (1.0f + std::exp(-accumulator));
            }
            output[row * scenario.cols + col] = accumulator;
        }
    }
    return output;
}

float ComputeMaxAbsError(const std::vector<float>& actual, const std::vector<float>& expected) {
    float max_abs_error = 0.0f;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        max_abs_error = std::max(max_abs_error, std::fabs(actual[index] - expected[index]));
    }
    return max_abs_error;
}

bool Upload(const std::shared_ptr<soc::gpu::MetalBuffer>& buffer,
            const std::vector<float>& values,
            std::string* error_message) {
    return buffer->Write(values.data(), sizeof(float) * values.size(), 0, error_message);
}

bool RunScenario(const BenchmarkScenario& scenario,
                 const soc::gpu::MetalContext& context,
                 soc::gpu::PipelineCache* pipeline_cache,
                 soc::gpu::BufferArena* arena,
                 std::string* error_message) {
    const auto lhs = MakeData(scenario.rows * scenario.inner_dim, 1.0f / 37.0f);
    const auto rhs = MakeData(scenario.inner_dim * scenario.cols, 1.0f / 53.0f);
    const auto bias = MakeData(scenario.cols, 1.0f / 71.0f);
    const auto residual = MakeData(scenario.rows * scenario.cols, 1.0f / 61.0f);
    const auto reference = ComputeReference(scenario, lhs, rhs, bias, residual);

    auto lhs_buffer = soc::gpu::MetalBuffer::CreateShared(context, sizeof(float) * lhs.size(), "bench_lhs", error_message);
    auto rhs_buffer = soc::gpu::MetalBuffer::CreateShared(context, sizeof(float) * rhs.size(), "bench_rhs", error_message);
    auto bias_buffer = soc::gpu::MetalBuffer::CreateShared(context, sizeof(float) * bias.size(), "bench_bias", error_message);
    auto residual_buffer = soc::gpu::MetalBuffer::CreateShared(context, sizeof(float) * residual.size(), "bench_residual", error_message);
    auto output_buffer = soc::gpu::MetalBuffer::CreateShared(context, sizeof(float) * reference.size(), "bench_output", error_message);
    if (lhs_buffer == nullptr || rhs_buffer == nullptr || bias_buffer == nullptr || residual_buffer == nullptr || output_buffer == nullptr) {
        return false;
    }

    if (!Upload(lhs_buffer, lhs, error_message) || !Upload(rhs_buffer, rhs, error_message) ||
        !Upload(bias_buffer, bias, error_message) || !Upload(residual_buffer, residual, error_message)) {
        return false;
    }

    const soc::gpu::TensorDesc lhs_desc =
        soc::gpu::TensorDesc::CreateContiguous(soc::gpu::DataType::kFloat32, {scenario.rows, scenario.inner_dim});
    const soc::gpu::TensorDesc rhs_desc =
        soc::gpu::TensorDesc::CreateContiguous(soc::gpu::DataType::kFloat32, {scenario.inner_dim, scenario.cols});
    const soc::gpu::TensorDesc bias_desc =
        soc::gpu::TensorDesc::CreateContiguous(soc::gpu::DataType::kFloat32, {scenario.cols});
    const soc::gpu::TensorDesc output_desc =
        soc::gpu::TensorDesc::CreateContiguous(soc::gpu::DataType::kFloat32, {scenario.rows, scenario.cols});

    const soc::gpu::DeviceTensor lhs_tensor(lhs_buffer, 0, lhs_desc);
    const soc::gpu::DeviceTensor rhs_tensor(rhs_buffer, 0, rhs_desc);
    const soc::gpu::DeviceTensor bias_tensor(bias_buffer, 0, bias_desc);
    const soc::gpu::DeviceTensor residual_tensor(residual_buffer, 0, output_desc);
    const soc::gpu::DeviceTensor output_tensor(output_buffer, 0, output_desc);

    std::vector<float> output(reference.size(), 0.0f);
    std::vector<BenchmarkResult> results;
    results.reserve(scenario.candidates.size());

    for (const PolicyCandidate& candidate : scenario.candidates) {
        soc::gpu::LinearParams params;
        params.matmul.row_count = static_cast<std::uint32_t>(scenario.rows);
        params.matmul.inner_dim = static_cast<std::uint32_t>(scenario.inner_dim);
        params.matmul.column_count = static_cast<std::uint32_t>(scenario.cols);
        params.matmul.use_bias = scenario.use_bias;
        params.matmul.decode_mode = scenario.decode_mode;
        params.matmul.enable_silu = scenario.enable_silu;
        params.matmul.add_residual = scenario.add_residual;
        params.activation = scenario.enable_silu ? soc::gpu::LinearActivation::kSiLU : soc::gpu::LinearActivation::kNone;
        params.add_residual = scenario.add_residual;
        params.matmul.preferred_threadgroup_width = candidate.threadgroup_width;
        params.matmul.preferred_threadgroup_height = candidate.threadgroup_height;
        params.matmul.preferred_tile_columns = candidate.threadgroup_width;
        params.matmul.preferred_tile_rows = candidate.threadgroup_height;

        for (int iteration = 0; iteration < scenario.warmup_iterations; ++iteration) {
            arena->Reset();
            if (!soc::gpu::LinearOp::Run(context,
                                         pipeline_cache,
                                         lhs_tensor,
                                         rhs_tensor,
                                         scenario.use_bias ? &bias_tensor : nullptr,
                                         scenario.add_residual ? &residual_tensor : nullptr,
                                         output_tensor,
                                         params,
                                         arena,
                                         error_message)) {
                return false;
            }
        }

        const auto start = Clock::now();
        for (int iteration = 0; iteration < scenario.timed_iterations; ++iteration) {
            arena->Reset();
            if (!soc::gpu::LinearOp::Run(context,
                                         pipeline_cache,
                                         lhs_tensor,
                                         rhs_tensor,
                                         scenario.use_bias ? &bias_tensor : nullptr,
                                         scenario.add_residual ? &residual_tensor : nullptr,
                                         output_tensor,
                                         params,
                                         arena,
                                         error_message)) {
                return false;
            }
        }
        const auto end = Clock::now();

        if (!output_buffer->Read(output.data(), sizeof(float) * output.size(), 0, error_message)) {
            return false;
        }

        const double total_ms =
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
        const double average_ms = total_ms / static_cast<double>(scenario.timed_iterations);
        const double flops = 2.0 * static_cast<double>(scenario.rows) * static_cast<double>(scenario.cols) *
                             static_cast<double>(scenario.inner_dim);
        const double gflops = flops / (average_ms * 1.0e6);
        results.push_back(BenchmarkResult{candidate, average_ms, gflops, ComputeMaxAbsError(output, reference)});
    }

    std::cout << "scenario=" << scenario.name << " rows=" << scenario.rows << " inner=" << scenario.inner_dim
              << " cols=" << scenario.cols << " decode=" << (scenario.decode_mode ? "true" : "false")
              << " fused_bias=" << (scenario.use_bias ? "true" : "false")
              << " fused_residual=" << (scenario.add_residual ? "true" : "false")
              << " fused_silu=" << (scenario.enable_silu ? "true" : "false") << '\n';
    std::cout << "tg_width,tg_height,avg_ms,gflops,max_abs_error\n";
    for (const BenchmarkResult& result : results) {
        std::cout << result.candidate.threadgroup_width << ',' << result.candidate.threadgroup_height << ','
                  << std::fixed << std::setprecision(4) << result.average_milliseconds << ',' << std::setprecision(3)
                  << result.gflops << ',' << std::setprecision(6) << result.max_abs_error << '\n';
    }
    std::cout << '\n';
    return true;
}

}  // namespace

int main() {
    std::string error_message;
    auto context = soc::gpu::MetalContext::CreateDefault("build/shaders/gpu.metallib",
                                                         "shaders/gpu_kernels.metal",
                                                         &error_message);
    if (context == nullptr) {
        std::cerr << "failed to create Metal context: " << error_message << '\n';
        return 1;
    }

    soc::gpu::PipelineCache pipeline_cache(*context);
    auto arena = soc::gpu::BufferArena::CreateShared(*context, 1 << 20, "bench_temp", &error_message);
    if (arena == nullptr) {
        std::cerr << "failed to create benchmark arena: " << error_message << '\n';
        return 1;
    }

    const std::vector<BenchmarkScenario> scenarios = {
        {"decode_1x512x512", 1, 512, 512, true, true, false, false, 5, 20, {{4, 1}, {8, 1}, {16, 1}, {32, 1}}},
        {"prefill_8x512x512", 8, 512, 512, false, true, false, false, 5, 20, {{4, 2}, {8, 2}, {8, 4}, {16, 4}}},
        {"fused_linear_32x256x1024", 32, 256, 1024, false, true, true, true, 3, 10, {{4, 2}, {8, 2}, {8, 4}, {16, 4}}},
    };

    for (const BenchmarkScenario& scenario : scenarios) {
        if (!RunScenario(scenario, *context, &pipeline_cache, arena.get(), &error_message)) {
            std::cerr << "benchmark failed for " << scenario.name << ": " << error_message << '\n';
            return 1;
        }
    }

    return 0;
}