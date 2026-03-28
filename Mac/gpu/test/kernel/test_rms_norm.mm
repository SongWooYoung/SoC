#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "buffer/buffer_arena.h"
#include "buffer/metal_buffer.h"
#include "kernel/pipeline_cache.h"
#include "metal/metal_context.h"
#include "op/rms_norm_op.h"
#include "tensor/device_tensor.h"
#include "tensor/tensor_desc.h"

namespace {

bool NearlyEqual(float lhs, float rhs) {
    return std::fabs(lhs - rhs) < 1.0e-4f;
}

std::vector<float> ComputeReference(const std::vector<float>& input,
                                    const std::vector<float>& weight,
                                    std::size_t row_count,
                                    std::size_t row_size,
                                    float epsilon) {
    std::vector<float> output(input.size(), 0.0f);
    for (std::size_t row = 0; row < row_count; ++row) {
        float sum_squares = 0.0f;
        for (std::size_t column = 0; column < row_size; ++column) {
            const float value = input[row * row_size + column];
            sum_squares += value * value;
        }

        const float inv_rms = 1.0f / std::sqrt(sum_squares / static_cast<float>(row_size) + epsilon);
        for (std::size_t column = 0; column < row_size; ++column) {
            output[row * row_size + column] = input[row * row_size + column] * inv_rms * weight[column];
        }
    }
    return output;
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

    auto input_buffer = soc::gpu::MetalBuffer::CreateShared(*context, sizeof(float) * 8, "input", &error_message);
    auto weight_buffer = soc::gpu::MetalBuffer::CreateShared(*context, sizeof(float) * 4, "weight", &error_message);
    auto output_buffer = soc::gpu::MetalBuffer::CreateShared(*context, sizeof(float) * 8, "output", &error_message);
    auto temporary_arena = soc::gpu::BufferArena::CreateShared(*context, 4096, "temp", &error_message);
    if (input_buffer == nullptr || weight_buffer == nullptr || output_buffer == nullptr || temporary_arena == nullptr) {
        std::cerr << "buffer allocation failed: " << error_message << '\n';
        return 1;
    }
    temporary_arena->ClearTelemetry();

    const std::vector<float> input = {1.0f, -2.0f, 0.5f, 4.0f, 3.0f, 1.5f, -1.0f, 2.5f};
    const std::vector<float> weight = {1.0f, 0.5f, 1.5f, 2.0f};
    if (!input_buffer->Write(input.data(), sizeof(float) * input.size(), 0, &error_message) ||
        !weight_buffer->Write(weight.data(), sizeof(float) * weight.size(), 0, &error_message)) {
        std::cerr << "buffer upload failed: " << error_message << '\n';
        return 1;
    }

    const soc::gpu::TensorDesc matrix_desc =
        soc::gpu::TensorDesc::CreateContiguous(soc::gpu::DataType::kFloat32, {2, 4});
    const soc::gpu::TensorDesc vector_desc =
        soc::gpu::TensorDesc::CreateContiguous(soc::gpu::DataType::kFloat32, {4});

    const soc::gpu::DeviceTensor input_tensor(input_buffer, 0, matrix_desc);
    const soc::gpu::DeviceTensor weight_tensor(weight_buffer, 0, vector_desc);
    const soc::gpu::DeviceTensor output_tensor(output_buffer, 0, matrix_desc);

    if (!input_tensor.IsValid() || !weight_tensor.IsValid() || !output_tensor.IsValid()) {
        std::cerr << "device tensor validation failed\n";
        return 1;
    }

    soc::gpu::PipelineCache pipeline_cache(*context);
    const soc::gpu::KernelKey key{soc::gpu::KernelKind::kRmsNorm,
                                  "rms_norm_f32_rowwise",
                                  1,
                                  1,
                                  soc::gpu::KernelSpecializationKey{false, false, false, false, 1, 1}};
    if (!pipeline_cache.WarmPipeline(key, &error_message)) {
        std::cerr << "pipeline warmup failed: " << error_message << '\n';
        return 1;
    }
    if (!pipeline_cache.HasPipeline(key)) {
        std::cerr << "pipeline cache did not retain RMSNorm pipeline\n";
        return 1;
    }

    soc::gpu::RmsNormParams params;
    params.epsilon = 1.0e-5f;
    temporary_arena->Reset();
    if (!soc::gpu::RmsNormOp::Run(*context,
                                  &pipeline_cache,
                                  input_tensor,
                                  weight_tensor,
                                  output_tensor,
                                  params,
                                  temporary_arena.get(),
                                  &error_message)) {
        std::cerr << "RMSNorm dispatch failed: " << error_message << '\n';
        return 1;
    }
    if (temporary_arena->GetUsedBytes() != 0) {
        std::cerr << "RMSNorm should release temporary arena usage back to the mark\n";
        return 1;
    }
    if (temporary_arena->GetPeakUsedBytes() == 0) {
        std::cerr << "RMSNorm never touched temporary arena scratch\n";
        return 1;
    }
    bool found_rmsnorm_scope = false;
    for (const auto& entry : temporary_arena->GetTelemetrySnapshot()) {
        if (entry.name == "RmsNormOp") {
            found_rmsnorm_scope = entry.invocation_count >= 1 && entry.peak_delta_bytes > 0;
        }
    }
    if (!found_rmsnorm_scope) {
        std::cerr << "missing RMSNorm arena telemetry\n";
        return 1;
    }

    std::vector<float> output(8, 0.0f);
    if (!output_buffer->Read(output.data(), sizeof(float) * output.size(), 0, &error_message)) {
        std::cerr << "buffer readback failed: " << error_message << '\n';
        return 1;
    }

    const std::vector<float> reference = ComputeReference(input, weight, 2, 4, params.epsilon);
    for (std::size_t index = 0; index < output.size(); ++index) {
        if (!NearlyEqual(output[index], reference[index])) {
            std::cerr << "mismatch at index " << index << ": gpu=" << output[index]
                      << " cpu=" << reference[index] << '\n';
            return 1;
        }
    }

    std::cout << "test_rms_norm passed\n";
    return 0;
}