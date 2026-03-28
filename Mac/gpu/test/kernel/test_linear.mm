#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "buffer/buffer_arena.h"
#include "buffer/metal_buffer.h"
#include "kernel/pipeline_cache.h"
#include "metal/metal_context.h"
#include "op/linear_op.h"
#include "op/matmul_op.h"
#include "op/rms_norm_op.h"
#include "tensor/device_tensor.h"
#include "tensor/tensor_desc.h"

namespace {

bool NearlyEqual(float lhs, float rhs) {
    return std::fabs(lhs - rhs) < 1.0e-4f;
}

std::vector<float> ComputeLinearReference(const std::vector<float>& input,
                                          const std::vector<float>& weight,
                                          const std::vector<float>& bias,
                                          std::size_t rows,
                                          std::size_t inner_dim,
                                          std::size_t cols) {
    std::vector<float> output(rows * cols, 0.0f);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            float accumulator = bias.empty() ? 0.0f : bias[col];
            for (std::size_t inner = 0; inner < inner_dim; ++inner) {
                accumulator += input[row * inner_dim + inner] * weight[inner * cols + col];
            }
            output[row * cols + col] = accumulator;
        }
    }
    return output;
}

float SiLU(float value) {
    return value / (1.0f + std::exp(-value));
}

std::vector<float> ApplyLinearPostprocessReference(const std::vector<float>& input,
                                                   const std::vector<float>& residual,
                                                   bool add_residual,
                                                   bool enable_silu) {
    std::vector<float> output = input;
    for (std::size_t index = 0; index < output.size(); ++index) {
        if (add_residual) {
            output[index] += residual[index];
        }
        if (enable_silu) {
            output[index] = SiLU(output[index]);
        }
    }
    return output;
}

std::vector<float> ComputeRmsNormReference(const std::vector<float>& input,
                                           const std::vector<float>& weight,
                                           std::size_t rows,
                                           std::size_t row_size,
                                           float epsilon) {
    std::vector<float> output(input.size(), 0.0f);
    for (std::size_t row = 0; row < rows; ++row) {
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

bool CheckVector(const std::vector<float>& actual, const std::vector<float>& expected, const std::string& label) {
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!NearlyEqual(actual[index], expected[index])) {
            std::cerr << label << " mismatch at index " << index << ": actual=" << actual[index]
                      << " expected=" << expected[index] << '\n';
            return false;
        }
    }
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

    auto temporary_arena = soc::gpu::BufferArena::CreateShared(*context, 4096, "temp", &error_message);
    if (temporary_arena == nullptr) {
        std::cerr << "failed to create temporary arena: " << error_message << '\n';
        return 1;
    }
    temporary_arena->ClearTelemetry();

    auto input_buffer = soc::gpu::MetalBuffer::CreateShared(*context, sizeof(float) * 8, "input", &error_message);
    auto weight_buffer = soc::gpu::MetalBuffer::CreateShared(*context, sizeof(float) * 12, "weight", &error_message);
    auto bias_buffer = soc::gpu::MetalBuffer::CreateShared(*context, sizeof(float) * 3, "bias", &error_message);
    auto linear_output_buffer = soc::gpu::MetalBuffer::CreateShared(*context, sizeof(float) * 6, "linear_out", &error_message);
    auto norm_weight_buffer = soc::gpu::MetalBuffer::CreateShared(*context, sizeof(float) * 3, "norm_weight", &error_message);
    auto norm_output_buffer = soc::gpu::MetalBuffer::CreateShared(*context, sizeof(float) * 6, "norm_out", &error_message);
    auto residual_buffer = soc::gpu::MetalBuffer::CreateShared(*context, sizeof(float) * 6, "residual", &error_message);
    auto decode_input_buffer = soc::gpu::MetalBuffer::CreateShared(*context, sizeof(float) * 4, "decode_input", &error_message);
    auto decode_output_buffer = soc::gpu::MetalBuffer::CreateShared(*context, sizeof(float) * 3, "decode_out", &error_message);
    if (input_buffer == nullptr || weight_buffer == nullptr || bias_buffer == nullptr ||
        linear_output_buffer == nullptr || norm_weight_buffer == nullptr || norm_output_buffer == nullptr || residual_buffer == nullptr ||
        decode_input_buffer == nullptr || decode_output_buffer == nullptr) {
        std::cerr << "buffer allocation failed: " << error_message << '\n';
        return 1;
    }

    const std::vector<float> input = {1.0f, 2.0f, -1.0f, 0.5f, -2.0f, 1.5f, 3.0f, 0.25f};
    const std::vector<float> weight = {0.5f, -1.0f, 2.0f, 1.0f, 0.25f, -0.5f, -1.5f, 0.75f, 1.25f, 2.0f, -0.25f, 0.5f};
    const std::vector<float> bias = {0.1f, -0.2f, 0.3f};
    const std::vector<float> norm_weight = {1.0f, 0.5f, 1.5f};
    const std::vector<float> residual = {0.2f, -0.3f, 0.4f, -0.5f, 0.6f, -0.7f};
    const std::vector<float> decode_input = {1.0f, -1.0f, 0.5f, 2.0f};

    if (!input_buffer->Write(input.data(), sizeof(float) * input.size(), 0, &error_message) ||
        !weight_buffer->Write(weight.data(), sizeof(float) * weight.size(), 0, &error_message) ||
        !bias_buffer->Write(bias.data(), sizeof(float) * bias.size(), 0, &error_message) ||
        !norm_weight_buffer->Write(norm_weight.data(), sizeof(float) * norm_weight.size(), 0, &error_message) ||
        !residual_buffer->Write(residual.data(), sizeof(float) * residual.size(), 0, &error_message) ||
        !decode_input_buffer->Write(decode_input.data(), sizeof(float) * decode_input.size(), 0, &error_message)) {
        std::cerr << "buffer upload failed: " << error_message << '\n';
        return 1;
    }

    const soc::gpu::TensorDesc input_desc = soc::gpu::TensorDesc::CreateContiguous(soc::gpu::DataType::kFloat32, {2, 4});
    const soc::gpu::TensorDesc weight_desc = soc::gpu::TensorDesc::CreateContiguous(soc::gpu::DataType::kFloat32, {4, 3});
    const soc::gpu::TensorDesc bias_desc = soc::gpu::TensorDesc::CreateContiguous(soc::gpu::DataType::kFloat32, {3});
    const soc::gpu::TensorDesc output_desc = soc::gpu::TensorDesc::CreateContiguous(soc::gpu::DataType::kFloat32, {2, 3});
    const soc::gpu::TensorDesc decode_input_desc = soc::gpu::TensorDesc::CreateContiguous(soc::gpu::DataType::kFloat32, {1, 4});
    const soc::gpu::TensorDesc decode_output_desc = soc::gpu::TensorDesc::CreateContiguous(soc::gpu::DataType::kFloat32, {1, 3});

    const soc::gpu::DeviceTensor input_tensor(input_buffer, 0, input_desc);
    const soc::gpu::DeviceTensor weight_tensor(weight_buffer, 0, weight_desc);
    const soc::gpu::DeviceTensor bias_tensor(bias_buffer, 0, bias_desc);
    const soc::gpu::DeviceTensor linear_output_tensor(linear_output_buffer, 0, output_desc);
    const soc::gpu::DeviceTensor norm_weight_tensor(norm_weight_buffer, 0, bias_desc);
    const soc::gpu::DeviceTensor norm_output_tensor(norm_output_buffer, 0, output_desc);
    const soc::gpu::DeviceTensor residual_tensor(residual_buffer, 0, output_desc);
    const soc::gpu::DeviceTensor decode_input_tensor(decode_input_buffer, 0, decode_input_desc);
    const soc::gpu::DeviceTensor decode_output_tensor(decode_output_buffer, 0, decode_output_desc);

    soc::gpu::BufferArenaSlice manual_slice_a;
    soc::gpu::BufferArenaSlice manual_slice_b;
    temporary_arena->Reset();
    const soc::gpu::BufferArenaMark base_mark = temporary_arena->GetMark();
    if (!temporary_arena->Allocate(128, 64, &manual_slice_a, &error_message) ||
        !temporary_arena->Allocate(256, 64, &manual_slice_b, &error_message)) {
        std::cerr << "manual arena allocation failed: " << error_message << '\n';
        return 1;
    }
    if (manual_slice_b.offset_bytes <= manual_slice_a.offset_bytes) {
        std::cerr << "arena allocation offsets are not increasing\n";
        return 1;
    }
    if (!temporary_arena->ResetToMark(base_mark, &error_message)) {
        std::cerr << "reset-to-mark failed: " << error_message << '\n';
        return 1;
    }
    if (temporary_arena->GetUsedBytes() != base_mark) {
        std::cerr << "arena usage did not return to the requested mark\n";
        return 1;
    }

    soc::gpu::PipelineCache pipeline_cache(*context);
    soc::gpu::LinearParams linear_params;
    linear_params.matmul.use_bias = true;
    linear_params.activation = soc::gpu::LinearActivation::kSiLU;
    linear_params.add_residual = true;
    temporary_arena->Reset();
    if (!soc::gpu::LinearOp::Run(*context,
                                 &pipeline_cache,
                                 input_tensor,
                                 weight_tensor,
                                 &bias_tensor,
                                 &residual_tensor,
                                 linear_output_tensor,
                                 linear_params,
                                 temporary_arena.get(),
                                 &error_message)) {
        std::cerr << "Linear dispatch failed: " << error_message << '\n';
        return 1;
    }
    if (temporary_arena->GetUsedBytes() != 0) {
        std::cerr << "Linear op should restore temporary arena usage to the incoming mark\n";
        return 1;
    }
    if (temporary_arena->GetPeakUsedBytes() == 0) {
        std::cerr << "Linear op never touched temporary arena scratch\n";
        return 1;
    }

    std::vector<float> linear_output(6, 0.0f);
    if (!linear_output_buffer->Read(linear_output.data(), sizeof(float) * linear_output.size(), 0, &error_message)) {
        std::cerr << "linear output readback failed: " << error_message << '\n';
        return 1;
    }
    const std::vector<float> linear_reference =
        ApplyLinearPostprocessReference(ComputeLinearReference(input, weight, bias, 2, 4, 3),
                                        residual,
                                        true,
                                        true);
    if (!CheckVector(linear_output, linear_reference, "linear")) {
        return 1;
    }

    temporary_arena->Reset();
    if (temporary_arena->GetUsedBytes() != 0) {
        std::cerr << "temporary arena did not reset to zero\n";
        return 1;
    }

    soc::gpu::RmsNormParams rms_norm_params;
    if (!soc::gpu::RmsNormOp::Run(*context,
                                  &pipeline_cache,
                                  linear_output_tensor,
                                  norm_weight_tensor,
                                  norm_output_tensor,
                                  rms_norm_params,
                                  temporary_arena.get(),
                                  &error_message)) {
        std::cerr << "RMSNorm after Linear failed: " << error_message << '\n';
        return 1;
    }
    if (temporary_arena->GetUsedBytes() != 0) {
        std::cerr << "RMSNorm should release temporary arena usage back to the mark\n";
        return 1;
    }

    std::vector<float> norm_output(6, 0.0f);
    if (!norm_output_buffer->Read(norm_output.data(), sizeof(float) * norm_output.size(), 0, &error_message)) {
        std::cerr << "norm output readback failed: " << error_message << '\n';
        return 1;
    }
    const std::vector<float> norm_reference = ComputeRmsNormReference(linear_reference, norm_weight, 2, 3, rms_norm_params.epsilon);
    if (!CheckVector(norm_output, norm_reference, "linear+rmsnorm")) {
        return 1;
    }

    temporary_arena->Reset();
    soc::gpu::LinearParams decode_params;
    decode_params.matmul.row_count = 1;
    decode_params.matmul.inner_dim = 4;
    decode_params.matmul.column_count = 3;
    decode_params.matmul.use_bias = true;
    decode_params.matmul.decode_mode = true;
    if (!soc::gpu::LinearOp::Run(*context,
                                 &pipeline_cache,
                                 decode_input_tensor,
                                 weight_tensor,
                                 &bias_tensor,
                                 nullptr,
                                 decode_output_tensor,
                                 decode_params,
                                 temporary_arena.get(),
                                 &error_message)) {
        std::cerr << "decode Linear dispatch failed: " << error_message << '\n';
        return 1;
    }

    const soc::gpu::MatMulExecutionPolicy regular_policy = soc::gpu::MatMulOp::SelectPolicy(linear_params.matmul, 2, 3, 4);
    const soc::gpu::MatMulExecutionPolicy decode_policy = soc::gpu::MatMulOp::SelectPolicy(decode_params.matmul, 1, 3, 4);
    const soc::gpu::KernelKey regular_key{soc::gpu::KernelKind::kMatMul,
                                          regular_policy.use_tiled_kernel ? "matmul_f32_tiled" : "matmul_f32_basic",
                                          regular_policy.threadgroup_width,
                                          regular_policy.threadgroup_height,
                                          {true,
                                           false,
                                                                                     false,
                                           true,
                                           true,
                                           regular_policy.tile_columns,
                                           regular_policy.tile_rows}};
    const soc::gpu::KernelKey decode_key{soc::gpu::KernelKind::kMatMul,
                                                                                 decode_policy.use_tiled_kernel ? "matmul_f32_decode_tiled" : "matmul_f32_basic",
                                         decode_policy.threadgroup_width,
                                         decode_policy.threadgroup_height,
                                         {true,
                                          true,
                                                                                    false,
                                          false,
                                          false,
                                          decode_policy.tile_columns,
                                          decode_policy.tile_rows}};
    if (!pipeline_cache.HasPipeline(regular_key) || !pipeline_cache.HasPipeline(decode_key)) {
        std::cerr << "pipeline cache did not retain fused regular/decode matmul pipelines\n";
        return 1;
    }

    bool found_matmul_scope = false;
    for (const auto& entry : temporary_arena->GetTelemetrySnapshot()) {
        if (entry.name == "MatMulOp") {
            found_matmul_scope = entry.invocation_count >= 2 && entry.peak_delta_bytes > 0 && entry.peak_allocation_count > 0;
        }
    }
    if (!found_matmul_scope) {
        std::cerr << "missing MatMul arena telemetry\n";
        return 1;
    }

    std::vector<float> decode_output(3, 0.0f);
    if (!decode_output_buffer->Read(decode_output.data(), sizeof(float) * decode_output.size(), 0, &error_message)) {
        std::cerr << "decode output readback failed: " << error_message << '\n';
        return 1;
    }
    const std::vector<float> decode_reference = ComputeLinearReference(decode_input, weight, bias, 1, 4, 3);
    if (!CheckVector(decode_output, decode_reference, "decode linear")) {
        return 1;
    }

    std::cout << "test_linear passed\n";
    return 0;
}