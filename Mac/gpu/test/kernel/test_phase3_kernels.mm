#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "buffer/buffer_arena.h"
#include "buffer/metal_buffer.h"
#include "kernel/pipeline_cache.h"
#include "metal/metal_context.h"
#include "op/embedding_op.h"
#include "op/rope_op.h"
#include "op/softmax_op.h"
#include "tensor/device_tensor.h"
#include "tensor/tensor_desc.h"

namespace {

bool NearlyEqual(float lhs, float rhs) {
    return std::fabs(lhs - rhs) < 1.0e-4f;
}

std::vector<float> ComputeEmbeddingReference(const std::vector<std::int32_t>& ids,
                                             const std::vector<float>& table,
                                             std::size_t hidden_size) {
    std::vector<float> output(ids.size() * hidden_size, 0.0f);
    for (std::size_t row = 0; row < ids.size(); ++row) {
        const std::size_t token_id = static_cast<std::size_t>(ids[row]);
        for (std::size_t col = 0; col < hidden_size; ++col) {
            output[row * hidden_size + col] = table[token_id * hidden_size + col];
        }
    }
    return output;
}

std::vector<float> ComputeSoftmaxReference(const std::vector<float>& input, std::size_t rows, std::size_t cols) {
    std::vector<float> output(input.size(), 0.0f);
    for (std::size_t row = 0; row < rows; ++row) {
        float max_value = input[row * cols];
        for (std::size_t col = 1; col < cols; ++col) {
            max_value = std::max(max_value, input[row * cols + col]);
        }
        float sum_value = 0.0f;
        for (std::size_t col = 0; col < cols; ++col) {
            output[row * cols + col] = std::exp(input[row * cols + col] - max_value);
            sum_value += output[row * cols + col];
        }
        for (std::size_t col = 0; col < cols; ++col) {
            output[row * cols + col] /= sum_value;
        }
    }
    return output;
}

std::vector<float> ComputeRopeReference(const std::vector<float>& input,
                                        std::size_t row_count,
                                        std::size_t head_count,
                                        std::size_t head_dim,
                                        std::size_t rotary_dim,
                                        std::size_t position_offset,
                                        float rope_theta) {
    std::vector<float> output = input;
    const std::size_t pair_count = rotary_dim / 2;
    for (std::size_t row = 0; row < row_count; ++row) {
        for (std::size_t head = 0; head < head_count; ++head) {
            const std::size_t base = row * head_count * head_dim + head * head_dim;
            for (std::size_t pair = 0; pair < pair_count; ++pair) {
                const float exponent = (2.0f * static_cast<float>(pair)) / static_cast<float>(rotary_dim);
                const float angle = static_cast<float>(row + position_offset) * std::pow(rope_theta, -exponent);
                const float cosine = std::cos(angle);
                const float sine = std::sin(angle);
                const float left_value = input[base + pair];
                const float right_value = input[base + pair_count + pair];
                output[base + pair] = left_value * cosine - right_value * sine;
                output[base + pair_count + pair] = left_value * sine + right_value * cosine;
            }
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
    auto context = soc::gpu::MetalContext::CreateDefault("build/shaders/gpu.metallib", "shaders/gpu_kernels.metal", &error_message);
    if (context == nullptr) {
        std::cerr << "failed to create Metal context: " << error_message << '\n';
        return 1;
    }

    soc::gpu::PipelineCache pipeline_cache(*context);
    auto arena = soc::gpu::BufferArena::CreateShared(*context, 16384, "phase3_temp", &error_message);
    if (arena == nullptr) {
        std::cerr << "failed to create temporary arena: " << error_message << '\n';
        return 1;
    }

    const std::vector<std::int32_t> token_ids = {2, 0};
    const std::vector<float> embedding_table = {
        0.0f, 1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f, 7.0f,
        8.0f, 9.0f, 10.0f, 11.0f,
    };
    const std::vector<float> rope_input = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
    };
    const std::vector<float> softmax_input = {
        1.0f, 2.0f, 3.0f,
        -1.0f, 0.0f, 1.0f,
    };

    auto id_buffer = soc::gpu::MetalBuffer::CreateShared(*context, sizeof(std::int32_t) * token_ids.size(), "ids", &error_message);
    auto embedding_buffer = soc::gpu::MetalBuffer::CreateShared(*context, sizeof(float) * embedding_table.size(), "embedding_table", &error_message);
    auto embedding_out_buffer = soc::gpu::MetalBuffer::CreateShared(*context, sizeof(float) * token_ids.size() * 4, "embedding_out", &error_message);
    auto rope_in_buffer = soc::gpu::MetalBuffer::CreateShared(*context, sizeof(float) * rope_input.size(), "rope_in", &error_message);
    auto rope_out_buffer = soc::gpu::MetalBuffer::CreateShared(*context, sizeof(float) * rope_input.size(), "rope_out", &error_message);
    auto softmax_in_buffer = soc::gpu::MetalBuffer::CreateShared(*context, sizeof(float) * softmax_input.size(), "softmax_in", &error_message);
    auto softmax_out_buffer = soc::gpu::MetalBuffer::CreateShared(*context, sizeof(float) * softmax_input.size(), "softmax_out", &error_message);
    if (id_buffer == nullptr || embedding_buffer == nullptr || embedding_out_buffer == nullptr ||
        rope_in_buffer == nullptr || rope_out_buffer == nullptr || softmax_in_buffer == nullptr || softmax_out_buffer == nullptr) {
        std::cerr << "buffer allocation failed: " << error_message << '\n';
        return 1;
    }

    if (!id_buffer->Write(token_ids.data(), sizeof(std::int32_t) * token_ids.size(), 0, &error_message) ||
        !embedding_buffer->Write(embedding_table.data(), sizeof(float) * embedding_table.size(), 0, &error_message) ||
        !rope_in_buffer->Write(rope_input.data(), sizeof(float) * rope_input.size(), 0, &error_message) ||
        !softmax_in_buffer->Write(softmax_input.data(), sizeof(float) * softmax_input.size(), 0, &error_message)) {
        std::cerr << "buffer upload failed: " << error_message << '\n';
        return 1;
    }

    const soc::gpu::DeviceTensor id_tensor(id_buffer, 0, soc::gpu::TensorDesc::CreateContiguous(soc::gpu::DataType::kInt32, {token_ids.size()}));
    const soc::gpu::DeviceTensor embedding_tensor(embedding_buffer, 0, soc::gpu::TensorDesc::CreateContiguous(soc::gpu::DataType::kFloat32, {3, 4}));
    const soc::gpu::DeviceTensor embedding_out_tensor(embedding_out_buffer, 0, soc::gpu::TensorDesc::CreateContiguous(soc::gpu::DataType::kFloat32, {token_ids.size(), 4}));
    const soc::gpu::DeviceTensor rope_in_tensor(rope_in_buffer, 0, soc::gpu::TensorDesc::CreateContiguous(soc::gpu::DataType::kFloat32, {2, 4}));
    const soc::gpu::DeviceTensor rope_out_tensor(rope_out_buffer, 0, soc::gpu::TensorDesc::CreateContiguous(soc::gpu::DataType::kFloat32, {2, 4}));
    const soc::gpu::DeviceTensor softmax_in_tensor(softmax_in_buffer, 0, soc::gpu::TensorDesc::CreateContiguous(soc::gpu::DataType::kFloat32, {2, 3}));
    const soc::gpu::DeviceTensor softmax_out_tensor(softmax_out_buffer, 0, soc::gpu::TensorDesc::CreateContiguous(soc::gpu::DataType::kFloat32, {2, 3}));

    soc::gpu::EmbeddingParams embedding_params;
    if (!soc::gpu::EmbeddingOp::Run(*context, &pipeline_cache, id_tensor, embedding_tensor, embedding_out_tensor, embedding_params, arena.get(), &error_message)) {
        std::cerr << "Embedding dispatch failed: " << error_message << '\n';
        return 1;
    }

    soc::gpu::RopeParams rope_params;
    rope_params.head_count = 1;
    rope_params.head_dim = 4;
    rope_params.rotary_dim = 4;
    rope_params.position_offset = 3;
    if (!soc::gpu::RopeOp::Run(*context, &pipeline_cache, rope_in_tensor, rope_out_tensor, rope_params, arena.get(), &error_message)) {
        std::cerr << "RoPE dispatch failed: " << error_message << '\n';
        return 1;
    }

    soc::gpu::SoftmaxParams softmax_params;
    if (!soc::gpu::SoftmaxOp::Run(*context, &pipeline_cache, softmax_in_tensor, softmax_out_tensor, softmax_params, arena.get(), &error_message)) {
        std::cerr << "Softmax dispatch failed: " << error_message << '\n';
        return 1;
    }

    std::vector<float> embedding_output(token_ids.size() * 4, 0.0f);
    std::vector<float> rope_output(rope_input.size(), 0.0f);
    std::vector<float> softmax_output(softmax_input.size(), 0.0f);
    if (!embedding_out_buffer->Read(embedding_output.data(), sizeof(float) * embedding_output.size(), 0, &error_message) ||
        !rope_out_buffer->Read(rope_output.data(), sizeof(float) * rope_output.size(), 0, &error_message) ||
        !softmax_out_buffer->Read(softmax_output.data(), sizeof(float) * softmax_output.size(), 0, &error_message)) {
        std::cerr << "buffer readback failed: " << error_message << '\n';
        return 1;
    }

    if (!CheckVector(embedding_output, ComputeEmbeddingReference(token_ids, embedding_table, 4), "embedding") ||
        !CheckVector(rope_output, ComputeRopeReference(rope_input, 2, 1, 4, 4, 3, rope_params.rope_theta), "rope") ||
        !CheckVector(softmax_output, ComputeSoftmaxReference(softmax_input, 2, 3), "softmax")) {
        return 1;
    }

    bool found_embedding = false;
    bool found_rope = false;
    bool found_softmax = false;
    for (const auto& entry : arena->GetTelemetrySnapshot()) {
        found_embedding = found_embedding || entry.name == "EmbeddingOp";
        found_rope = found_rope || entry.name == "RopeOp";
        found_softmax = found_softmax || entry.name == "SoftmaxOp";
    }
    if (!found_embedding || !found_rope || !found_softmax) {
        std::cerr << "missing Phase 3 telemetry entries\n";
        return 1;
    }

    std::cout << "test_phase3_kernels passed\n";
    return 0;
}