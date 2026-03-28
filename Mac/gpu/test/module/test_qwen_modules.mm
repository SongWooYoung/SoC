#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "buffer/buffer_arena.h"
#include "buffer/metal_buffer.h"
#include "kernel/pipeline_cache.h"
#include "metal/metal_context.h"
#include "module/qwen_attention.h"
#include "module/qwen_block.h"
#include "module/qwen_mlp.h"
#include "runtime/kv_cache.h"
#include "tensor/device_tensor.h"
#include "tensor/tensor_desc.h"

namespace {

struct TestConfig {
    std::size_t row_count = 2;
    std::size_t hidden_size = 8;
    std::size_t num_attention_heads = 4;
    std::size_t num_key_value_heads = 2;
    std::size_t head_dim = 2;
    std::size_t intermediate_size = 12;
    std::size_t position_offset = 2;
    float rope_theta = 1000000.0f;
    float rms_epsilon = 1.0e-6f;
};

float MakeValue(std::size_t index, float scale) {
    return static_cast<float>((static_cast<int>(index % 17) - 8)) * scale;
}

std::vector<float> MakeData(std::size_t count, float scale) {
    std::vector<float> values(count, 0.0f);
    for (std::size_t index = 0; index < count; ++index) {
        values[index] = MakeValue(index, scale);
    }
    return values;
}

bool NearlyEqual(float lhs, float rhs) {
    return std::fabs(lhs - rhs) < 1.0e-3f;
}

bool CheckVector(const std::vector<float>& actual, const std::vector<float>& expected, const std::string& label) {
    if (actual.size() != expected.size()) {
        std::cerr << label << " size mismatch\n";
        return false;
    }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!NearlyEqual(actual[index], expected[index])) {
            std::cerr << label << " mismatch at index " << index << ": actual=" << actual[index]
                      << " expected=" << expected[index] << '\n';
            return false;
        }
    }
    return true;
}

std::vector<float> LinearReference(const std::vector<float>& input,
                                   std::size_t row_count,
                                   std::size_t input_size,
                                   const std::vector<float>& weight,
                                   std::size_t output_size) {
    std::vector<float> output(row_count * output_size, 0.0f);
    for (std::size_t row = 0; row < row_count; ++row) {
        for (std::size_t out = 0; out < output_size; ++out) {
            float value = 0.0f;
            for (std::size_t inner = 0; inner < input_size; ++inner) {
                value += input[row * input_size + inner] * weight[out * input_size + inner];
            }
            output[row * output_size + out] = value;
        }
    }
    return output;
}

std::vector<float> ApplySiLU(const std::vector<float>& input) {
    std::vector<float> output = input;
    for (float& value : output) {
        value = value / (1.0f + std::exp(-value));
    }
    return output;
}

std::vector<float> ElementwiseMul(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    std::vector<float> output(lhs.size(), 0.0f);
    for (std::size_t index = 0; index < lhs.size(); ++index) {
        output[index] = lhs[index] * rhs[index];
    }
    return output;
}

std::vector<float> AddVectors(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    std::vector<float> output(lhs.size(), 0.0f);
    for (std::size_t index = 0; index < lhs.size(); ++index) {
        output[index] = lhs[index] + rhs[index];
    }
    return output;
}

std::vector<float> RmsNormReference(const std::vector<float>& input,
                                    std::size_t row_count,
                                    std::size_t row_size,
                                    const std::vector<float>& weight,
                                    float epsilon) {
    std::vector<float> output(input.size(), 0.0f);
    for (std::size_t row = 0; row < row_count; ++row) {
        float sum_squares = 0.0f;
        for (std::size_t col = 0; col < row_size; ++col) {
            const float value = input[row * row_size + col];
            sum_squares += value * value;
        }
        const float inv_rms = 1.0f / std::sqrt(sum_squares / static_cast<float>(row_size) + epsilon);
        for (std::size_t col = 0; col < row_size; ++col) {
            output[row * row_size + col] = input[row * row_size + col] * inv_rms * weight[col];
        }
    }
    return output;
}

std::vector<float> RopeReference(const std::vector<float>& input,
                                 std::size_t row_count,
                                 const TestConfig& config,
                                 std::size_t head_count,
                                 std::size_t position_offset) {
    std::vector<float> output = input;
    const std::size_t pair_count = config.head_dim / 2;
    for (std::size_t row = 0; row < row_count; ++row) {
        for (std::size_t head = 0; head < head_count; ++head) {
            const std::size_t base = row * head_count * config.head_dim + head * config.head_dim;
            for (std::size_t pair = 0; pair < pair_count; ++pair) {
                const float exponent = (2.0f * static_cast<float>(pair)) / static_cast<float>(config.head_dim);
                const float angle = static_cast<float>(row + position_offset) * std::pow(config.rope_theta, -exponent);
                const float cosine = std::cos(angle);
                const float sine = std::sin(angle);
                const float left = input[base + pair];
                const float right = input[base + pair + pair_count];
                output[base + pair] = left * cosine - right * sine;
                output[base + pair + pair_count] = left * sine + right * cosine;
            }
        }
    }
    return output;
}

std::vector<float> SoftmaxRows(const std::vector<float>& input, std::size_t row_count, std::size_t row_size) {
    std::vector<float> output(input.size(), 0.0f);
    for (std::size_t row = 0; row < row_count; ++row) {
        float max_value = input[row * row_size];
        for (std::size_t col = 1; col < row_size; ++col) {
            max_value = std::max(max_value, input[row * row_size + col]);
        }
        float sum = 0.0f;
        for (std::size_t col = 0; col < row_size; ++col) {
            output[row * row_size + col] = std::exp(input[row * row_size + col] - max_value);
            sum += output[row * row_size + col];
        }
        for (std::size_t col = 0; col < row_size; ++col) {
            output[row * row_size + col] /= sum;
        }
    }
    return output;
}

std::vector<float> AttentionReference(const std::vector<float>& input,
                                      const std::vector<float>& residual,
                                      const std::vector<float>& q_weight,
                                      const std::vector<float>& k_weight,
                                      const std::vector<float>& v_weight,
                                      const std::vector<float>& o_weight,
                                      const std::vector<float>& q_norm_weight,
                                      const std::vector<float>& k_norm_weight,
                                      std::size_t row_count,
                                      std::size_t position_offset,
                                      const TestConfig& config,
                                      bool add_residual) {
    const std::size_t query_hidden = config.num_attention_heads * config.head_dim;
    const std::size_t key_value_hidden = config.num_key_value_heads * config.head_dim;
    std::vector<float> queries = LinearReference(input, row_count, config.hidden_size, q_weight, query_hidden);
    std::vector<float> keys = LinearReference(input, row_count, config.hidden_size, k_weight, key_value_hidden);
    std::vector<float> values = LinearReference(input, row_count, config.hidden_size, v_weight, key_value_hidden);

    queries = RmsNormReference(queries,
                               row_count * config.num_attention_heads,
                               config.head_dim,
                               q_norm_weight,
                               config.rms_epsilon);
    keys = RmsNormReference(keys,
                            row_count * config.num_key_value_heads,
                            config.head_dim,
                            k_norm_weight,
                            config.rms_epsilon);
    queries = RopeReference(queries, row_count, config, config.num_attention_heads, position_offset);
    keys = RopeReference(keys, row_count, config, config.num_key_value_heads, position_offset);

    const std::size_t score_rows = row_count * config.num_attention_heads;
    const std::size_t group_size = config.num_attention_heads / config.num_key_value_heads;
    std::vector<float> scores(score_rows * row_count, 0.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(config.head_dim));
    for (std::size_t row = 0; row < row_count; ++row) {
        for (std::size_t q_head = 0; q_head < config.num_attention_heads; ++q_head) {
            const std::size_t score_row = row * config.num_attention_heads + q_head;
            const std::size_t kv_head = q_head / group_size;
            const std::size_t query_base = row * query_hidden + q_head * config.head_dim;
            for (std::size_t key_row = 0; key_row < row_count; ++key_row) {
                if (key_row > row) {
                    scores[score_row * row_count + key_row] = -INFINITY;
                    continue;
                }
                const std::size_t key_base = key_row * key_value_hidden + kv_head * config.head_dim;
                float value = 0.0f;
                for (std::size_t dim = 0; dim < config.head_dim; ++dim) {
                    value += queries[query_base + dim] * keys[key_base + dim];
                }
                scores[score_row * row_count + key_row] = value * scale;
            }
        }
    }

    const std::vector<float> probabilities = SoftmaxRows(scores, score_rows, row_count);
    std::vector<float> context(row_count * query_hidden, 0.0f);
    for (std::size_t row = 0; row < row_count; ++row) {
        for (std::size_t q_head = 0; q_head < config.num_attention_heads; ++q_head) {
            const std::size_t score_row = row * config.num_attention_heads + q_head;
            const std::size_t kv_head = q_head / group_size;
            const std::size_t context_base = row * query_hidden + q_head * config.head_dim;
            for (std::size_t dim = 0; dim < config.head_dim; ++dim) {
                float value = 0.0f;
                for (std::size_t key_row = 0; key_row < row_count; ++key_row) {
                    const std::size_t value_base = key_row * key_value_hidden + kv_head * config.head_dim;
                    value += probabilities[score_row * row_count + key_row] * values[value_base + dim];
                }
                context[context_base + dim] = value;
            }
        }
    }

    std::vector<float> output = LinearReference(context, row_count, query_hidden, o_weight, config.hidden_size);
    if (add_residual) {
        output = AddVectors(output, residual);
    }
    return output;
}

std::vector<float> SliceRows(const std::vector<float>& input,
                             std::size_t row_size,
                             std::size_t row_offset,
                             std::size_t row_count) {
    return std::vector<float>(input.begin() + row_offset * row_size,
                              input.begin() + (row_offset + row_count) * row_size);
}

std::vector<float> MlpReference(const std::vector<float>& input,
                                const std::vector<float>* residual,
                                const std::vector<float>& gate_weight,
                                const std::vector<float>& up_weight,
                                const std::vector<float>& down_weight,
                                const TestConfig& config,
                                bool add_residual) {
    std::vector<float> gate = ApplySiLU(LinearReference(input, config.row_count, config.hidden_size, gate_weight, config.intermediate_size));
    std::vector<float> up = LinearReference(input, config.row_count, config.hidden_size, up_weight, config.intermediate_size);
    std::vector<float> fused = ElementwiseMul(gate, up);
    std::vector<float> output = LinearReference(fused, config.row_count, config.intermediate_size, down_weight, config.hidden_size);
    if (add_residual) {
        output = AddVectors(output, residual == nullptr ? input : *residual);
    }
    return output;
}

std::vector<float> BlockReference(const std::vector<float>& input,
                                  const std::vector<float>& input_ln_weight,
                                  const std::vector<float>& q_weight,
                                  const std::vector<float>& k_weight,
                                  const std::vector<float>& v_weight,
                                  const std::vector<float>& o_weight,
                                  const std::vector<float>& q_norm_weight,
                                  const std::vector<float>& k_norm_weight,
                                  const std::vector<float>& post_ln_weight,
                                  const std::vector<float>& gate_weight,
                                  const std::vector<float>& up_weight,
                                  const std::vector<float>& down_weight,
                                  const TestConfig& config) {
    const std::vector<float> input_norm =
        RmsNormReference(input, config.row_count, config.hidden_size, input_ln_weight, config.rms_epsilon);
    const std::vector<float> attention = AttentionReference(input_norm,
                                                            input,
                                                            q_weight,
                                                            k_weight,
                                                            v_weight,
                                                            o_weight,
                                                            q_norm_weight,
                                                            k_norm_weight,
                                                            config.row_count,
                                                            config.position_offset,
                                                            config,
                                                            true);
    const std::vector<float> post_norm =
        RmsNormReference(attention, config.row_count, config.hidden_size, post_ln_weight, config.rms_epsilon);
    return MlpReference(post_norm, &attention, gate_weight, up_weight, down_weight, config, true);
}

std::shared_ptr<soc::gpu::MetalBuffer> CreateBuffer(const soc::gpu::MetalContext& context,
                                                    const std::vector<float>& values,
                                                    const std::string& label,
                                                    std::string* error_message) {
    auto buffer = soc::gpu::MetalBuffer::CreateShared(context, sizeof(float) * values.size(), label, error_message);
    if (buffer == nullptr) {
        return nullptr;
    }
    if (!buffer->Write(values.data(), sizeof(float) * values.size(), 0, error_message)) {
        return nullptr;
    }
    return buffer;
}

soc::gpu::DeviceTensor CreateTensor(const std::shared_ptr<soc::gpu::MetalBuffer>& buffer,
                                    const std::vector<std::size_t>& shape) {
    return soc::gpu::DeviceTensor(buffer, 0, soc::gpu::TensorDesc::CreateContiguous(soc::gpu::DataType::kFloat32, shape));
}

}  // namespace

int main() {
    const TestConfig config;
    const std::vector<float> input = MakeData(config.row_count * config.hidden_size, 1.0f / 23.0f);
    const std::vector<float> decode_input = MakeData(config.hidden_size, 1.0f / 59.0f);
    std::vector<float> full_input = input;
    full_input.insert(full_input.end(), decode_input.begin(), decode_input.end());
    const std::vector<float> q_weight = MakeData(config.hidden_size * config.hidden_size, 1.0f / 29.0f);
    const std::vector<float> k_weight = MakeData((config.num_key_value_heads * config.head_dim) * config.hidden_size, 1.0f / 31.0f);
    const std::vector<float> v_weight = MakeData((config.num_key_value_heads * config.head_dim) * config.hidden_size, 1.0f / 37.0f);
    const std::vector<float> o_weight = MakeData(config.hidden_size * config.hidden_size, 1.0f / 41.0f);
    const std::vector<float> q_norm_weight = MakeData(config.head_dim, 1.0f / 11.0f);
    const std::vector<float> k_norm_weight = MakeData(config.head_dim, 1.0f / 13.0f);
    const std::vector<float> input_ln_weight = MakeData(config.hidden_size, 1.0f / 17.0f);
    const std::vector<float> post_ln_weight = MakeData(config.hidden_size, 1.0f / 19.0f);
    const std::vector<float> gate_weight = MakeData(config.intermediate_size * config.hidden_size, 1.0f / 43.0f);
    const std::vector<float> up_weight = MakeData(config.intermediate_size * config.hidden_size, 1.0f / 47.0f);
    const std::vector<float> down_weight = MakeData(config.hidden_size * config.intermediate_size, 1.0f / 53.0f);

    std::string error_message;
    auto context = soc::gpu::MetalContext::CreateDefault("build/shaders/gpu.metallib", "shaders/gpu_kernels.metal", &error_message);
    if (context == nullptr) {
        std::cerr << "failed to create Metal context: " << error_message << '\n';
        return 1;
    }

    soc::gpu::PipelineCache pipeline_cache(*context);
    auto arena = soc::gpu::BufferArena::CreateShared(*context, 1 << 20, "module_temp", &error_message);
    if (arena == nullptr) {
        std::cerr << "failed to create temporary arena: " << error_message << '\n';
        return 1;
    }

    auto input_buffer = CreateBuffer(*context, input, "input", &error_message);
    auto decode_input_buffer = CreateBuffer(*context, decode_input, "decode_input", &error_message);
    auto q_weight_buffer = CreateBuffer(*context, q_weight, "q_weight", &error_message);
    auto k_weight_buffer = CreateBuffer(*context, k_weight, "k_weight", &error_message);
    auto v_weight_buffer = CreateBuffer(*context, v_weight, "v_weight", &error_message);
    auto o_weight_buffer = CreateBuffer(*context, o_weight, "o_weight", &error_message);
    auto q_norm_buffer = CreateBuffer(*context, q_norm_weight, "q_norm", &error_message);
    auto k_norm_buffer = CreateBuffer(*context, k_norm_weight, "k_norm", &error_message);
    auto input_ln_buffer = CreateBuffer(*context, input_ln_weight, "input_ln", &error_message);
    auto post_ln_buffer = CreateBuffer(*context, post_ln_weight, "post_ln", &error_message);
    auto gate_weight_buffer = CreateBuffer(*context, gate_weight, "gate_weight", &error_message);
    auto up_weight_buffer = CreateBuffer(*context, up_weight, "up_weight", &error_message);
    auto down_weight_buffer = CreateBuffer(*context, down_weight, "down_weight", &error_message);
    auto attention_output_buffer = soc::gpu::MetalBuffer::CreateShared(*context, sizeof(float) * input.size(), "attention_out", &error_message);
    auto prefill_output_buffer = soc::gpu::MetalBuffer::CreateShared(*context, sizeof(float) * input.size(), "prefill_out", &error_message);
    auto decode_output_buffer = soc::gpu::MetalBuffer::CreateShared(*context, sizeof(float) * config.hidden_size, "decode_out", &error_message);
    auto mlp_output_buffer = soc::gpu::MetalBuffer::CreateShared(*context, sizeof(float) * input.size(), "mlp_out", &error_message);
    auto block_output_buffer = soc::gpu::MetalBuffer::CreateShared(*context, sizeof(float) * input.size(), "block_out", &error_message);
    if (input_buffer == nullptr || decode_input_buffer == nullptr || q_weight_buffer == nullptr || k_weight_buffer == nullptr || v_weight_buffer == nullptr ||
        o_weight_buffer == nullptr || q_norm_buffer == nullptr || k_norm_buffer == nullptr || input_ln_buffer == nullptr ||
        post_ln_buffer == nullptr || gate_weight_buffer == nullptr || up_weight_buffer == nullptr || down_weight_buffer == nullptr ||
        attention_output_buffer == nullptr || prefill_output_buffer == nullptr || decode_output_buffer == nullptr ||
        mlp_output_buffer == nullptr || block_output_buffer == nullptr) {
        std::cerr << "buffer allocation failed: " << error_message << '\n';
        return 1;
    }

    const soc::gpu::DeviceTensor input_tensor = CreateTensor(input_buffer, {config.row_count, config.hidden_size});
    const soc::gpu::DeviceTensor decode_input_tensor = CreateTensor(decode_input_buffer, {1, config.hidden_size});
    const soc::gpu::DeviceTensor attention_output_tensor = CreateTensor(attention_output_buffer, {config.row_count, config.hidden_size});
    const soc::gpu::DeviceTensor prefill_output_tensor = CreateTensor(prefill_output_buffer, {config.row_count, config.hidden_size});
    const soc::gpu::DeviceTensor decode_output_tensor = CreateTensor(decode_output_buffer, {1, config.hidden_size});
    const soc::gpu::DeviceTensor mlp_output_tensor = CreateTensor(mlp_output_buffer, {config.row_count, config.hidden_size});
    const soc::gpu::DeviceTensor block_output_tensor = CreateTensor(block_output_buffer, {config.row_count, config.hidden_size});

    soc::gpu::QwenAttentionWeights attention_weights;
    attention_weights.q_proj_weight = CreateTensor(q_weight_buffer, {config.hidden_size, config.hidden_size});
    attention_weights.k_proj_weight = CreateTensor(k_weight_buffer, {config.num_key_value_heads * config.head_dim, config.hidden_size});
    attention_weights.v_proj_weight = CreateTensor(v_weight_buffer, {config.num_key_value_heads * config.head_dim, config.hidden_size});
    attention_weights.o_proj_weight = CreateTensor(o_weight_buffer, {config.hidden_size, config.hidden_size});
    attention_weights.q_norm_weight = CreateTensor(q_norm_buffer, {config.head_dim});
    attention_weights.k_norm_weight = CreateTensor(k_norm_buffer, {config.head_dim});

    soc::gpu::QwenAttentionParams attention_params;
    attention_params.num_attention_heads = config.num_attention_heads;
    attention_params.num_key_value_heads = config.num_key_value_heads;
    attention_params.head_dim = config.head_dim;
    attention_params.rotary_dim = config.head_dim;
    attention_params.position_offset = config.position_offset;
    attention_params.rope_theta = config.rope_theta;
    attention_params.rms_epsilon = config.rms_epsilon;
    attention_params.add_residual = true;
    if (!soc::gpu::QwenAttention::Run(*context,
                                      &pipeline_cache,
                                      input_tensor,
                                      &input_tensor,
                                      attention_weights,
                                      attention_output_tensor,
                                      attention_params,
                                      arena.get(),
                                      &error_message)) {
        std::cerr << "QwenAttention failed: " << error_message << '\n';
        return 1;
    }

    auto kv_cache = soc::gpu::KVCache::CreateShared(*context,
                                                    1,
                                                    config.num_key_value_heads,
                                                    config.head_dim,
                                                    4,
                                                    "test_kv_cache",
                                                    &error_message);
    if (kv_cache == nullptr) {
        std::cerr << "KVCache creation failed: " << error_message << '\n';
        return 1;
    }
    if (!soc::gpu::QwenAttention::RunPrefill(*context,
                                             &pipeline_cache,
                                             input_tensor,
                                             &input_tensor,
                                             attention_weights,
                                             kv_cache.get(),
                                             0,
                                             prefill_output_tensor,
                                             attention_params,
                                             arena.get(),
                                             &error_message)) {
        std::cerr << "QwenAttention prefill failed: " << error_message << '\n';
        return 1;
    }
    if (kv_cache->ViewForLayer(0).sequence_length != config.row_count) {
        std::cerr << "KVCache prefill sequence length mismatch\n";
        return 1;
    }
    soc::gpu::QwenAttentionParams decode_attention_params = attention_params;
    decode_attention_params.position_offset = config.position_offset + config.row_count;
    if (!soc::gpu::QwenAttention::RunDecode(*context,
                                            &pipeline_cache,
                                            decode_input_tensor,
                                            &decode_input_tensor,
                                            attention_weights,
                                            kv_cache.get(),
                                            0,
                                            decode_output_tensor,
                                            decode_attention_params,
                                            arena.get(),
                                            &error_message)) {
        std::cerr << "QwenAttention decode failed: " << error_message << '\n';
        return 1;
    }
    if (kv_cache->ViewForLayer(0).sequence_length != config.row_count + 1) {
        std::cerr << "KVCache decode sequence length mismatch\n";
        return 1;
    }

    soc::gpu::QwenMlpWeights mlp_weights;
    mlp_weights.gate_proj_weight = CreateTensor(gate_weight_buffer, {config.intermediate_size, config.hidden_size});
    mlp_weights.up_proj_weight = CreateTensor(up_weight_buffer, {config.intermediate_size, config.hidden_size});
    mlp_weights.down_proj_weight = CreateTensor(down_weight_buffer, {config.hidden_size, config.intermediate_size});
    soc::gpu::QwenMlpParams mlp_params;
    mlp_params.intermediate_size = config.intermediate_size;
    mlp_params.add_residual = true;
    if (!soc::gpu::QwenMLP::Run(*context,
                                &pipeline_cache,
                                input_tensor,
                                &input_tensor,
                                mlp_weights,
                                mlp_output_tensor,
                                mlp_params,
                                arena.get(),
                                &error_message)) {
        std::cerr << "QwenMLP failed: " << error_message << '\n';
        return 1;
    }

    soc::gpu::QwenBlockWeights block_weights;
    block_weights.input_layernorm_weight = CreateTensor(input_ln_buffer, {config.hidden_size});
    block_weights.attention = attention_weights;
    block_weights.post_attention_layernorm_weight = CreateTensor(post_ln_buffer, {config.hidden_size});
    block_weights.mlp = mlp_weights;
    soc::gpu::QwenBlockParams block_params;
    block_params.attention = attention_params;
    block_params.mlp = mlp_params;
    block_params.rms_epsilon = config.rms_epsilon;
    if (!soc::gpu::QwenBlock::Run(*context,
                                  &pipeline_cache,
                                  input_tensor,
                                  block_weights,
                                  block_output_tensor,
                                  block_params,
                                  arena.get(),
                                  &error_message)) {
        std::cerr << "QwenBlock failed: " << error_message << '\n';
        return 1;
    }

    std::vector<float> attention_output(input.size(), 0.0f);
    std::vector<float> prefill_output(input.size(), 0.0f);
    std::vector<float> decode_output(config.hidden_size, 0.0f);
    std::vector<float> mlp_output(input.size(), 0.0f);
    std::vector<float> block_output(input.size(), 0.0f);
    if (!attention_output_buffer->Read(attention_output.data(), sizeof(float) * attention_output.size(), 0, &error_message) ||
        !prefill_output_buffer->Read(prefill_output.data(), sizeof(float) * prefill_output.size(), 0, &error_message) ||
        !decode_output_buffer->Read(decode_output.data(), sizeof(float) * decode_output.size(), 0, &error_message) ||
        !mlp_output_buffer->Read(mlp_output.data(), sizeof(float) * mlp_output.size(), 0, &error_message) ||
        !block_output_buffer->Read(block_output.data(), sizeof(float) * block_output.size(), 0, &error_message)) {
        std::cerr << "readback failed: " << error_message << '\n';
        return 1;
    }

    const std::vector<float> attention_reference = AttentionReference(input,
                                                                      input,
                                                                      q_weight,
                                                                      k_weight,
                                                                      v_weight,
                                                                      o_weight,
                                                                      q_norm_weight,
                                                                      k_norm_weight,
                                                                    config.row_count,
                                                                    config.position_offset,
                                                                      config,
                                                                      true);
                const std::vector<float> full_attention_reference = AttentionReference(full_input,
                                                                        full_input,
                                                                        q_weight,
                                                                        k_weight,
                                                                        v_weight,
                                                                        o_weight,
                                                                        q_norm_weight,
                                                                        k_norm_weight,
                                                                        config.row_count + 1,
                                                                        config.position_offset,
                                                                        config,
                                                                        true);
    const std::vector<float> mlp_reference = MlpReference(input,
                                                          &input,
                                                          gate_weight,
                                                          up_weight,
                                                          down_weight,
                                                          config,
                                                          true);
    const std::vector<float> block_reference = BlockReference(input,
                                                              input_ln_weight,
                                                              q_weight,
                                                              k_weight,
                                                              v_weight,
                                                              o_weight,
                                                              q_norm_weight,
                                                              k_norm_weight,
                                                              post_ln_weight,
                                                              gate_weight,
                                                              up_weight,
                                                              down_weight,
                                                              config);
    if (!CheckVector(attention_output, attention_reference, "attention") ||
        !CheckVector(prefill_output,
                     SliceRows(full_attention_reference, config.hidden_size, 0, config.row_count),
                     "attention_prefill") ||
        !CheckVector(decode_output,
                     SliceRows(full_attention_reference, config.hidden_size, config.row_count, 1),
                     "attention_decode") ||
        !CheckVector(mlp_output, mlp_reference, "mlp") ||
        !CheckVector(block_output, block_reference, "block")) {
        return 1;
    }

    bool found_attention = false;
    bool found_mlp = false;
    bool found_block = false;
    for (const auto& entry : arena->GetTelemetrySnapshot()) {
        found_attention = found_attention || entry.name == "QwenAttention";
        found_mlp = found_mlp || entry.name == "QwenMLP";
        found_block = found_block || entry.name == "QwenBlock";
    }
    if (!found_attention || !found_mlp || !found_block) {
        std::cerr << "missing module telemetry entries\n";
        return 1;
    }

    std::cout << "test_qwen_modules passed\n";
    return 0;
}