#include "models/qwen3_5/qwen3_5_runner.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "buffer/buffer_arena.h"
#include "buffer/metal_buffer.h"
#include "header/dtype.h"
#include "header/embedding.h"
#include "header/grouped_query_attention.h"
#include "header/linear.h"
#include "header/qwen_mlp.h"
#include "header/rope.h"
#include "models/qwen3/modules/qwen_mlp.h"
#include "op/embedding_op.h"
#include "op/linear_op.h"
#include "op/rms_norm_op.h"

namespace soc::gpu::models::qwen3_5 {
namespace {

bool DebugGpuPath() {
    const char* value = std::getenv("SOC_GPU_DEBUG_QWEN3_5_GPU_PATH");
    return value != nullptr && std::string(value) == "1";
}

bool UseExperimentalQwen3_5ProjectionGpu() {
    const char* value = std::getenv("SOC_GPU_ENABLE_EXPERIMENTAL_QWEN3_5_PROJECTION_GPU");
    return value != nullptr && std::string(value) == "1";
}

bool UseExperimentalQwen3_5SingleTokenProjectionGpu() {
    const char* value = std::getenv("SOC_GPU_ENABLE_EXPERIMENTAL_QWEN3_5_SINGLE_TOKEN_PROJECTION_GPU");
    return value != nullptr && std::string(value) == "1";
}

bool UseExperimentalQwen3_5EmbeddingGpu() {
    const char* value = std::getenv("SOC_GPU_ENABLE_EXPERIMENTAL_QWEN3_5_EMBEDDING_GPU");
    return value != nullptr && std::string(value) == "1";
}

bool UseExperimentalQwen3_5DecodeEmbeddingGpu() {
    const char* value = std::getenv("SOC_GPU_ENABLE_EXPERIMENTAL_QWEN3_5_DECODE_EMBEDDING_GPU");
    return value != nullptr && std::string(value) == "1";
}

bool UseExperimentalQwen3_5DecodeOutputProjectionGpu() {
    const char* value = std::getenv("SOC_GPU_ENABLE_EXPERIMENTAL_QWEN3_5_DECODE_OUTPUT_PROJECTION_GPU");
    return value != nullptr && std::string(value) == "1";
}

void Require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::size_t Product(const std::vector<std::size_t>& shape) {
    std::size_t product = 1;
    for (std::size_t dim : shape) {
        product *= dim;
    }
    return product;
}

Tensor MakeFloatTensor(const std::vector<std::size_t>& shape) {
    Storage storage = Storage::AllocateOwned(Product(shape) * sizeof(float));
    return Tensor(std::move(storage), DType::Float32, shape);
}

Tensor MakeIntTensor(const std::vector<int32_t>& values, const std::vector<std::size_t>& shape) {
    return Tensor(Storage::FromOwnedCopy(values.data(), values.size() * sizeof(int32_t)), DType::Int32, shape);
}

Tensor MakeFloatTensorFromVector(const std::vector<float>& values, const std::vector<std::size_t>& shape) {
    Require(Product(shape) == values.size(), "float tensor vector size mismatch");
    Tensor tensor = MakeFloatTensor(shape);
    std::memcpy(tensor.data<float>(), values.data(), values.size() * sizeof(float));
    return tensor;
}

std::vector<Qwen3_5TopLogitEntry> ComputeTopLogits(const Tensor& logits, std::size_t top_k) {
    Require(logits.dtype() == DType::Float32, "top logits expect float32 tensor");
    const Tensor flat = logits.dim() == 2 ? logits : logits.Reshape({1, logits.shape().back()});
    const float* data = flat.data<float>();
    const std::size_t vocab_size = flat.shape()[1];
    std::vector<std::size_t> indices(vocab_size);
    for (std::size_t index = 0; index < vocab_size; ++index) {
        indices[index] = index;
    }
    const std::size_t effective_top_k = std::min<std::size_t>(std::max<std::size_t>(top_k, 1), vocab_size);
    std::partial_sort(indices.begin(),
                      indices.begin() + effective_top_k,
                      indices.end(),
                      [&](const std::size_t left, const std::size_t right) { return data[left] > data[right]; });
    std::vector<Qwen3_5TopLogitEntry> entries;
    entries.reserve(effective_top_k);
    for (std::size_t rank = 0; rank < effective_top_k; ++rank) {
        entries.push_back(Qwen3_5TopLogitEntry{static_cast<int>(indices[rank]), data[indices[rank]]});
    }
    return entries;
}

int ArgmaxTokenId(const Tensor& logits) {
    Require(logits.dtype() == DType::Float32, "argmax expects float32 tensor");
    const Tensor flat = logits.dim() == 2 ? logits : logits.Reshape({1, logits.shape().back()});
    const float* data = flat.data<float>();
    const std::size_t vocab_size = flat.shape()[1];
    std::size_t best_index = 0;
    for (std::size_t index = 1; index < vocab_size; ++index) {
        if (data[index] > data[best_index]) {
            best_index = index;
        }
    }
    return static_cast<int>(best_index);
}

std::pair<float, float> LogitDiffStats(const Tensor& left, const Tensor& right) {
    Require(left.dtype() == DType::Float32 && right.dtype() == DType::Float32, "logit diff expects float32 tensors");
    const Tensor flat_left = left.dim() == 2 ? left : left.Reshape({1, left.shape().back()});
    const Tensor flat_right = right.dim() == 2 ? right : right.Reshape({1, right.shape().back()});
    Require(flat_left.shape() == flat_right.shape(), "logit diff shape mismatch");
    const float* left_data = flat_left.data<float>();
    const float* right_data = flat_right.data<float>();
    float max_abs = 0.0f;
    float sum_abs = 0.0f;
    for (std::size_t index = 0; index < flat_left.numel(); ++index) {
        const float diff = std::fabs(left_data[index] - right_data[index]);
        max_abs = std::max(max_abs, diff);
        sum_abs += diff;
    }
    return {max_abs, flat_left.numel() == 0 ? 0.0f : (sum_abs / static_cast<float>(flat_left.numel()))};
}

Tensor MakeExternalFloatTensor(float* values, const std::vector<std::size_t>& shape) {
    Require(values != nullptr, "external float tensor requires non-null data");
    return Tensor(Storage::External(values, Product(shape) * sizeof(float)), DType::Float32, shape);
}

Tensor SliceLastToken(const Tensor& input) {
    Require(input.dtype() == DType::Float32, "last token slice expects float32 tensor");
    Require(input.dim() == 3 && input.shape()[0] == 1 && input.shape()[1] >= 1, "last token slice expects [1, seq, hidden]");
    const std::size_t sequence_length = input.shape()[1];
    const std::size_t hidden_size = input.shape()[2];
    Tensor output = MakeFloatTensor({1, 1, hidden_size});
    std::memcpy(output.data<float>(),
                input.data<float>() + (sequence_length - 1) * hidden_size,
                hidden_size * sizeof(float));
    return output;
}

Tensor SliceLastLogitRow(const Tensor& input) {
    Require(input.dtype() == DType::Float32, "last logit slice expects float32 tensor");
    Require(input.dim() >= 2, "last logit slice expects rank >= 2 tensor");
    const std::size_t column_count = input.shape().back();
    const std::size_t row_count = input.numel() / column_count;
    Require(row_count >= 1, "last logit slice expects at least one row");
    Tensor output = MakeFloatTensor({1, column_count});
    std::memcpy(output.data<float>(),
                input.data<float>() + (row_count - 1) * column_count,
                column_count * sizeof(float));
    return output;
}

bool ForwardLinearGpuToCpu(const MetalContext& context,
                           PipelineCache* pipeline_cache,
                           BufferArena* temporary_arena,
                           const DeviceTensor& weight,
                           const Tensor& input,
                           std::size_t output_columns,
                           const char* profile_label,
                           Tensor* output,
                           std::string* error_message);

template <typename T>
bool UploadCpuTensor(const MetalContext& context,
                     const T* data,
                     const std::size_t byte_count,
                     const DataType data_type,
                     const std::vector<std::size_t>& shape,
                     const std::string& label,
                     DeviceTensor* output,
                     std::string* error_message) {
    if (output == nullptr) {
        if (error_message != nullptr) {
            *error_message = "device tensor output must not be null";
        }
        return false;
    }
    auto buffer = MetalBuffer::CreateForTensorClass(context,
                                                    byte_count,
                                                    label,
                                                    TensorClass::kTokenMetadata,
                                                    error_message);
    if (buffer == nullptr) {
        return false;
    }
    if (!buffer->Write(data, byte_count, 0, error_message)) {
        return false;
    }
    *output = DeviceTensor(buffer, 0, TensorDesc::CreateContiguous(data_type, shape));
    return true;
}

Tensor AddTensors(const Tensor& left, const Tensor& right) {
    Require(left.dtype() == DType::Float32 && right.dtype() == DType::Float32, "add expects float32 tensors");
    Require(left.shape() == right.shape(), "add expects matching shapes");
    Tensor output = MakeFloatTensor(left.shape());
    const float* left_data = left.data<float>();
    const float* right_data = right.data<float>();
    float* output_data = output.data<float>();
    for (std::size_t index = 0; index < left.numel(); ++index) {
        output_data[index] = left_data[index] + right_data[index];
    }
    return output;
}

Tensor SliceLastDim(const Tensor& input, const std::size_t start, const std::size_t length) {
    Require(input.dtype() == DType::Float32, "slice expects float32 tensor");
    Require(input.dim() >= 1, "slice expects rank >= 1");
    const std::vector<std::size_t>& shape = input.shape();
    const std::size_t last_dim = shape.back();
    Require(start + length <= last_dim, "slice range exceeds last dimension");

    std::vector<std::size_t> output_shape = shape;
    output_shape.back() = length;
    Tensor output = MakeFloatTensor(output_shape);

    const std::size_t outer = input.numel() / last_dim;
    const float* input_data = input.data<float>();
    float* output_data = output.data<float>();
    for (std::size_t row = 0; row < outer; ++row) {
        const float* source = input_data + row * last_dim + start;
        float* destination = output_data + row * length;
        std::memcpy(destination, source, length * sizeof(float));
    }
    return output;
}

Tensor ApplySigmoid(const Tensor& input) {
    Require(input.dtype() == DType::Float32, "sigmoid expects float32 tensor");
    Tensor output = MakeFloatTensor(input.shape());
    const float* input_data = input.data<float>();
    float* output_data = output.data<float>();
    for (std::size_t index = 0; index < input.numel(); ++index) {
        const float value = input_data[index];
        output_data[index] = 1.0f / (1.0f + std::exp(-value));
    }
    return output;
}

Tensor MultiplyTensors(const Tensor& left, const Tensor& right) {
    Require(left.dtype() == DType::Float32 && right.dtype() == DType::Float32, "mul expects float32 tensors");
    Require(left.shape() == right.shape(), "mul expects matching shapes");
    Tensor output = MakeFloatTensor(left.shape());
    const float* left_data = left.data<float>();
    const float* right_data = right.data<float>();
    float* output_data = output.data<float>();
    for (std::size_t index = 0; index < left.numel(); ++index) {
        output_data[index] = left_data[index] * right_data[index];
    }
    return output;
}

Tensor RepeatInterleaveHeads(const Tensor& input, const std::size_t repeat_factor) {
    Require(input.dtype() == DType::Float32, "repeat_interleave expects float32 tensor");
    Require(input.dim() == 4, "repeat_interleave expects rank-4 tensor");
    if (repeat_factor == 1) {
        return input;
    }

    const std::vector<std::size_t>& shape = input.shape();
    const std::size_t batch_size = shape[0];
    const std::size_t sequence_length = shape[1];
    const std::size_t head_count = shape[2];
    const std::size_t head_dim = shape[3];

    Tensor output = MakeFloatTensor({batch_size, sequence_length, head_count * repeat_factor, head_dim});
    const float* input_data = input.data<float>();
    float* output_data = output.data<float>();

    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        for (std::size_t position = 0; position < sequence_length; ++position) {
            for (std::size_t head = 0; head < head_count; ++head) {
                const float* source =
                    input_data + (((batch * sequence_length + position) * head_count + head) * head_dim);
                for (std::size_t repeat = 0; repeat < repeat_factor; ++repeat) {
                    float* destination = output_data +
                        (((batch * sequence_length + position) * (head_count * repeat_factor) +
                          head * repeat_factor + repeat) *
                         head_dim);
                    std::memcpy(destination, source, head_dim * sizeof(float));
                }
            }
        }
    }
    return output;
}

Tensor Qwen3_5RmsNorm(const Tensor& input, const Tensor& weight, const float epsilon) {
    Require(input.dtype() == DType::Float32, "Qwen3.5 RMSNorm expects float32 input");
    Require(weight.dim() == 1, "Qwen3.5 RMSNorm weight must be rank-1");
    Require(input.shape().back() == weight.shape()[0], "Qwen3.5 RMSNorm hidden size mismatch");

    const std::size_t hidden_size = weight.shape()[0];
    const std::size_t rows = input.numel() / hidden_size;
    Tensor output = MakeFloatTensor(input.shape());
    const float* input_data = input.data<float>();
    const void* weight_data = static_cast<const void*>(weight.data<std::byte>());
    float* output_data = output.data<float>();

    for (std::size_t row = 0; row < rows; ++row) {
        const float* input_row = input_data + row * hidden_size;
        float* output_row = output_data + row * hidden_size;
        float mean_square = 0.0f;
        for (std::size_t index = 0; index < hidden_size; ++index) {
            mean_square += input_row[index] * input_row[index];
        }
        mean_square /= static_cast<float>(hidden_size);
        const float scale = 1.0f / std::sqrt(mean_square + epsilon);
        for (std::size_t index = 0; index < hidden_size; ++index) {
            output_row[index] =
                input_row[index] * scale * (1.0f + DTypeReadFloat(weight_data, weight.dtype(), index));
        }
    }
    return output;
}

Tensor Qwen3_5RmsNormGated(const Tensor& hidden_states,
                           const Tensor& gate,
                           const Tensor& weight,
                           const float epsilon) {
    Require(hidden_states.dtype() == DType::Float32 && gate.dtype() == DType::Float32,
            "Qwen3.5 gated RMSNorm expects float32 tensors");
    Require(hidden_states.shape() == gate.shape(), "Qwen3.5 gated RMSNorm expects matching shapes");
    Require(weight.dim() == 1, "Qwen3.5 gated RMSNorm weight must be rank-1");
    Require(hidden_states.shape().back() == weight.shape()[0], "Qwen3.5 gated RMSNorm hidden size mismatch");

    const std::size_t hidden_size = weight.shape()[0];
    const std::size_t rows = hidden_states.numel() / hidden_size;
    Tensor output = MakeFloatTensor(hidden_states.shape());
    const float* hidden_data = hidden_states.data<float>();
    const float* gate_data = gate.data<float>();
    const void* weight_data = static_cast<const void*>(weight.data<std::byte>());
    float* output_data = output.data<float>();

    for (std::size_t row = 0; row < rows; ++row) {
        const float* hidden_row = hidden_data + row * hidden_size;
        const float* gate_row = gate_data + row * hidden_size;
        float* output_row = output_data + row * hidden_size;
        float mean_square = 0.0f;
        for (std::size_t index = 0; index < hidden_size; ++index) {
            mean_square += hidden_row[index] * hidden_row[index];
        }
        mean_square /= static_cast<float>(hidden_size);
        const float scale = 1.0f / std::sqrt(mean_square + epsilon);
        for (std::size_t index = 0; index < hidden_size; ++index) {
            const float gate_value = gate_row[index] / (1.0f + std::exp(-gate_row[index]));
            output_row[index] = hidden_row[index] * scale *
                                DTypeReadFloat(weight_data, weight.dtype(), index) * gate_value;
        }
    }
    return output;
}

Tensor ApplyPartialRoPE(const Tensor& input,
                        const std::size_t rotary_dim,
                        const double rope_theta,
                        const std::size_t position_offset) {
    Require(input.dtype() == DType::Float32, "RoPE expects float32 tensor");
    Require(input.dim() == 4, "RoPE expects [batch, seq, heads, head_dim]");
    const std::size_t head_dim = input.shape()[3];
    Require(rotary_dim <= head_dim && (rotary_dim % 2) == 0, "RoPE rotary_dim must be even and <= head_dim");

    Tensor output = MakeFloatTensor(input.shape());
    std::memcpy(output.data<float>(), input.data<float>(), input.numel() * sizeof(float));
    float* output_data = output.data<float>();
    const std::size_t batch_size = input.shape()[0];
    const std::size_t sequence_length = input.shape()[1];
    const std::size_t head_count = input.shape()[2];
    const std::size_t half_rotary_dim = rotary_dim / 2;

    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        for (std::size_t position = 0; position < sequence_length; ++position) {
            const double absolute_position = static_cast<double>(position_offset + position);
            for (std::size_t head = 0; head < head_count; ++head) {
                float* head_base =
                    output_data + (((batch * sequence_length + position) * head_count + head) * head_dim);
                for (std::size_t pair_index = 0; pair_index < half_rotary_dim; ++pair_index) {
                    const double exponent = static_cast<double>(pair_index * 2) / static_cast<double>(rotary_dim);
                    const double angle = absolute_position / std::pow(rope_theta, exponent);
                    const float cosine = static_cast<float>(std::cos(angle));
                    const float sine = static_cast<float>(std::sin(angle));
                    const float first = head_base[pair_index];
                    const float second = head_base[pair_index + half_rotary_dim];
                    head_base[pair_index] = first * cosine - second * sine;
                    head_base[pair_index + half_rotary_dim] = first * sine + second * cosine;
                }
            }
        }
    }
    return output;
}

Tensor L2NormalizeLastDim(const Tensor& input, const float epsilon = 1e-6f) {
    Require(input.dtype() == DType::Float32, "l2 norm expects float32 tensor");
    Require(input.dim() >= 1, "l2 norm expects rank >= 1");
    const std::size_t hidden_size = input.shape().back();
    const std::size_t rows = input.numel() / hidden_size;
    Tensor output = MakeFloatTensor(input.shape());
    const float* input_data = input.data<float>();
    float* output_data = output.data<float>();
    for (std::size_t row = 0; row < rows; ++row) {
        const float* input_row = input_data + row * hidden_size;
        float* output_row = output_data + row * hidden_size;
        float norm_square = 0.0f;
        for (std::size_t index = 0; index < hidden_size; ++index) {
            norm_square += input_row[index] * input_row[index];
        }
        const float scale = 1.0f / std::sqrt(norm_square + epsilon);
        for (std::size_t index = 0; index < hidden_size; ++index) {
            output_row[index] = input_row[index] * scale;
        }
    }
    return output;
}

Tensor CausalDepthwiseConv1dSiLU(const Tensor& input, const Tensor& weight, const std::size_t kernel_size) {
    Require(input.dtype() == DType::Float32, "causal conv expects float32 input");
    Require(input.dim() == 3, "causal conv expects [batch, seq, channels]");
    Require(weight.dim() == 3, "causal conv weight expects [channels, 1, kernel]");

    const std::size_t batch_size = input.shape()[0];
    const std::size_t sequence_length = input.shape()[1];
    const std::size_t channel_count = input.shape()[2];
    Require(weight.shape()[0] == channel_count && weight.shape()[2] == kernel_size,
            "causal conv weight shape mismatch");

    Tensor output = MakeFloatTensor(input.shape());
    const float* input_data = input.data<float>();
    const void* weight_data = static_cast<const void*>(weight.data<std::byte>());
    float* output_data = output.data<float>();

    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        for (std::size_t position = 0; position < sequence_length; ++position) {
            for (std::size_t channel = 0; channel < channel_count; ++channel) {
                float acc = 0.0f;
                for (std::size_t kernel_index = 0; kernel_index < kernel_size; ++kernel_index) {
                    const std::ptrdiff_t source_position = static_cast<std::ptrdiff_t>(position + kernel_index) -
                                                           static_cast<std::ptrdiff_t>(kernel_size - 1);
                    if (source_position < 0 ||
                        source_position >= static_cast<std::ptrdiff_t>(sequence_length)) {
                        continue;
                    }
                    const std::size_t input_offset =
                        ((batch * sequence_length + static_cast<std::size_t>(source_position)) * channel_count) +
                        channel;
                    const std::size_t weight_offset = channel * kernel_size + kernel_index;
                    acc += input_data[input_offset] * DTypeReadFloat(weight_data, weight.dtype(), weight_offset);
                }
                const std::size_t output_offset = ((batch * sequence_length + position) * channel_count) + channel;
                output_data[output_offset] = acc / (1.0f + std::exp(-acc));
            }
        }
    }
    return output;
}

Tensor ComputeDeltaDecay(const Tensor& a, const Tensor& a_log, const Tensor& dt_bias) {
    Require(a.dtype() == DType::Float32, "delta decay expects float32 activations");
    Require(a.dim() == 3, "delta decay expects [batch, seq, heads]");
    Require(a_log.dim() == 1 && dt_bias.dim() == 1, "delta decay expects rank-1 parameters");
    Require(a.shape()[2] == a_log.shape()[0] && a.shape()[2] == dt_bias.shape()[0], "delta decay head mismatch");

    Tensor output = MakeFloatTensor(a.shape());
    const float* a_data = a.data<float>();
    const void* a_log_data = static_cast<const void*>(a_log.data<std::byte>());
    const void* dt_bias_data = static_cast<const void*>(dt_bias.data<std::byte>());
    float* output_data = output.data<float>();

    const std::size_t heads = a.shape()[2];
    for (std::size_t index = 0; index < a.numel(); ++index) {
        const std::size_t head = index % heads;
        const float decay = std::exp(DTypeReadFloat(a_log_data, a_log.dtype(), head));
        const float dt = DTypeReadFloat(dt_bias_data, dt_bias.dtype(), head);
        const float softplus = std::log1p(std::exp(a_data[index] + dt));
        output_data[index] = -decay * softplus;
    }
    return output;
}

void ResetDecodeRuntimeState(const Qwen3_5ArchitectureSpec& spec,
                             Qwen3_5DecodeRuntimeState* state) {
    if (state == nullptr) {
        return;
    }
    state->ready = false;
    state->cached_sequence_length = 0;
    state->token_ids.clear();
    state->attention_layers.clear();
    state->attention_layers.resize(spec.num_hidden_layers);
    state->deltanet_layers.clear();
    state->deltanet_layers.resize(spec.num_hidden_layers);
    for (std::size_t layer_index = 0; layer_index < spec.num_hidden_layers; ++layer_index) {
        if (spec.layer_types[layer_index] == Qwen3_5LayerType::kGatedAttention) {
            auto& cache = state->attention_layers[layer_index];
            const std::size_t kv_slice_size = spec.num_key_value_heads * spec.attention_head_dim;
            cache.key_values.clear();
            cache.value_values.clear();
            cache.key_values.reserve(kv_slice_size * 64);
            cache.value_values.reserve(kv_slice_size * 64);
            continue;
        }
        auto& cache = state->deltanet_layers[layer_index];
        cache.recurrent_state.assign(spec.linear_num_value_heads * spec.linear_key_head_dim * spec.linear_value_head_dim,
                                     0.0f);
        cache.conv_history.clear();
        cache.conv_history.reserve((spec.linear_conv_kernel_dim > 0 ? spec.linear_conv_kernel_dim - 1 : 0) *
                                   (spec.linear_num_key_heads * spec.linear_key_head_dim * 2 +
                                    spec.linear_num_value_heads * spec.linear_value_head_dim));
        cache.conv_sequence_scratch.clear();
        cache.conv_sequence_scratch.reserve((spec.linear_conv_kernel_dim > 0 ? spec.linear_conv_kernel_dim : 1) *
                                            (spec.linear_num_key_heads * spec.linear_key_head_dim * 2 +
                                             spec.linear_num_value_heads * spec.linear_value_head_dim));
        cache.conv_history_tokens = 0;
    }
}

Tensor BuildAttentionCacheTensor(const std::vector<float>& values,
                                 const std::size_t sequence_length,
                                 const std::size_t head_count,
                                 const std::size_t head_dim) {
    Require(!values.empty(), "attention cache tensor requires non-empty values");
    return MakeExternalFloatTensor(const_cast<float*>(values.data()), {1, sequence_length, head_count, head_dim});
}

Tensor BuildDeltaConvSequence(Qwen3_5DeltaDecodeCache* cache,
                              const Tensor& current_projection,
                              const std::size_t channel_count,
                              const std::size_t kernel_size) {
    Require(cache != nullptr, "delta conv cache must not be null");
    Require(current_projection.dtype() == DType::Float32, "delta conv sequence expects float32 tensor");
    Require(current_projection.dim() == 2 && current_projection.shape()[0] == 1 && current_projection.shape()[1] == channel_count,
            "delta conv sequence expects [1, channels]");
    const std::size_t history_tokens =
        std::min<std::size_t>(cache->conv_history_tokens, kernel_size > 0 ? kernel_size - 1 : 0);
    cache->conv_sequence_scratch.resize((history_tokens + 1) * channel_count);
    float* sequence_data = cache->conv_sequence_scratch.data();
    if (history_tokens > 0) {
        std::memcpy(sequence_data, cache->conv_history.data(), history_tokens * channel_count * sizeof(float));
    }
    std::memcpy(sequence_data + history_tokens * channel_count,
                current_projection.data<float>(),
                channel_count * sizeof(float));
    return MakeExternalFloatTensor(sequence_data, {1, history_tokens + 1, channel_count});
}

void UpdateDeltaConvHistory(Qwen3_5DeltaDecodeCache* cache,
                            const Tensor& current_projection,
                            const std::size_t channel_count,
                            const std::size_t kernel_size) {
    if (cache == nullptr) {
        return;
    }
    const std::size_t max_history_tokens = kernel_size > 0 ? kernel_size - 1 : 0;
    if (max_history_tokens == 0) {
        cache->conv_history.clear();
        cache->conv_history_tokens = 0;
        return;
    }
    std::vector<float> next_history;
    const std::size_t retained_tokens = std::min(cache->conv_history_tokens, max_history_tokens - 1);
    next_history.resize((retained_tokens + 1) * channel_count, 0.0f);
    if (retained_tokens > 0) {
        const std::size_t source_offset = (cache->conv_history_tokens - retained_tokens) * channel_count;
        std::memcpy(next_history.data(),
                    cache->conv_history.data() + source_offset,
                    retained_tokens * channel_count * sizeof(float));
    }
    std::memcpy(next_history.data() + retained_tokens * channel_count,
                current_projection.data<float>(),
                channel_count * sizeof(float));
    cache->conv_history = std::move(next_history);
    cache->conv_history_tokens = std::min(max_history_tokens, retained_tokens + 1);
}

Tensor ForwardGatedAttentionPrompt(const Qwen3_5ArchitectureSpec& spec,
                                   const Qwen3_5GatedAttentionWeights& device_weights,
                                   const Qwen3_5HostGatedAttentionWeights& weights,
                                   const MetalContext& context,
                                   PipelineCache* pipeline_cache,
                                   BufferArena* temporary_arena,
                                   const Tensor& hidden_states,
                                   const std::size_t position_offset,
                                   Qwen3_5AttentionDecodeCache* cache_output) {
    const std::size_t batch_size = hidden_states.shape()[0];
    const std::size_t sequence_length = hidden_states.shape()[1];
    const std::size_t attention_dim = spec.num_attention_heads * spec.attention_head_dim;

    Tensor q_projected;
    std::string gpu_linear_error;
    if (UseExperimentalQwen3_5ProjectionGpu() &&
        ForwardLinearGpuToCpu(context,
                              pipeline_cache,
                              temporary_arena,
                              device_weights.q_proj_weight,
                              hidden_states,
                              attention_dim * 2,
                              "Qwen3_5AttnQProjPrefill",
                              &q_projected,
                              &gpu_linear_error)) {
    } else {
        q_projected = ::Linear(weights.q_proj_weight).Forward(hidden_states);
    }
    Tensor query = MakeFloatTensor({batch_size, sequence_length, spec.num_attention_heads, spec.attention_head_dim});
    Tensor gate = MakeFloatTensor({batch_size, sequence_length, spec.num_attention_heads, spec.attention_head_dim});
    const float* q_projected_data = q_projected.data<float>();
    float* query_data = query.data<float>();
    float* gate_data = gate.data<float>();
    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        for (std::size_t position = 0; position < sequence_length; ++position) {
            const std::size_t token_base = ((batch * sequence_length + position) * attention_dim * 2);
            for (std::size_t head = 0; head < spec.num_attention_heads; ++head) {
                const std::size_t source_base = token_base + head * spec.attention_head_dim * 2;
                const std::size_t destination_base =
                    (((batch * sequence_length + position) * spec.num_attention_heads + head) * spec.attention_head_dim);
                std::memcpy(query_data + destination_base,
                            q_projected_data + source_base,
                            spec.attention_head_dim * sizeof(float));
                std::memcpy(gate_data + destination_base,
                            q_projected_data + source_base + spec.attention_head_dim,
                            spec.attention_head_dim * sizeof(float));
            }
        }
    }
    Tensor key_projected;
    if (UseExperimentalQwen3_5ProjectionGpu() &&
        ForwardLinearGpuToCpu(context,
                              pipeline_cache,
                              temporary_arena,
                              device_weights.k_proj_weight,
                              hidden_states,
                              spec.num_key_value_heads * spec.attention_head_dim,
                              "Qwen3_5AttnKProjPrefill",
                              &key_projected,
                              &gpu_linear_error)) {
    } else {
        key_projected = ::Linear(weights.k_proj_weight).Forward(hidden_states);
    }
    Tensor value_projected;
    if (UseExperimentalQwen3_5ProjectionGpu() &&
        ForwardLinearGpuToCpu(context,
                              pipeline_cache,
                              temporary_arena,
                              device_weights.v_proj_weight,
                              hidden_states,
                              spec.num_key_value_heads * spec.attention_head_dim,
                              "Qwen3_5AttnVProjPrefill",
                              &value_projected,
                              &gpu_linear_error)) {
    } else {
        value_projected = ::Linear(weights.v_proj_weight).Forward(hidden_states);
    }
    Tensor key = key_projected.Reshape({batch_size, sequence_length, spec.num_key_value_heads, spec.attention_head_dim});
    const Tensor value =
        value_projected.Reshape({batch_size, sequence_length, spec.num_key_value_heads, spec.attention_head_dim});

    query = Qwen3_5RmsNorm(query, weights.q_norm_weight, spec.rms_norm_eps);
    key = Qwen3_5RmsNorm(key, weights.k_norm_weight, spec.rms_norm_eps);

    query = ApplyPartialRoPE(query, spec.rotary_dim, spec.rope.rope_theta, position_offset);
    key = ApplyPartialRoPE(key, spec.rotary_dim, spec.rope.rope_theta, position_offset);

    if (cache_output != nullptr) {
        const std::size_t kv_slice_size = spec.num_key_value_heads * spec.attention_head_dim;
        cache_output->sequence_length = sequence_length;
        cache_output->key_values.assign(key.data<float>(), key.data<float>() + sequence_length * kv_slice_size);
        cache_output->value_values.assign(value.data<float>(), value.data<float>() + sequence_length * kv_slice_size);
    }

    ::GroupedQueryAttention attention(spec.num_attention_heads,
                                      spec.num_key_value_heads,
                                      spec.attention_head_dim,
                                      true);
    Tensor attention_output = attention.Forward(query, key, value);
    attention_output = MultiplyTensors(attention_output, ApplySigmoid(gate));
    Tensor o_output;
    if ((UseExperimentalQwen3_5ProjectionGpu() || UseExperimentalQwen3_5DecodeOutputProjectionGpu()) &&
        ForwardLinearGpuToCpu(context,
                              pipeline_cache,
                              temporary_arena,
                              device_weights.o_proj_weight,
                              attention_output.Reshape({batch_size, sequence_length, attention_dim}),
                              spec.hidden_size,
                              "Qwen3_5AttnOProjPrefill",
                              &o_output,
                              &gpu_linear_error)) {
        return o_output;
    }
    return ::Linear(weights.o_proj_weight).Forward(attention_output.Reshape({batch_size, sequence_length, attention_dim}));
}

Tensor ForwardGatedDeltaNetPrompt(const Qwen3_5ArchitectureSpec& spec,
                                  const Qwen3_5GatedDeltaNetWeights& device_weights,
                                  const Qwen3_5HostGatedDeltaNetWeights& weights,
                                  const MetalContext& context,
                                  PipelineCache* pipeline_cache,
                                  BufferArena* temporary_arena,
                                  const Tensor& hidden_states,
                                  Qwen3_5DeltaDecodeCache* cache_output) {
    const std::size_t batch_size = hidden_states.shape()[0];
    const std::size_t sequence_length = hidden_states.shape()[1];
    const std::size_t key_dim = spec.linear_num_key_heads * spec.linear_key_head_dim;
    const std::size_t value_dim = spec.linear_num_value_heads * spec.linear_value_head_dim;

    std::string gpu_linear_error;
    Tensor mixed_qkv;
    if (UseExperimentalQwen3_5ProjectionGpu() &&
        ForwardLinearGpuToCpu(context,
                              pipeline_cache,
                              temporary_arena,
                              device_weights.in_proj_qkv_weight,
                              hidden_states,
                              key_dim * 2 + value_dim,
                              "Qwen3_5DeltaInProjQkvPrefill",
                              &mixed_qkv,
                              &gpu_linear_error)) {
    } else {
        mixed_qkv = ::Linear(weights.in_proj_qkv_weight).Forward(hidden_states);
    }
    mixed_qkv = CausalDepthwiseConv1dSiLU(mixed_qkv, weights.conv1d_weight, spec.linear_conv_kernel_dim);

    Tensor query = SliceLastDim(mixed_qkv, 0, key_dim)
                       .Reshape({batch_size, sequence_length, spec.linear_num_key_heads, spec.linear_key_head_dim});
    Tensor key = SliceLastDim(mixed_qkv, key_dim, key_dim)
                     .Reshape({batch_size, sequence_length, spec.linear_num_key_heads, spec.linear_key_head_dim});
    const Tensor value =
        SliceLastDim(mixed_qkv, key_dim * 2, value_dim)
            .Reshape({batch_size, sequence_length, spec.linear_num_value_heads, spec.linear_value_head_dim});
    Tensor z_projected;
    if (UseExperimentalQwen3_5ProjectionGpu() &&
        ForwardLinearGpuToCpu(context,
                              pipeline_cache,
                              temporary_arena,
                              device_weights.in_proj_z_weight,
                              hidden_states,
                              value_dim,
                              "Qwen3_5DeltaInProjZPrefill",
                              &z_projected,
                              &gpu_linear_error)) {
    } else {
        z_projected = ::Linear(weights.in_proj_z_weight).Forward(hidden_states);
    }
    const Tensor z =
        z_projected.Reshape({batch_size, sequence_length, spec.linear_num_value_heads, spec.linear_value_head_dim});
    Tensor beta_projected;
    if (UseExperimentalQwen3_5ProjectionGpu() &&
        ForwardLinearGpuToCpu(context,
                              pipeline_cache,
                              temporary_arena,
                              device_weights.in_proj_b_weight,
                              hidden_states,
                              spec.linear_num_value_heads,
                              "Qwen3_5DeltaInProjBPrefill",
                              &beta_projected,
                              &gpu_linear_error)) {
    } else {
        beta_projected = ::Linear(weights.in_proj_b_weight).Forward(hidden_states);
    }
    const Tensor beta = ApplySigmoid(beta_projected);
    Tensor g_projected;
    if (UseExperimentalQwen3_5ProjectionGpu() &&
        ForwardLinearGpuToCpu(context,
                              pipeline_cache,
                              temporary_arena,
                              device_weights.in_proj_a_weight,
                              hidden_states,
                              spec.linear_num_value_heads,
                              "Qwen3_5DeltaInProjAPrefill",
                              &g_projected,
                              &gpu_linear_error)) {
    } else {
        g_projected = ::Linear(weights.in_proj_a_weight).Forward(hidden_states);
    }
    const Tensor g = ComputeDeltaDecay(g_projected, weights.a_log, weights.dt_bias);

    query = L2NormalizeLastDim(query);
    key = L2NormalizeLastDim(key);
    {
        float* query_data = query.data<float>();
        const float query_scale = 1.0f / std::sqrt(static_cast<float>(spec.linear_key_head_dim));
        for (std::size_t index = 0; index < query.numel(); ++index) {
            query_data[index] *= query_scale;
        }
    }

    const std::size_t repeat_factor = spec.linear_num_value_heads / spec.linear_num_key_heads;
    query = RepeatInterleaveHeads(query, repeat_factor);
    key = RepeatInterleaveHeads(key, repeat_factor);

    Tensor core = MakeFloatTensor({batch_size, sequence_length, spec.linear_num_value_heads, spec.linear_value_head_dim});
    float* core_data = core.data<float>();
    const float* query_data = query.data<float>();
    const float* key_data = key.data<float>();
    const float* value_data = value.data<float>();
    const float* beta_data = beta.data<float>();
    const float* g_data = g.data<float>();

    const std::size_t state_rows = spec.linear_key_head_dim;
    const std::size_t state_cols = spec.linear_value_head_dim;
    const std::size_t state_count = spec.linear_num_value_heads;
    std::vector<float> recurrent_state(state_count * state_rows * state_cols, 0.0f);
    std::vector<float> kv_mem(state_cols, 0.0f);
    std::vector<float> delta(state_cols, 0.0f);

    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        std::fill(recurrent_state.begin(), recurrent_state.end(), 0.0f);
        for (std::size_t position = 0; position < sequence_length; ++position) {
            for (std::size_t head = 0; head < state_count; ++head) {
                const std::size_t qkv_base =
                    (((batch * sequence_length + position) * state_count + head) * state_rows);
                const std::size_t value_base =
                    (((batch * sequence_length + position) * state_count + head) * state_cols);
                const float* q_vector = query_data + qkv_base;
                const float* k_vector = key_data + qkv_base;
                const float* v_vector = value_data + value_base;
                const float beta_value =
                    beta_data[(batch * sequence_length + position) * state_count + head];
                const float g_value = g_data[(batch * sequence_length + position) * state_count + head];
                const float decay = std::exp(g_value);

                float* state = recurrent_state.data() + (head * state_rows * state_cols);
                for (std::size_t index = 0; index < state_rows * state_cols; ++index) {
                    state[index] *= decay;
                }

                std::fill(kv_mem.begin(), kv_mem.end(), 0.0f);
                for (std::size_t row = 0; row < state_rows; ++row) {
                    const float key_value = k_vector[row];
                    const float* state_row = state + row * state_cols;
                    for (std::size_t col = 0; col < state_cols; ++col) {
                        kv_mem[col] += state_row[col] * key_value;
                    }
                }

                for (std::size_t col = 0; col < state_cols; ++col) {
                    delta[col] = (v_vector[col] - kv_mem[col]) * beta_value;
                }
                for (std::size_t row = 0; row < state_rows; ++row) {
                    float* state_row = state + row * state_cols;
                    const float key_value = k_vector[row];
                    for (std::size_t col = 0; col < state_cols; ++col) {
                        state_row[col] += key_value * delta[col];
                    }
                }

                float* output_vector = core_data + value_base;
                for (std::size_t col = 0; col < state_cols; ++col) {
                    output_vector[col] = 0.0f;
                }
                for (std::size_t row = 0; row < state_rows; ++row) {
                    const float query_value = q_vector[row];
                    const float* state_row = state + row * state_cols;
                    for (std::size_t col = 0; col < state_cols; ++col) {
                        output_vector[col] += state_row[col] * query_value;
                    }
                }
            }
        }
    }

    if (cache_output != nullptr) {
        cache_output->recurrent_state = recurrent_state;
        const std::size_t max_history_tokens = spec.linear_conv_kernel_dim > 0 ? spec.linear_conv_kernel_dim - 1 : 0;
        cache_output->conv_history_tokens = std::min<std::size_t>(sequence_length, max_history_tokens);
        cache_output->conv_history.clear();
        const std::size_t conv_channel_count = key_dim * 2 + value_dim;
        if (cache_output->conv_history_tokens > 0) {
            cache_output->conv_history.resize(cache_output->conv_history_tokens * conv_channel_count);
            const float* mixed_qkv_data = mixed_qkv.data<float>();
            const std::size_t source_token_start = sequence_length - cache_output->conv_history_tokens;
            std::memcpy(cache_output->conv_history.data(),
                        mixed_qkv_data + source_token_start * conv_channel_count,
                        cache_output->conv_history.size() * sizeof(float));
        }
        cache_output->conv_sequence_scratch.clear();
        cache_output->conv_sequence_scratch.reserve((spec.linear_conv_kernel_dim > 0 ? spec.linear_conv_kernel_dim : 1) *
                                                    conv_channel_count);
    }

    Tensor gated = Qwen3_5RmsNormGated(core.Reshape({batch_size * sequence_length * spec.linear_num_value_heads,
                                                     spec.linear_value_head_dim}),
                                       z.Reshape({batch_size * sequence_length * spec.linear_num_value_heads,
                                                  spec.linear_value_head_dim}),
                                       weights.norm_weight,
                                       spec.rms_norm_eps);
    Tensor out_projected;
    if ((UseExperimentalQwen3_5ProjectionGpu() || UseExperimentalQwen3_5DecodeOutputProjectionGpu()) &&
        ForwardLinearGpuToCpu(context,
                              pipeline_cache,
                              temporary_arena,
                              device_weights.out_proj_weight,
                              gated.Reshape({batch_size, sequence_length, value_dim}),
                              spec.hidden_size,
                              "Qwen3_5DeltaOutProjPrefill",
                              &out_projected,
                              &gpu_linear_error)) {
        return out_projected;
    }
    return ::Linear(weights.out_proj_weight).Forward(gated.Reshape({batch_size, sequence_length, value_dim}));
}

Tensor ForwardQwen3_5Mlp(const Qwen3_5HostMlpWeights& weights, const Tensor& hidden_states) {
    ::QwenMLP mlp(::Linear(weights.gate_proj_weight),
                  ::Linear(weights.up_proj_weight),
                  ::Linear(weights.down_proj_weight));
    return mlp.Forward(hidden_states);
}

bool UploadCpuFloatTensor(const MetalContext& context,
                          const Tensor& tensor,
                          const std::vector<std::size_t>& shape,
                          const std::string& label,
                          DeviceTensor* output,
                          std::string* error_message) {
    Require(tensor.dtype() == DType::Float32, "upload expects float32 tensor");
    Require(tensor.is_contiguous(), "upload expects contiguous tensor");
    return UploadCpuTensor(context,
                           tensor.data<float>(),
                           tensor.nbytes(),
                           DataType::kFloat32,
                           shape,
                           label,
                           output,
                           error_message);
}

bool UploadCpuIntTensor(const MetalContext& context,
                        const std::vector<int32_t>& values,
                        const std::vector<std::size_t>& shape,
                        const std::string& label,
                        DeviceTensor* output,
                        std::string* error_message) {
    return UploadCpuTensor(context,
                           values.data(),
                           values.size() * sizeof(int32_t),
                           DataType::kInt32,
                           shape,
                           label,
                           output,
                           error_message);
}

bool DownloadDeviceFloatTensor(const DeviceTensor& tensor,
                               Tensor* output,
                               std::string* error_message) {
    if (output == nullptr) {
        if (error_message != nullptr) {
            *error_message = "cpu tensor output must not be null";
        }
        return false;
    }
    if (!tensor.IsValid() || tensor.GetDesc().GetDataType() != DataType::kFloat32) {
        if (error_message != nullptr) {
            *error_message = "download expects valid float32 device tensor";
        }
        return false;
    }
    Tensor cpu = MakeFloatTensor(tensor.GetDesc().GetShape());
    if (!tensor.GetBuffer()->Read(cpu.data<float>(), cpu.nbytes(), tensor.GetByteOffset(), error_message)) {
        return false;
    }
    *output = std::move(cpu);
    return true;
}

bool ForwardQwen3_5MlpGpu(const MetalContext& context,
                          PipelineCache* pipeline_cache,
                          BufferArena* temporary_arena,
                          const Qwen3_5MlpWeights& device_weights,
                          const Qwen3_5HostMlpWeights& host_weights,
                          const Qwen3_5ArchitectureSpec& spec,
                          const Tensor& hidden_states,
                          Tensor* output,
                          std::string* error_message) {
    if (output == nullptr || pipeline_cache == nullptr || temporary_arena == nullptr) {
        return false;
    }

    const Tensor flattened = hidden_states.Reshape({hidden_states.shape()[0] * hidden_states.shape()[1], hidden_states.shape()[2]});
    DeviceTensor input_tensor;
    if (!UploadCpuFloatTensor(context,
                              flattened,
                              {flattened.shape()[0], flattened.shape()[1]},
                              "qwen3_5_mlp_input",
                              &input_tensor,
                              error_message)) {
        return false;
    }

    auto output_buffer = MetalBuffer::CreateForTensorClass(context,
                                                           flattened.shape()[0] * spec.hidden_size * sizeof(float),
                                                           "qwen3_5_mlp_output",
                                                           TensorClass::kTokenMetadata,
                                                           error_message);
    if (output_buffer == nullptr) {
        return false;
    }
    DeviceTensor output_tensor(output_buffer,
                               0,
                               TensorDesc::CreateContiguous(DataType::kFloat32,
                                                            {flattened.shape()[0], spec.hidden_size}));
    QwenMlpWeights mlp_weights;
    mlp_weights.gate_proj_weight = device_weights.gate_proj_weight;
    mlp_weights.up_proj_weight = device_weights.up_proj_weight;
    mlp_weights.down_proj_weight = device_weights.down_proj_weight;

    QwenMlpParams params;
    params.intermediate_size = spec.intermediate_size;
    if (!QwenMLP::Run(context,
                      pipeline_cache,
                      input_tensor,
                      nullptr,
                      mlp_weights,
                      output_tensor,
                      params,
                      temporary_arena,
                      nullptr,
                      nullptr,
                      error_message)) {
        return false;
    }

    Tensor flat_output;
    if (!DownloadDeviceFloatTensor(output_tensor, &flat_output, error_message)) {
        return false;
    }
    *output = flat_output.Reshape(hidden_states.shape());
    (void)host_weights;
    return true;
}

bool ForwardEmbeddingGpu(const MetalContext& context,
                         PipelineCache* pipeline_cache,
                         BufferArena* temporary_arena,
                         const DeviceTensor& embedding_weight,
                         const std::size_t vocab_size,
                         const std::size_t hidden_size,
                         const std::vector<int32_t>& token_ids,
                         Tensor* output,
                         std::string* error_message) {
    if (output == nullptr || pipeline_cache == nullptr || temporary_arena == nullptr) {
        return false;
    }

    DeviceTensor token_tensor;
    if (!UploadCpuIntTensor(context,
                            token_ids,
                            {token_ids.size()},
                            "qwen3_5_token_ids",
                            &token_tensor,
                            error_message)) {
        return false;
    }

    auto output_buffer = MetalBuffer::CreateForTensorClass(context,
                                                           token_ids.size() * hidden_size * sizeof(float),
                                                           "qwen3_5_embedding_output",
                                                           TensorClass::kTokenMetadata,
                                                           error_message);
    if (output_buffer == nullptr) {
        return false;
    }
    DeviceTensor output_tensor(output_buffer,
                               0,
                               TensorDesc::CreateContiguous(DataType::kFloat32, {token_ids.size(), hidden_size}));
    EmbeddingParams params;
    params.token_count = static_cast<std::uint32_t>(token_ids.size());
    params.hidden_size = static_cast<std::uint32_t>(hidden_size);
    params.vocab_size = static_cast<std::uint32_t>(vocab_size);
    if (!EmbeddingOp::Run(context,
                          pipeline_cache,
                          token_tensor,
                          embedding_weight,
                          output_tensor,
                          params,
                          temporary_arena,
                          nullptr,
                          error_message)) {
        return false;
    }

    Tensor flat_output;
    if (!DownloadDeviceFloatTensor(output_tensor, &flat_output, error_message)) {
        return false;
    }
    *output = flat_output.Reshape({1, token_ids.size(), hidden_size});
    return true;
}

bool ForwardLinearGpuToCpu(const MetalContext& context,
                           PipelineCache* pipeline_cache,
                           BufferArena* temporary_arena,
                           const DeviceTensor& weight,
                           const Tensor& input,
                           const std::size_t output_columns,
                           const char* profile_label,
                           Tensor* output,
                           std::string* error_message) {
    if (output == nullptr || pipeline_cache == nullptr || temporary_arena == nullptr) {
        return false;
    }

    const Tensor flattened =
        input.dim() == 3 ? input.Reshape({input.shape()[0] * input.shape()[1], input.shape()[2]}) : input;
    DeviceTensor input_tensor;
    if (!UploadCpuFloatTensor(context,
                              flattened,
                              {flattened.shape()[0], flattened.shape()[1]},
                              std::string(profile_label) + "_input",
                              &input_tensor,
                              error_message)) {
        return false;
    }

    auto output_buffer = MetalBuffer::CreateForTensorClass(context,
                                                           flattened.shape()[0] * output_columns * sizeof(float),
                                                           std::string(profile_label) + "_output",
                                                           TensorClass::kTokenMetadata,
                                                           error_message);
    if (output_buffer == nullptr) {
        return false;
    }
    DeviceTensor output_tensor(output_buffer,
                               0,
                               TensorDesc::CreateContiguous(DataType::kFloat32,
                                                            {flattened.shape()[0], output_columns}));

    LinearParams params;
    params.matmul.row_count = static_cast<std::uint32_t>(flattened.shape()[0]);
    params.matmul.inner_dim = static_cast<std::uint32_t>(flattened.shape()[1]);
    params.matmul.column_count = static_cast<std::uint32_t>(output_columns);
    params.matmul.transpose_rhs = true;
    params.matmul.decode_mode = flattened.shape()[0] == 1;
    params.matmul.profile_label = profile_label;
    if (!LinearOp::Run(context,
                       pipeline_cache,
                       input_tensor,
                       weight,
                       nullptr,
                       nullptr,
                       output_tensor,
                       params,
                       temporary_arena,
                       nullptr,
                       error_message)) {
        return false;
    }

    Tensor flat_output;
    if (!DownloadDeviceFloatTensor(output_tensor, &flat_output, error_message)) {
        return false;
    }
    *output = input.dim() == 3 ? flat_output.Reshape({input.shape()[0], input.shape()[1], output_columns})
                               : std::move(flat_output);
    return true;
}

bool ForwardLogitsFromHiddenGpu(const MetalContext& context,
                                PipelineCache* pipeline_cache,
                                const DeviceTensor& lm_head_weight,
                                const Tensor& normalized_hidden,
                                std::size_t vocab_size,
                                DeviceTensor* logits_tensor,
                                Tensor* logits_cpu,
                                std::string* error_message) {
    if (pipeline_cache == nullptr || logits_tensor == nullptr) {
        return false;
    }

    const Tensor flattened = normalized_hidden.dim() == 3
        ? normalized_hidden.Reshape({normalized_hidden.shape()[1], normalized_hidden.shape()[2]})
        : normalized_hidden;
    DeviceTensor input_tensor;
    if (!UploadCpuFloatTensor(context,
                              flattened,
                              {flattened.shape()[0], flattened.shape()[1]},
                              "qwen3_5_lm_head_input",
                              &input_tensor,
                              error_message)) {
        return false;
    }

    auto output_buffer = MetalBuffer::CreateForTensorClass(context,
                                                           flattened.shape()[0] * vocab_size * sizeof(float),
                                                           "qwen3_5_lm_head_output",
                                                           TensorClass::kTokenMetadata,
                                                           error_message);
    if (output_buffer == nullptr) {
        return false;
    }
    *logits_tensor = DeviceTensor(output_buffer,
                                  0,
                                  TensorDesc::CreateContiguous(DataType::kFloat32,
                                                               {flattened.shape()[0], vocab_size}));

    LinearParams params;
    params.matmul.row_count = static_cast<std::uint32_t>(flattened.shape()[0]);
    params.matmul.inner_dim = static_cast<std::uint32_t>(flattened.shape()[1]);
    params.matmul.column_count = static_cast<std::uint32_t>(vocab_size);
    params.matmul.transpose_rhs = true;
    params.matmul.decode_mode = flattened.shape()[0] == 1;
    params.matmul.profile_label = flattened.shape()[0] == 1 ? "Qwen3_5LMHeadDecode" : "Qwen3_5LMHeadPrefill";
    if (!LinearOp::Run(context,
                       pipeline_cache,
                       input_tensor,
                       lm_head_weight,
                       nullptr,
                       nullptr,
                       *logits_tensor,
                       params,
                       nullptr,
                       nullptr,
                       error_message)) {
        return false;
    }

    if (logits_cpu != nullptr && !DownloadDeviceFloatTensor(*logits_tensor, logits_cpu, error_message)) {
        return false;
    }
    return true;
}

Tensor ForwardPromptHidden(const Qwen3_5ArchitectureSpec& spec,
                           const Qwen3_5HostWeights& host_weights,
                           const Qwen3_5Weights& device_weights,
                           const MetalContext& context,
                           PipelineCache* pipeline_cache,
                           BufferArena* temporary_arena,
                           const std::vector<int32_t>& token_ids,
                           Qwen3_5DecodeRuntimeState* decode_state) {
    Require(!token_ids.empty(), "Qwen3.5 prompt forward requires at least one token");
    Tensor hidden_states;
    std::string gpu_embedding_error;
    if (!(UseExperimentalQwen3_5EmbeddingGpu() &&
          ForwardEmbeddingGpu(context,
                              pipeline_cache,
                              temporary_arena,
                              device_weights.embed_tokens_weight,
                              spec.vocab_size,
                              spec.hidden_size,
                              token_ids,
                              &hidden_states,
                              &gpu_embedding_error))) {
        hidden_states = ::Embedding(host_weights.embed_tokens_weight).Forward(MakeIntTensor(token_ids, {1, token_ids.size()}));
    }

    for (std::size_t layer_index = 0; layer_index < spec.num_hidden_layers; ++layer_index) {
        const Qwen3_5HostBlockWeights& block = host_weights.blocks[layer_index];
        const Tensor normed_attention_input =
            Qwen3_5RmsNorm(hidden_states, block.input_layernorm_weight, spec.rms_norm_eps);

        Tensor attention_or_linear_output;
        if (spec.layer_types[layer_index] == Qwen3_5LayerType::kGatedAttention) {
            attention_or_linear_output =
                ForwardGatedAttentionPrompt(spec,
                                           device_weights.blocks[layer_index].attention,
                                           block.attention,
                                           context,
                                           pipeline_cache,
                                           temporary_arena,
                                           normed_attention_input,
                                           0,
                                           decode_state != nullptr ? &decode_state->attention_layers[layer_index] : nullptr);
        } else {
            attention_or_linear_output =
                ForwardGatedDeltaNetPrompt(spec,
                                          device_weights.blocks[layer_index].linear,
                                          block.linear,
                                          context,
                                          pipeline_cache,
                                          temporary_arena,
                                          normed_attention_input,
                                          decode_state != nullptr ? &decode_state->deltanet_layers[layer_index] : nullptr);
        }

        const Tensor residual_after_attention = AddTensors(hidden_states, attention_or_linear_output);
        const Tensor normed_mlp_input =
            Qwen3_5RmsNorm(residual_after_attention, block.post_attention_layernorm_weight, spec.rms_norm_eps);
        Tensor mlp_output;
        std::string gpu_mlp_error;
        if (!ForwardQwen3_5MlpGpu(context,
                                  pipeline_cache,
                                  temporary_arena,
                                  device_weights.blocks[layer_index].mlp,
                                  block.mlp,
                                  spec,
                                  normed_mlp_input,
                                  &mlp_output,
                                  &gpu_mlp_error)) {
            if (DebugGpuPath() && !gpu_mlp_error.empty()) {
                std::cerr << "qwen3_5 gpu mlp fallback layer " << layer_index << ": " << gpu_mlp_error << "\n";
            }
            const Tensor normed_mlp_input =
                Qwen3_5RmsNorm(residual_after_attention, block.post_attention_layernorm_weight, spec.rms_norm_eps);
            mlp_output = ForwardQwen3_5Mlp(block.mlp, normed_mlp_input);
        }
        hidden_states = AddTensors(residual_after_attention, mlp_output);
    }
    return hidden_states;
}

Tensor ForwardPromptLogits(const Qwen3_5ArchitectureSpec& spec,
                          const Qwen3_5HostWeights& host_weights,
                          const Qwen3_5Weights& device_weights,
                          const MetalContext& context,
                          PipelineCache* pipeline_cache,
                          BufferArena* temporary_arena,
                          const std::vector<int32_t>& token_ids) {
    const Tensor hidden_states =
        ForwardPromptHidden(spec, host_weights, device_weights, context, pipeline_cache, temporary_arena, token_ids, nullptr);
    const Tensor normalized = Qwen3_5RmsNorm(hidden_states, host_weights.final_norm_weight, spec.rms_norm_eps);
    DeviceTensor gpu_logits;
    const DeviceTensor& lm_head_weight = device_weights.tie_word_embeddings ? device_weights.embed_tokens_weight
                                                                            : device_weights.lm_head_weight;
    std::string gpu_lm_head_error;
    Tensor logits_cpu;
    if (ForwardLogitsFromHiddenGpu(context,
                                   pipeline_cache,
                                   lm_head_weight,
                                   normalized,
                                   spec.vocab_size,
                                   &gpu_logits,
                                   &logits_cpu,
                                   &gpu_lm_head_error)) {
        return logits_cpu;
    }
    ::Linear lm_head(host_weights.tie_word_embeddings ? host_weights.embed_tokens_weight
                                                      : host_weights.lm_head_weight);
    return lm_head.Forward(normalized);
}

Tensor ForwardGatedAttentionDecode(const Qwen3_5ArchitectureSpec& spec,
                                   const Qwen3_5GatedAttentionWeights& device_weights,
                                   const Qwen3_5HostGatedAttentionWeights& weights,
                                   const MetalContext& context,
                                   PipelineCache* pipeline_cache,
                                   BufferArena* temporary_arena,
                                   Qwen3_5AttentionDecodeCache* cache,
                                   const Tensor& hidden_state,
                                   const std::size_t position_offset) {
    Require(cache != nullptr, "attention decode cache must not be null");
    const std::size_t attention_dim = spec.num_attention_heads * spec.attention_head_dim;
    const bool use_decode_projection_gpu =
        UseExperimentalQwen3_5ProjectionGpu() || UseExperimentalQwen3_5SingleTokenProjectionGpu();
    const bool use_decode_output_projection_gpu =
        use_decode_projection_gpu || UseExperimentalQwen3_5DecodeOutputProjectionGpu();

    Tensor q_projected;
    std::string gpu_linear_error;
    if (use_decode_projection_gpu &&
        ForwardLinearGpuToCpu(context,
                              pipeline_cache,
                              temporary_arena,
                              device_weights.q_proj_weight,
                              hidden_state,
                              attention_dim * 2,
                              "Qwen3_5AttnQProjDecode",
                              &q_projected,
                              &gpu_linear_error)) {
    } else {
        q_projected = ::Linear(weights.q_proj_weight).Forward(hidden_state);
    }
    Tensor query = MakeFloatTensor({1, 1, spec.num_attention_heads, spec.attention_head_dim});
    Tensor gate = MakeFloatTensor({1, 1, spec.num_attention_heads, spec.attention_head_dim});
    const float* q_projected_data = q_projected.data<float>();
    float* query_data = query.data<float>();
    float* gate_data = gate.data<float>();
    for (std::size_t head = 0; head < spec.num_attention_heads; ++head) {
        const std::size_t source_base = head * spec.attention_head_dim * 2;
        const std::size_t destination_base = head * spec.attention_head_dim;
        std::memcpy(query_data + destination_base,
                    q_projected_data + source_base,
                    spec.attention_head_dim * sizeof(float));
        std::memcpy(gate_data + destination_base,
                    q_projected_data + source_base + spec.attention_head_dim,
                    spec.attention_head_dim * sizeof(float));
    }

    Tensor key_projected;
    if (use_decode_projection_gpu &&
        ForwardLinearGpuToCpu(context,
                              pipeline_cache,
                              temporary_arena,
                              device_weights.k_proj_weight,
                              hidden_state,
                              spec.num_key_value_heads * spec.attention_head_dim,
                              "Qwen3_5AttnKProjDecode",
                              &key_projected,
                              &gpu_linear_error)) {
    } else {
        key_projected = ::Linear(weights.k_proj_weight).Forward(hidden_state);
    }
    Tensor value_projected;
    if (use_decode_projection_gpu &&
        ForwardLinearGpuToCpu(context,
                              pipeline_cache,
                              temporary_arena,
                              device_weights.v_proj_weight,
                              hidden_state,
                              spec.num_key_value_heads * spec.attention_head_dim,
                              "Qwen3_5AttnVProjDecode",
                              &value_projected,
                              &gpu_linear_error)) {
    } else {
        value_projected = ::Linear(weights.v_proj_weight).Forward(hidden_state);
    }

    Tensor key = key_projected.Reshape({1, 1, spec.num_key_value_heads, spec.attention_head_dim});
    const Tensor value = value_projected.Reshape({1, 1, spec.num_key_value_heads, spec.attention_head_dim});
    query = Qwen3_5RmsNorm(query, weights.q_norm_weight, spec.rms_norm_eps);
    key = Qwen3_5RmsNorm(key, weights.k_norm_weight, spec.rms_norm_eps);
    query = ApplyPartialRoPE(query, spec.rotary_dim, spec.rope.rope_theta, position_offset);
    key = ApplyPartialRoPE(key, spec.rotary_dim, spec.rope.rope_theta, position_offset);

    const std::size_t kv_slice_size = spec.num_key_value_heads * spec.attention_head_dim;
    cache->key_values.resize((cache->sequence_length + 1) * kv_slice_size);
    cache->value_values.resize((cache->sequence_length + 1) * kv_slice_size);
    std::memcpy(cache->key_values.data() + cache->sequence_length * kv_slice_size,
                key.data<float>(),
                kv_slice_size * sizeof(float));
    std::memcpy(cache->value_values.data() + cache->sequence_length * kv_slice_size,
                value.data<float>(),
                kv_slice_size * sizeof(float));
    cache->sequence_length += 1;

    const Tensor cached_key =
        BuildAttentionCacheTensor(cache->key_values, cache->sequence_length, spec.num_key_value_heads, spec.attention_head_dim);
    const Tensor cached_value =
        BuildAttentionCacheTensor(cache->value_values, cache->sequence_length, spec.num_key_value_heads, spec.attention_head_dim);

    ::GroupedQueryAttention attention(spec.num_attention_heads,
                                      spec.num_key_value_heads,
                                      spec.attention_head_dim,
                                      true);
    Tensor attention_output = attention.Forward(query, cached_key, cached_value);
    attention_output = MultiplyTensors(attention_output, ApplySigmoid(gate));

    Tensor o_output;
    if (use_decode_output_projection_gpu &&
        ForwardLinearGpuToCpu(context,
                              pipeline_cache,
                              temporary_arena,
                              device_weights.o_proj_weight,
                              attention_output.Reshape({1, 1, attention_dim}),
                              spec.hidden_size,
                              "Qwen3_5AttnOProjDecode",
                              &o_output,
                              &gpu_linear_error)) {
        return o_output;
    }
    return ::Linear(weights.o_proj_weight).Forward(attention_output.Reshape({1, 1, attention_dim}));
}

Tensor ForwardGatedDeltaNetDecode(const Qwen3_5ArchitectureSpec& spec,
                                  const Qwen3_5GatedDeltaNetWeights& device_weights,
                                  const Qwen3_5HostGatedDeltaNetWeights& weights,
                                  const MetalContext& context,
                                  PipelineCache* pipeline_cache,
                                  BufferArena* temporary_arena,
                                  Qwen3_5DeltaDecodeCache* cache,
                                  const Tensor& hidden_state) {
    Require(cache != nullptr, "deltanet decode cache must not be null");
    const std::size_t key_dim = spec.linear_num_key_heads * spec.linear_key_head_dim;
    const std::size_t value_dim = spec.linear_num_value_heads * spec.linear_value_head_dim;
    const std::size_t channel_count = key_dim * 2 + value_dim;
    const bool use_decode_projection_gpu =
        UseExperimentalQwen3_5ProjectionGpu() || UseExperimentalQwen3_5SingleTokenProjectionGpu();
    const bool use_decode_output_projection_gpu =
        use_decode_projection_gpu || UseExperimentalQwen3_5DecodeOutputProjectionGpu();

    std::string gpu_linear_error;
    Tensor mixed_qkv;
    if (use_decode_projection_gpu &&
        ForwardLinearGpuToCpu(context,
                              pipeline_cache,
                              temporary_arena,
                              device_weights.in_proj_qkv_weight,
                              hidden_state,
                              channel_count,
                              "Qwen3_5DeltaInProjQkvDecode",
                              &mixed_qkv,
                              &gpu_linear_error)) {
    } else {
        mixed_qkv = ::Linear(weights.in_proj_qkv_weight).Forward(hidden_state);
    }

    Tensor conv_sequence =
        BuildDeltaConvSequence(cache, mixed_qkv.Reshape({1, channel_count}), channel_count, spec.linear_conv_kernel_dim);
    Tensor conv_output = CausalDepthwiseConv1dSiLU(conv_sequence, weights.conv1d_weight, spec.linear_conv_kernel_dim);
    UpdateDeltaConvHistory(cache, mixed_qkv.Reshape({1, channel_count}), channel_count, spec.linear_conv_kernel_dim);
    const Tensor current = SliceLastToken(conv_output);

    Tensor query = SliceLastDim(current, 0, key_dim).Reshape({1, 1, spec.linear_num_key_heads, spec.linear_key_head_dim});
    Tensor key = SliceLastDim(current, key_dim, key_dim).Reshape({1, 1, spec.linear_num_key_heads, spec.linear_key_head_dim});
    const Tensor value =
        SliceLastDim(current, key_dim * 2, value_dim).Reshape({1, 1, spec.linear_num_value_heads, spec.linear_value_head_dim});

    Tensor z_projected;
    if (use_decode_projection_gpu &&
        ForwardLinearGpuToCpu(context,
                              pipeline_cache,
                              temporary_arena,
                              device_weights.in_proj_z_weight,
                              hidden_state,
                              value_dim,
                              "Qwen3_5DeltaInProjZDecode",
                              &z_projected,
                              &gpu_linear_error)) {
    } else {
        z_projected = ::Linear(weights.in_proj_z_weight).Forward(hidden_state);
    }
    Tensor beta_projected;
    if (use_decode_projection_gpu &&
        ForwardLinearGpuToCpu(context,
                              pipeline_cache,
                              temporary_arena,
                              device_weights.in_proj_b_weight,
                              hidden_state,
                              spec.linear_num_value_heads,
                              "Qwen3_5DeltaInProjBDecode",
                              &beta_projected,
                              &gpu_linear_error)) {
    } else {
        beta_projected = ::Linear(weights.in_proj_b_weight).Forward(hidden_state);
    }
    Tensor a_projected;
    if (use_decode_projection_gpu &&
        ForwardLinearGpuToCpu(context,
                              pipeline_cache,
                              temporary_arena,
                              device_weights.in_proj_a_weight,
                              hidden_state,
                              spec.linear_num_value_heads,
                              "Qwen3_5DeltaInProjADecode",
                              &a_projected,
                              &gpu_linear_error)) {
    } else {
        a_projected = ::Linear(weights.in_proj_a_weight).Forward(hidden_state);
    }

    const Tensor z = z_projected.Reshape({1, 1, spec.linear_num_value_heads, spec.linear_value_head_dim});
    const Tensor beta = ApplySigmoid(beta_projected.Reshape({1, 1, spec.linear_num_value_heads}));
    const Tensor g = ComputeDeltaDecay(a_projected.Reshape({1, 1, spec.linear_num_value_heads}), weights.a_log, weights.dt_bias);

    query = L2NormalizeLastDim(query);
    key = L2NormalizeLastDim(key);
    {
        float* query_data = query.data<float>();
        const float query_scale = 1.0f / std::sqrt(static_cast<float>(spec.linear_key_head_dim));
        for (std::size_t index = 0; index < query.numel(); ++index) {
            query_data[index] *= query_scale;
        }
    }

    const std::size_t repeat_factor = spec.linear_num_value_heads / spec.linear_num_key_heads;
    query = RepeatInterleaveHeads(query, repeat_factor);
    key = RepeatInterleaveHeads(key, repeat_factor);

    Tensor core = MakeFloatTensor({1, 1, spec.linear_num_value_heads, spec.linear_value_head_dim});
    const float* query_data = query.data<float>();
    const float* key_data = key.data<float>();
    const float* value_data = value.data<float>();
    const float* beta_data = beta.data<float>();
    const float* g_data = g.data<float>();
    float* core_data = core.data<float>();

    const std::size_t state_rows = spec.linear_key_head_dim;
    const std::size_t state_cols = spec.linear_value_head_dim;
    const std::size_t state_count = spec.linear_num_value_heads;
    for (std::size_t head = 0; head < state_count; ++head) {
        const std::size_t qkv_base = head * state_rows;
        const std::size_t value_base = head * state_cols;
        const float* q_vector = query_data + qkv_base;
        const float* k_vector = key_data + qkv_base;
        const float* v_vector = value_data + value_base;
        const float beta_value = beta_data[head];
        const float decay = std::exp(g_data[head]);
        float* state = cache->recurrent_state.data() + head * state_rows * state_cols;

        for (std::size_t index = 0; index < state_rows * state_cols; ++index) {
            state[index] *= decay;
        }

        std::vector<float> kv_mem(state_cols, 0.0f);
        for (std::size_t row = 0; row < state_rows; ++row) {
            const float key_value = k_vector[row];
            const float* state_row = state + row * state_cols;
            for (std::size_t col = 0; col < state_cols; ++col) {
                kv_mem[col] += state_row[col] * key_value;
            }
        }

        std::vector<float> delta(state_cols, 0.0f);
        for (std::size_t col = 0; col < state_cols; ++col) {
            delta[col] = (v_vector[col] - kv_mem[col]) * beta_value;
        }
        for (std::size_t row = 0; row < state_rows; ++row) {
            float* state_row = state + row * state_cols;
            const float key_value = k_vector[row];
            for (std::size_t col = 0; col < state_cols; ++col) {
                state_row[col] += key_value * delta[col];
            }
        }

        float* output_vector = core_data + value_base;
        std::fill(output_vector, output_vector + state_cols, 0.0f);
        for (std::size_t row = 0; row < state_rows; ++row) {
            const float query_value = q_vector[row];
            const float* state_row = state + row * state_cols;
            for (std::size_t col = 0; col < state_cols; ++col) {
                output_vector[col] += state_row[col] * query_value;
            }
        }
    }

    Tensor gated = Qwen3_5RmsNormGated(core.Reshape({state_count, spec.linear_value_head_dim}),
                                       z.Reshape({state_count, spec.linear_value_head_dim}),
                                       weights.norm_weight,
                                       spec.rms_norm_eps);
    Tensor out_output;
    if (use_decode_output_projection_gpu &&
        ForwardLinearGpuToCpu(context,
                              pipeline_cache,
                              temporary_arena,
                              device_weights.out_proj_weight,
                              gated.Reshape({1, 1, value_dim}),
                              spec.hidden_size,
                              "Qwen3_5DeltaOutProjDecode",
                              &out_output,
                              &gpu_linear_error)) {
        return out_output;
    }
    return ::Linear(weights.out_proj_weight).Forward(gated.Reshape({1, 1, value_dim}));
}

bool RunDecodeTokenStep(const Qwen3_5ArchitectureSpec& spec,
                        const Qwen3_5HostWeights& host_weights,
                        const Qwen3_5Weights& device_weights,
                        const MetalContext& context,
                        PipelineCache* pipeline_cache,
                        BufferArena* temporary_arena,
                        Qwen3_5DecodeRuntimeState* decode_state,
                        const int32_t token_id,
                        const std::size_t position_offset,
                        Tensor* logits_output,
                        std::string* error_message) {
    if (decode_state == nullptr || logits_output == nullptr) {
        if (error_message != nullptr) {
            *error_message = "qwen3_5 decode step requires non-null outputs";
        }
        return false;
    }

    Tensor hidden_states;
    std::string gpu_embedding_error;
    if (!((UseExperimentalQwen3_5EmbeddingGpu() || UseExperimentalQwen3_5DecodeEmbeddingGpu()) &&
          ForwardEmbeddingGpu(context,
                              pipeline_cache,
                              temporary_arena,
                              device_weights.embed_tokens_weight,
                              spec.vocab_size,
                              spec.hidden_size,
                              {token_id},
                              &hidden_states,
                              &gpu_embedding_error))) {
        if (DebugGpuPath() && !gpu_embedding_error.empty()) {
            std::cerr << "qwen3_5 gpu embedding fallback decode: " << gpu_embedding_error << "\n";
        }
        hidden_states = ::Embedding(host_weights.embed_tokens_weight).Forward(MakeIntTensor({token_id}, {1, 1}));
    }
    for (std::size_t layer_index = 0; layer_index < spec.num_hidden_layers; ++layer_index) {
        const Qwen3_5HostBlockWeights& block = host_weights.blocks[layer_index];
        const Tensor normed_attention_input =
            Qwen3_5RmsNorm(hidden_states, block.input_layernorm_weight, spec.rms_norm_eps);

        Tensor attention_or_linear_output;
        if (spec.layer_types[layer_index] == Qwen3_5LayerType::kGatedAttention) {
            attention_or_linear_output =
                ForwardGatedAttentionDecode(spec,
                                            device_weights.blocks[layer_index].attention,
                                            block.attention,
                                            context,
                                            pipeline_cache,
                                            temporary_arena,
                                            &decode_state->attention_layers[layer_index],
                                            normed_attention_input,
                                            position_offset);
        } else {
            attention_or_linear_output =
                ForwardGatedDeltaNetDecode(spec,
                                           device_weights.blocks[layer_index].linear,
                                           block.linear,
                                           context,
                                           pipeline_cache,
                                           temporary_arena,
                                           &decode_state->deltanet_layers[layer_index],
                                           normed_attention_input);
        }

        const Tensor residual_after_attention = AddTensors(hidden_states, attention_or_linear_output);
        const Tensor normed_mlp_input =
            Qwen3_5RmsNorm(residual_after_attention, block.post_attention_layernorm_weight, spec.rms_norm_eps);
        Tensor mlp_output;
        std::string gpu_mlp_error;
        if (!ForwardQwen3_5MlpGpu(context,
                                  pipeline_cache,
                                  temporary_arena,
                                  device_weights.blocks[layer_index].mlp,
                                  block.mlp,
                                  spec,
                                  normed_mlp_input,
                                  &mlp_output,
                                  &gpu_mlp_error)) {
            if (DebugGpuPath() && !gpu_mlp_error.empty()) {
                std::cerr << "qwen3_5 gpu mlp fallback layer " << layer_index << ": " << gpu_mlp_error << "\n";
            }
            mlp_output = ForwardQwen3_5Mlp(block.mlp, normed_mlp_input);
        }
        hidden_states = AddTensors(residual_after_attention, mlp_output);
    }

    const Tensor normalized = Qwen3_5RmsNorm(hidden_states, host_weights.final_norm_weight, spec.rms_norm_eps);
    DeviceTensor gpu_logits;
    const DeviceTensor& lm_head_weight = device_weights.tie_word_embeddings ? device_weights.embed_tokens_weight
                                                                            : device_weights.lm_head_weight;
    std::string gpu_lm_head_error;
    if (ForwardLogitsFromHiddenGpu(context,
                                   pipeline_cache,
                                   lm_head_weight,
                                   normalized,
                                   spec.vocab_size,
                                   &gpu_logits,
                                   logits_output,
                                   &gpu_lm_head_error)) {
        return true;
    }
    if (DebugGpuPath() && !gpu_lm_head_error.empty()) {
        std::cerr << "qwen3_5 gpu lm_head fallback decode: " << gpu_lm_head_error << "\n";
    }
    *logits_output = ::Linear(host_weights.tie_word_embeddings ? host_weights.embed_tokens_weight
                                                               : host_weights.lm_head_weight)
                         .Forward(normalized);
    return true;
}

bool ReadTokenIds(const DeviceTensor& token_ids, std::vector<int32_t>* values, std::string* error_message) {
    if (values == nullptr) {
        if (error_message != nullptr) {
            *error_message = "token id output must not be null";
        }
        return false;
    }
    if (!token_ids.IsValid() || token_ids.GetDesc().GetDataType() != DataType::kInt32 || token_ids.GetDesc().Rank() != 1) {
        if (error_message != nullptr) {
            *error_message = "Qwen3.5 currently expects rank-1 int32 token ids";
        }
        return false;
    }
    values->resize(token_ids.GetDesc().GetShape()[0]);
    return token_ids.GetBuffer()->Read(values->data(),
                                       values->size() * sizeof(int32_t),
                                       token_ids.GetByteOffset(),
                                       error_message);
}

bool ReadFloatTensor2D(const DeviceTensor& tensor, Tensor* output, std::string* error_message) {
    if (output == nullptr) {
        if (error_message != nullptr) {
            *error_message = "float tensor output must not be null";
        }
        return false;
    }
    if (!tensor.IsValid() || tensor.GetDesc().GetDataType() != DataType::kFloat32 || tensor.GetDesc().Rank() != 2) {
        if (error_message != nullptr) {
            *error_message = "expected rank-2 float32 device tensor";
        }
        return false;
    }
    const std::size_t row_count = tensor.GetDesc().GetShape()[0];
    const std::size_t column_count = tensor.GetDesc().GetShape()[1];
    Tensor cpu_tensor = MakeFloatTensor({1, row_count, column_count});
    if (!tensor.GetBuffer()->Read(cpu_tensor.data<float>(),
                                  row_count * column_count * sizeof(float),
                                  tensor.GetByteOffset(),
                                  error_message)) {
        return false;
    }
    *output = std::move(cpu_tensor);
    return true;
}

bool WriteFloatTensor2D(const Tensor& tensor, const DeviceTensor& output, std::string* error_message) {
    Require(tensor.dtype() == DType::Float32, "write expects float32 cpu tensor");
    Require(tensor.dim() == 3 || tensor.dim() == 2, "write expects rank-2 or rank-3 cpu tensor");
    Tensor flattened = tensor.dim() == 2 ? tensor : tensor.Reshape({tensor.shape()[1], tensor.shape()[2]});
    if (!output.IsValid() || output.GetDesc().GetDataType() != DataType::kFloat32 || output.GetDesc().Rank() != 2) {
        if (error_message != nullptr) {
            *error_message = "Qwen3.5 output must be a rank-2 float32 device tensor";
        }
        return false;
    }
    if (flattened.shape()[0] != output.GetDesc().GetShape()[0] ||
        flattened.shape()[1] != output.GetDesc().GetShape()[1]) {
        if (error_message != nullptr) {
            *error_message = "Qwen3.5 output tensor shape mismatch";
        }
        return false;
    }
    return output.GetBuffer()->Write(flattened.data<float>(),
                                     flattened.numel() * sizeof(float),
                                     output.GetByteOffset(),
                                     error_message);
}

}  // namespace

Qwen3_5Runner::Qwen3_5Runner(Qwen3_5ArchitectureSpec spec)
    : spec_(std::move(spec)), state_layout_(BuildStateLayout(spec_)) {
    ResetDecodeRuntimeState(spec_, &decode_state_);
}

Qwen3_5Runner::Qwen3_5Runner(Qwen3_5ArchitectureSpec spec,
                             Qwen3_5Weights weights,
                             Qwen3_5HostWeights host_weights,
                             Qwen3_5StateLayout state_layout)
    : spec_(std::move(spec)),
      weights_(std::move(weights)),
      host_weights_(std::move(host_weights)),
      state_layout_(std::move(state_layout)) {
    ResetDecodeRuntimeState(spec_, &decode_state_);
}

std::size_t Qwen3_5Runner::num_layers() const { return spec_.num_hidden_layers; }
std::size_t Qwen3_5Runner::hidden_size() const { return spec_.hidden_size; }
std::size_t Qwen3_5Runner::num_key_value_heads() const { return spec_.num_key_value_heads; }
std::size_t Qwen3_5Runner::head_dim() const { return spec_.attention_head_dim; }
std::size_t Qwen3_5Runner::vocab_size() const { return spec_.vocab_size; }
std::size_t Qwen3_5Runner::max_position_embeddings() const { return spec_.max_position_embeddings; }

bool Qwen3_5Runner::ForwardLogitsCached(const MetalContext& context,
                                        PipelineCache* pipeline_cache,
                                        const DeviceTensor& token_ids,
                                        KVCache*,
                                        const DeviceTensor& logits_output,
                                        BufferArena* temporary_arena,
                                        const std::size_t position_offset,
                                        std::string* error_message) const {
    if (!has_loaded_weights()) {
        if (error_message != nullptr) {
            *error_message = "Qwen3.5 weights are not loaded";
        }
        return false;
    }

    try {
        std::vector<int32_t> prompt_token_ids;
        if (!ReadTokenIds(token_ids, &prompt_token_ids, error_message)) {
            return false;
        }
        if (position_offset == 0) {
            ResetDecodeRuntimeState(spec_, &decode_state_);
        } else if (!decode_state_.ready || decode_state_.cached_sequence_length != position_offset) {
            if (error_message != nullptr) {
                *error_message = "Qwen3.5 decode state is not initialized for the requested position offset";
            }
            return false;
        }

        if (position_offset == 0 && prompt_token_ids.size() > 1) {
            ResetDecodeRuntimeState(spec_, &decode_state_);
            const Tensor prompt_logits =
                [&]() {
                    const Tensor hidden_states =
                        ForwardPromptHidden(spec_,
                                            host_weights_,
                                            weights_,
                                            context,
                                            pipeline_cache,
                                            temporary_arena,
                                            prompt_token_ids,
                                            &decode_state_);
                    const Tensor normalized =
                        Qwen3_5RmsNorm(hidden_states, host_weights_.final_norm_weight, spec_.rms_norm_eps);
                    DeviceTensor gpu_logits;
                    const DeviceTensor& lm_head_weight = weights_.tie_word_embeddings ? weights_.embed_tokens_weight
                                                                                      : weights_.lm_head_weight;
                    std::string gpu_lm_head_error;
                    Tensor logits_cpu;
                    if (ForwardLogitsFromHiddenGpu(context,
                                                   pipeline_cache,
                                                   lm_head_weight,
                                                   normalized,
                                                   spec_.vocab_size,
                                                   &gpu_logits,
                                                   &logits_cpu,
                                                   &gpu_lm_head_error)) {
                        return logits_cpu;
                    }
                    ::Linear lm_head(host_weights_.tie_word_embeddings ? host_weights_.embed_tokens_weight
                                                                      : host_weights_.lm_head_weight);
                    return lm_head.Forward(normalized);
                }();
            if (!WriteFloatTensor2D(prompt_logits, logits_output, error_message)) {
                return false;
            }
            decode_state_.token_ids.assign(prompt_token_ids.begin(), prompt_token_ids.end());
            decode_state_.cached_sequence_length = prompt_token_ids.size();
            decode_state_.ready = true;
            return true;
        }

        Tensor logits_rows = MakeFloatTensor({1, prompt_token_ids.size(), spec_.vocab_size});
        float* logits_rows_data = logits_rows.data<float>();
        for (std::size_t token_index = 0; token_index < prompt_token_ids.size(); ++token_index) {
            Tensor row_logits;
            if (!RunDecodeTokenStep(spec_,
                                    host_weights_,
                                    weights_,
                                    context,
                                    pipeline_cache,
                                    temporary_arena,
                                    &decode_state_,
                                    prompt_token_ids[token_index],
                                    position_offset + token_index,
                                    &row_logits,
                                    error_message)) {
                return false;
            }
            const Tensor flat_row = row_logits.dim() == 2 ? row_logits : row_logits.Reshape({1, spec_.vocab_size});
            std::memcpy(logits_rows_data + token_index * spec_.vocab_size,
                        flat_row.data<float>(),
                        spec_.vocab_size * sizeof(float));
            decode_state_.token_ids.push_back(prompt_token_ids[token_index]);
            decode_state_.cached_sequence_length += 1;
        }
        decode_state_.ready = true;
        return WriteFloatTensor2D(logits_rows, logits_output, error_message);
    } catch (const std::exception& error) {
        if (error_message != nullptr) {
            *error_message = error.what();
        }
        return false;
    }
}

bool Qwen3_5Runner::ForwardLogitsFromHidden(const MetalContext& context,
                                            PipelineCache* pipeline_cache,
                                            const DeviceTensor& hidden_states,
                                            const DeviceTensor& logits_output,
                                            BufferArena*,
                                            std::string* error_message) const {
    if (!has_loaded_weights()) {
        if (error_message != nullptr) {
            *error_message = "Qwen3.5 weights are not loaded";
        }
        return false;
    }
    try {
        Tensor cpu_hidden;
        if (hidden_states.IsValid()) {
            if (!ReadFloatTensor2D(hidden_states, &cpu_hidden, error_message)) {
                return false;
            }
        } else {
            if (error_message != nullptr) {
                *error_message = "Qwen3.5 hidden-to-logits path requires valid hidden states";
            }
            return false;
        }

        DeviceTensor gpu_logits;
        const DeviceTensor& lm_head_weight = weights_.tie_word_embeddings ? weights_.embed_tokens_weight
                                                                          : weights_.lm_head_weight;
        const Tensor normalized =
            Qwen3_5RmsNorm(cpu_hidden, host_weights_.final_norm_weight, spec_.rms_norm_eps);
        std::string gpu_lm_head_error;
        if (ForwardLogitsFromHiddenGpu(context,
                                       pipeline_cache,
                                       lm_head_weight,
                                       normalized,
                                       spec_.vocab_size,
                                       &gpu_logits,
                                       nullptr,
                                       &gpu_lm_head_error)) {
            std::vector<float> logits_bytes(gpu_logits.GetDesc().ElementCount(), 0.0f);
            if (!gpu_logits.GetBuffer()->Read(logits_bytes.data(),
                                              logits_bytes.size() * sizeof(float),
                                              gpu_logits.GetByteOffset(),
                                              error_message)) {
                return false;
            }
            return logits_output.GetBuffer()->Write(logits_bytes.data(),
                                                    logits_bytes.size() * sizeof(float),
                                                    logits_output.GetByteOffset(),
                                                    error_message);
        }
        if (DebugGpuPath() && !gpu_lm_head_error.empty()) {
            std::cerr << "qwen3_5 gpu hidden->lm_head fallback: " << gpu_lm_head_error << "\n";
        }
        ::Linear lm_head(host_weights_.tie_word_embeddings ? host_weights_.embed_tokens_weight
                                                           : host_weights_.lm_head_weight);
        const Tensor logits = lm_head.Forward(normalized);
        return WriteFloatTensor2D(logits, logits_output, error_message);
    } catch (const std::exception& error) {
        if (error_message != nullptr) {
            *error_message = error.what();
        }
        return false;
    }
}

bool Qwen3_5Runner::DebugBoundaryProbe(const MetalContext& context,
                                       PipelineCache* pipeline_cache,
                                       BufferArena* temporary_arena,
                                       const std::vector<int32_t>& prompt_token_ids,
                                       const std::size_t top_k,
                                       Qwen3_5BoundaryProbe* output,
                                       std::string* error_message) const {
    if (output == nullptr) {
        if (error_message != nullptr) {
            *error_message = "Qwen3.5 boundary probe output must not be null";
        }
        return false;
    }
    if (!has_loaded_weights()) {
        if (error_message != nullptr) {
            *error_message = "Qwen3.5 weights are not loaded";
        }
        return false;
    }
    if (prompt_token_ids.empty()) {
        if (error_message != nullptr) {
            *error_message = "Qwen3.5 boundary probe requires at least one prompt token";
        }
        return false;
    }

    try {
        *output = {};
        ResetDecodeRuntimeState(spec_, &decode_state_);

        const Tensor prompt_logits =
            ForwardPromptLogits(spec_, host_weights_, weights_, context, pipeline_cache, temporary_arena, prompt_token_ids);
        const Tensor prompt_last_row = SliceLastLogitRow(prompt_logits);

        Tensor replay_last_logits;
        for (std::size_t token_index = 0; token_index < prompt_token_ids.size(); ++token_index) {
            if (!RunDecodeTokenStep(spec_,
                                    host_weights_,
                                    weights_,
                                    context,
                                    pipeline_cache,
                                    temporary_arena,
                                    &decode_state_,
                                    prompt_token_ids[token_index],
                                    token_index,
                                    &replay_last_logits,
                                    error_message)) {
                return false;
            }
            decode_state_.token_ids.push_back(prompt_token_ids[token_index]);
            decode_state_.cached_sequence_length += 1;
        }
        decode_state_.ready = true;

        output->full_prompt_argmax_id = ArgmaxTokenId(prompt_last_row);
        output->replay_warm_argmax_id = ArgmaxTokenId(replay_last_logits);
        output->full_prompt_top_logits = ComputeTopLogits(prompt_last_row, top_k);
        output->replay_warm_top_logits = ComputeTopLogits(replay_last_logits, top_k);
        const auto [max_abs_logit_diff, mean_abs_logit_diff] = LogitDiffStats(prompt_last_row, replay_last_logits);
        output->max_abs_logit_diff = max_abs_logit_diff;
        output->mean_abs_logit_diff = mean_abs_logit_diff;

        output->attention_cache_lengths.reserve(spec_.num_hidden_layers);
        output->deltanet_state_l2.reserve(spec_.num_hidden_layers);
        for (std::size_t layer_index = 0; layer_index < spec_.num_hidden_layers; ++layer_index) {
            if (spec_.layer_types[layer_index] == Qwen3_5LayerType::kGatedAttention) {
                output->attention_cache_lengths.push_back(decode_state_.attention_layers[layer_index].sequence_length);
                output->deltanet_state_l2.push_back(0.0f);
                continue;
            }
            output->attention_cache_lengths.push_back(0);
            const auto& recurrent_state = decode_state_.deltanet_layers[layer_index].recurrent_state;
            double sum_square = 0.0;
            for (float value : recurrent_state) {
                sum_square += static_cast<double>(value) * static_cast<double>(value);
            }
            output->deltanet_state_l2.push_back(static_cast<float>(std::sqrt(sum_square)));
        }
        return true;
    } catch (const std::exception& error) {
        if (error_message != nullptr) {
            *error_message = error.what();
        }
        return false;
    }
}

DecodePlanner* Qwen3_5Runner::GetDecodePlanner() { return nullptr; }
const DecodePlanner* Qwen3_5Runner::GetDecodePlanner() const { return nullptr; }

const Qwen3_5ArchitectureSpec& Qwen3_5Runner::spec() const { return spec_; }
const Qwen3_5Weights& Qwen3_5Runner::weights() const { return weights_; }
const Qwen3_5HostWeights& Qwen3_5Runner::host_weights() const { return host_weights_; }
const Qwen3_5StateLayout& Qwen3_5Runner::state_layout() const { return state_layout_; }
bool Qwen3_5Runner::has_loaded_weights() const {
    return weights_.embed_tokens_weight.IsValid() && host_weights_.embed_tokens_weight.storage().valid();
}

}  // namespace soc::gpu::models::qwen3_5
