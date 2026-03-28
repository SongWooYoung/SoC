#include "header/qwen_attention.h"

#include <utility>
#include <vector>
#include <stdexcept>

namespace {
void Require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

Tensor SliceSequenceWindow(const Tensor& tensor, std::size_t start_position, std::size_t sequence_length) {
    const std::size_t batch_size = tensor.shape()[0];
    const std::size_t num_heads = tensor.shape()[2];
    const std::size_t head_dim = tensor.shape()[3];

    Storage output_storage = Storage::AllocateOwned(batch_size * sequence_length * num_heads * head_dim * sizeof(float));
    Tensor output(std::move(output_storage), DType::Float32, {batch_size, sequence_length, num_heads, head_dim});

    const float* input_data = tensor.data<const float>();
    float* output_data = output.data<float>();
    const std::size_t input_batch_stride = tensor.shape()[1] * num_heads * head_dim;
    const std::size_t output_batch_stride = sequence_length * num_heads * head_dim;
    const std::size_t sequence_stride = num_heads * head_dim;

    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        for (std::size_t position = 0; position < sequence_length; ++position) {
            const float* source = input_data + batch * input_batch_stride + (start_position + position) * sequence_stride;
            float* destination = output_data + batch * output_batch_stride + position * sequence_stride;
            for (std::size_t index = 0; index < sequence_stride; ++index) {
                destination[index] = source[index];
            }
        }
    }

    return output;
}

Tensor BuildCachedTensor(const TensorKVCache& cache, std::size_t layer_index, std::size_t batch_size, std::size_t total_length, std::size_t num_heads, std::size_t head_dim, bool read_keys) {
    Storage output_storage = Storage::AllocateOwned(batch_size * total_length * num_heads * head_dim * sizeof(float));
    Tensor output(std::move(output_storage), DType::Float32, {batch_size, total_length, num_heads, head_dim});
    float* output_data = output.data<float>();

    const std::size_t batch_stride = total_length * num_heads * head_dim;
    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        const auto cached_pair = cache.Read(layer_index, batch, 0, total_length);
        const Tensor& source_tensor = read_keys ? cached_pair.first : cached_pair.second;
        const float* source_data = source_tensor.data<float>();
        float* destination = output_data + batch * batch_stride;
        for (std::size_t index = 0; index < total_length * num_heads * head_dim; ++index) {
            destination[index] = source_data[index];
        }
    }

    return output;
}
}

QwenAttention::QwenAttention(
    Linear q_proj,
    Linear k_proj,
    Linear v_proj,
    Linear o_proj,
        RMSNormModule q_norm,
        RMSNormModule k_norm,
    RoPE rope,
    GroupedQueryAttention attention)
    : q_proj_(std::move(q_proj)),
      k_proj_(std::move(k_proj)),
      v_proj_(std::move(v_proj)),
      o_proj_(std::move(o_proj)),
            q_norm_(std::move(q_norm)),
            k_norm_(std::move(k_norm)),
      rope_(std::move(rope)),
      attention_(std::move(attention)) {
    Require(q_proj_.output_dim() == attention_.num_query_heads() * attention_.head_dim(),
            "q_proj output_dim must equal num_query_heads * head_dim");
    Require(k_proj_.output_dim() == attention_.num_key_value_heads() * attention_.head_dim(),
            "k_proj output_dim must equal num_key_value_heads * head_dim");
    Require(v_proj_.output_dim() == attention_.num_key_value_heads() * attention_.head_dim(),
            "v_proj output_dim must equal num_key_value_heads * head_dim");
    Require(o_proj_.input_dim() == attention_.num_query_heads() * attention_.head_dim(),
            "o_proj input_dim must equal num_query_heads * head_dim");
        Require(q_norm_.weight().shape()[0] == attention_.head_dim(), "q_norm hidden size must match attention head_dim");
        Require(k_norm_.weight().shape()[0] == attention_.head_dim(), "k_norm hidden size must match attention head_dim");
    Require(rope_.head_dim() == attention_.head_dim(), "rope head_dim must match attention head_dim");
}

std::size_t QwenAttention::num_query_heads() const {
    return attention_.num_query_heads();
}

std::size_t QwenAttention::num_key_value_heads() const {
    return attention_.num_key_value_heads();
}

std::size_t QwenAttention::head_dim() const {
    return attention_.head_dim();
}

Tensor QwenAttention::Forward(const Tensor& hidden_states, std::size_t position_offset) const {
    Require(hidden_states.dtype() == DType::Float32, "attention hidden_states must be float32");
    Require(hidden_states.dim() == 3, "attention hidden_states must have shape [batch, seq, hidden_size]");

    const std::size_t batch_size = hidden_states.shape()[0];
    const std::size_t sequence_length = hidden_states.shape()[1];

    Tensor query = q_proj_.Forward(hidden_states).Reshape({
        batch_size,
        sequence_length,
        attention_.num_query_heads(),
        attention_.head_dim(),
    });
    Tensor key = k_proj_.Forward(hidden_states).Reshape({
        batch_size,
        sequence_length,
        attention_.num_key_value_heads(),
        attention_.head_dim(),
    });
    Tensor value = v_proj_.Forward(hidden_states).Reshape({
        batch_size,
        sequence_length,
        attention_.num_key_value_heads(),
        attention_.head_dim(),
    });

    query = q_norm_.Forward(query);
    key = k_norm_.Forward(key);
    query = rope_.Forward(query, position_offset);
    key = rope_.Forward(key, position_offset);

    Tensor attention_output = attention_.Forward(query, key, value).Reshape({
        batch_size,
        sequence_length,
        attention_.num_query_heads() * attention_.head_dim(),
    });
    return o_proj_.Forward(attention_output);
}

Tensor QwenAttention::ForwardCached(const Tensor& hidden_states, TensorKVCache& cache, std::size_t layer_index, std::size_t position_offset) const {
    Require(hidden_states.dtype() == DType::Float32, "attention hidden_states must be float32");
    Require(hidden_states.dim() == 3, "attention hidden_states must have shape [batch, seq, hidden_size]");
    Require(layer_index < cache.num_layers(), "attention cache layer_index out of range");

    const std::size_t batch_size = hidden_states.shape()[0];
    const std::size_t sequence_length = hidden_states.shape()[1];
    Tensor query = q_proj_.Forward(hidden_states).Reshape({
        batch_size,
        sequence_length,
        attention_.num_query_heads(),
        attention_.head_dim(),
    });
    Tensor key = k_proj_.Forward(hidden_states).Reshape({
        batch_size,
        sequence_length,
        attention_.num_key_value_heads(),
        attention_.head_dim(),
    });
    Tensor value = v_proj_.Forward(hidden_states).Reshape({
        batch_size,
        sequence_length,
        attention_.num_key_value_heads(),
        attention_.head_dim(),
    });

    query = q_norm_.Forward(query);
    key = k_norm_.Forward(key);
    query = rope_.Forward(query, position_offset);
    key = rope_.Forward(key, position_offset);

    const Tensor& key_view = key;
    const Tensor& value_view = value;
    const float* key_data = key_view.data<float>();
    const float* value_data = value_view.data<float>();
    const std::size_t kv_heads = attention_.num_key_value_heads();
    const std::size_t head_dim = attention_.head_dim();
    const std::size_t token_stride = kv_heads * head_dim;
    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        for (std::size_t position = 0; position < sequence_length; ++position) {
            const std::size_t flat_offset = (batch * sequence_length + position) * token_stride;
            const Tensor key_token(
                Storage::FromOwnedCopy(key_data + flat_offset, token_stride * sizeof(float)),
                DType::Float32,
                {kv_heads, head_dim});
            const Tensor value_token(
                Storage::FromOwnedCopy(value_data + flat_offset, token_stride * sizeof(float)),
                DType::Float32,
                {kv_heads, head_dim});
            cache.Write(layer_index, batch, position_offset + position, key_token, value_token);
        }
    }

    const std::size_t total_length = position_offset + sequence_length;
    const Tensor cached_key = BuildCachedTensor(cache, layer_index, batch_size, total_length, kv_heads, head_dim, true);
    const Tensor cached_value = BuildCachedTensor(cache, layer_index, batch_size, total_length, kv_heads, head_dim, false);
    const Tensor query_window = sequence_length == query.shape()[1] ? query : SliceSequenceWindow(query, 0, sequence_length);
    Tensor attention_output = attention_.ForwardWithPrefix(query_window, cached_key, cached_value, position_offset).Reshape({
        batch_size,
        sequence_length,
        attention_.num_query_heads() * attention_.head_dim(),
    });
    return o_proj_.Forward(attention_output);
}