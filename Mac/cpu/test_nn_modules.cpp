#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "header/dtype.h"
#include "header/embedding.h"
#include "header/generation_session.h"
#include "header/grouped_query_attention.h"
#include "header/kv_cache.h"
#include "header/linear.h"
#include "header/qwen_block.h"
#include "header/qwen_attention.h"
#include "header/qwen_causal_lm.h"
#include "header/qwen_mlp.h"
#include "header/rope.h"
#include "header/rms_norm_module.h"
#include "header/sampler.h"

namespace {
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

Tensor MakeFloatTensor(const std::vector<float>& values, const std::vector<std::size_t>& shape) {
    return Tensor(Storage::FromOwnedCopy(values.data(), values.size() * sizeof(float)), DType::Float32, shape);
}

Tensor MakeInt32Tensor(const std::vector<int32_t>& values, const std::vector<std::size_t>& shape) {
    return Tensor(Storage::FromOwnedCopy(values.data(), values.size() * sizeof(int32_t)), DType::Int32, shape);
}

Tensor MakeFloat16Tensor(const std::vector<float>& values, const std::vector<std::size_t>& shape) {
    std::vector<std::uint16_t> encoded(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        encoded[index] = Float32ToFloat16(values[index]);
    }
    return Tensor(Storage::FromOwnedCopy(encoded.data(), encoded.size() * sizeof(std::uint16_t)), DType::Float16, shape);
}

Tensor MakeBFloat16Tensor(const std::vector<float>& values, const std::vector<std::size_t>& shape) {
    std::vector<std::uint16_t> encoded(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        encoded[index] = Float32ToBFloat16(values[index]);
    }
    return Tensor(Storage::FromOwnedCopy(encoded.data(), encoded.size() * sizeof(std::uint16_t)), DType::BFloat16, shape);
}

bool NearlyEqual(float lhs, float rhs, float tolerance = 1e-5f) {
    return std::fabs(lhs - rhs) <= tolerance;
}

void test_embedding_module() {
    std::cout << "=== Testing Embedding Module ===" << std::endl;

    const Tensor weight = MakeFloatTensor({
        0.1f, 0.2f, 0.3f,
        1.0f, 1.1f, 1.2f,
        2.0f, 2.1f, 2.2f,
        3.0f, 3.1f, 3.2f,
    }, {4, 3});
    const Embedding embedding(weight);

    const Tensor token_ids = MakeInt32Tensor({2, 0, 3, 1}, {2, 2});
    const Tensor output = embedding.Forward(token_ids);

    require(output.dtype() == DType::Float32, "embedding output must be float32");
    require(output.shape() == std::vector<std::size_t>({2, 2, 3}), "embedding output shape must append embedding dimension");

    const float* output_data = output.data<const float>();
    require(NearlyEqual(output_data[0], 2.0f) && NearlyEqual(output_data[1], 2.1f) && NearlyEqual(output_data[2], 2.2f),
            "embedding must gather the requested token row");
    require(NearlyEqual(output_data[9], 1.0f) && NearlyEqual(output_data[10], 1.1f) && NearlyEqual(output_data[11], 1.2f),
            "embedding must preserve token order across batches");

    bool threw = false;
    try {
        static_cast<void>(embedding.Forward(MakeInt32Tensor({4}, {1})));
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw, "embedding must reject out-of-range token ids");
}

void test_embedding_mixed_precision_weights() {
    std::cout << "=== Testing Embedding Mixed Precision Weights ===" << std::endl;

    const Embedding embedding(MakeFloat16Tensor({
        0.5f, -1.0f,
        1.5f, 2.0f,
        -3.0f, 4.5f,
    }, {3, 2}));

    const Tensor output = embedding.Forward(MakeInt32Tensor({2, 1, 0}, {3}));
    const float* output_data = output.data<const float>();

    require(NearlyEqual(output_data[0], -3.0f, 1e-3f), "embedding must decode float16 row 2 col 0");
    require(NearlyEqual(output_data[1], 4.5f, 1e-3f), "embedding must decode float16 row 2 col 1");
    require(NearlyEqual(output_data[2], 1.5f, 1e-3f), "embedding must decode float16 row 1 col 0");
    require(NearlyEqual(output_data[5], -1.0f, 1e-3f), "embedding must decode float16 row 0 col 1");
}

void test_linear_module() {
    std::cout << "=== Testing Linear Module ===" << std::endl;

    const Tensor weight = MakeFloatTensor({
        1.0f, 2.0f, 3.0f,
        -1.0f, 0.5f, 4.0f,
    }, {2, 3});
    const Tensor bias = MakeFloatTensor({0.25f, -0.5f}, {2});
    const Linear linear(weight, bias);

    const Tensor input = MakeFloatTensor({
        1.0f, 0.0f, -1.0f,
        2.0f, 1.0f, 0.5f,
    }, {2, 3});
    const Tensor output = linear.Forward(input);

    require(output.shape() == std::vector<std::size_t>({2, 2}), "linear output shape must replace last dimension with output features");

    const float* output_data = output.data<const float>();
    require(NearlyEqual(output_data[0], -1.75f), "linear must apply matmul and bias for row 0 col 0");
    require(NearlyEqual(output_data[1], -5.5f), "linear must apply matmul and bias for row 0 col 1");
    require(NearlyEqual(output_data[2], 5.75f), "linear must apply matmul and bias for row 1 col 0");
    require(NearlyEqual(output_data[3], 0.0f), "linear must apply matmul and bias for row 1 col 1");

    bool threw = false;
    try {
        static_cast<void>(linear.Forward(MakeFloatTensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2})));
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw, "linear must reject mismatched input feature sizes");
}

void test_linear_mixed_precision_weights() {
    std::cout << "=== Testing Linear Mixed Precision Weights ===" << std::endl;

    const Linear linear(
        MakeBFloat16Tensor({
            1.0f, -0.5f,
            0.25f, 2.0f,
        }, {2, 2}),
        MakeFloat16Tensor({0.5f, -1.0f}, {2}));

    const Tensor input = MakeFloatTensor({2.0f, 4.0f}, {1, 2});
    const Tensor output = linear.Forward(input);
    const float* output_data = output.data<const float>();

    require(NearlyEqual(output_data[0], 0.5f, 5e-2f), "linear must accumulate bfloat16 weights with float16 bias for output 0");
    require(NearlyEqual(output_data[1], 7.5f, 5e-2f), "linear must accumulate bfloat16 weights with float16 bias for output 1");
}

void test_rms_norm_module() {
    std::cout << "=== Testing RMSNorm Module ===" << std::endl;

    const Tensor weight = MakeFloatTensor({1.0f, 1.5f, 0.5f}, {3});
    const RMSNormModule norm(weight, 1e-6f);

    const Tensor input = MakeFloatTensor({
        3.0f, 4.0f, 0.0f,
        1.0f, 2.0f, 2.0f,
    }, {2, 3});
    const Tensor output = norm.Forward(input);

    const float* output_data = output.data<const float>();
    const float row0_scale = 1.0f / std::sqrt((9.0f + 16.0f + 0.0f) / 3.0f + 1e-6f);
    const float row1_scale = 1.0f / std::sqrt((1.0f + 4.0f + 4.0f) / 3.0f + 1e-6f);

    require(NearlyEqual(output_data[0], 3.0f * row0_scale), "rms norm must normalize row 0 feature 0");
    require(NearlyEqual(output_data[1], 4.0f * row0_scale * 1.5f), "rms norm must apply scaling weights");
    require(NearlyEqual(output_data[3], 1.0f * row1_scale), "rms norm must normalize row 1 feature 0");
    require(NearlyEqual(output_data[5], 2.0f * row1_scale * 0.5f), "rms norm must normalize row 1 feature 2");

    bool threw = false;
    try {
        const RMSNormModule invalid(weight, 0.0f);
        static_cast<void>(invalid);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw, "rms norm must reject non-positive epsilon");
}

void test_rms_norm_mixed_precision_weight() {
    std::cout << "=== Testing RMSNorm Mixed Precision Weight ===" << std::endl;

    const RMSNormModule norm(MakeBFloat16Tensor({1.0f, 0.5f}, {2}), 1e-6f);
    const Tensor input = MakeFloatTensor({3.0f, 4.0f}, {1, 2});
    const Tensor output = norm.Forward(input);

    const float scale = 1.0f / std::sqrt((9.0f + 16.0f) / 2.0f + 1e-6f);
    const float* output_data = output.data<const float>();
    require(NearlyEqual(output_data[0], 3.0f * scale, 5e-2f), "rms norm must decode bfloat16 scale weight for feature 0");
    require(NearlyEqual(output_data[1], 4.0f * scale * 0.5f, 5e-2f), "rms norm must decode bfloat16 scale weight for feature 1");
}

void test_rope_module() {
    std::cout << "=== Testing RoPE Module ===" << std::endl;

    const RoPE rope(4, 10000.0);
    const Tensor input = MakeFloatTensor({
        1.0f, 2.0f, 10.0f, 20.0f,
        3.0f, 4.0f, 30.0f, 40.0f,
    }, {1, 2, 1, 4});

    const Tensor output = rope.Forward(input, 0);
    const float* output_data = output.data<const float>();

    require(NearlyEqual(output_data[0], 1.0f), "rope position 0 must preserve the first first-half feature");
    require(NearlyEqual(output_data[1], 2.0f), "rope position 0 must preserve the second first-half feature");
    require(NearlyEqual(output_data[2], 10.0f), "rope position 0 must preserve the first second-half feature");
    require(NearlyEqual(output_data[3], 20.0f), "rope position 0 must preserve the second second-half feature");

    const float angle0 = 1.0f;
    const float angle1 = 0.01f;
    const float expected4 = 3.0f * std::cos(angle0) - 30.0f * std::sin(angle0);
    const float expected5 = 4.0f * std::cos(angle1) - 40.0f * std::sin(angle1);
    const float expected6 = 3.0f * std::sin(angle0) + 30.0f * std::cos(angle0);
    const float expected7 = 4.0f * std::sin(angle1) + 40.0f * std::cos(angle1);
    require(NearlyEqual(output_data[4], expected4), "rope must rotate position 1 pair 0");
    require(NearlyEqual(output_data[5], expected5), "rope must rotate position 1 pair 1");
    require(NearlyEqual(output_data[6], expected6), "rope must rotate second-half pair 0");
    require(NearlyEqual(output_data[7], expected7), "rope must rotate second-half pair 1");

    Tensor inplace = MakeFloatTensor({
        5.0f, 6.0f, 7.0f, 8.0f,
    }, {1, 1, 1, 4});
    rope.ApplyInPlace(inplace, 3);
    const Tensor& inplace_view = inplace;
    const float* inplace_data = inplace_view.data<float>();
    const float offset_angle0 = 3.0f;
    const float offset_angle1 = 0.03f;
    require(NearlyEqual(inplace_data[0], 5.0f * std::cos(offset_angle0) - 7.0f * std::sin(offset_angle0)),
            "rope in-place path must apply position_offset to pair 0");
    require(NearlyEqual(inplace_data[1], 6.0f * std::cos(offset_angle1) - 8.0f * std::sin(offset_angle1)),
            "rope in-place path must apply position_offset to pair 1");

    bool threw = false;
    try {
        const RoPE invalid(3, 10000.0);
        static_cast<void>(invalid);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw, "rope must reject odd head_dim");
}

void test_grouped_query_attention_module() {
    std::cout << "=== Testing Grouped Query Attention Module ===" << std::endl;

    const GroupedQueryAttention attention(2, 1, 2, true);
    const Tensor query = MakeFloatTensor({
        1.0f, 0.0f, 0.0f, 1.0f,
        0.5f, 0.5f, 1.0f, 0.0f,
    }, {1, 2, 2, 2});
    const Tensor key = MakeFloatTensor({
        1.0f, 0.0f,
        0.0f, 1.0f,
    }, {1, 2, 1, 2});
    const Tensor value = MakeFloatTensor({
        10.0f, 1.0f,
        2.0f, 20.0f,
    }, {1, 2, 1, 2});

    const Tensor output = attention.Forward(query, key, value);
    require(output.shape() == std::vector<std::size_t>({1, 2, 2, 2}), "attention output must preserve query shape");

    const Tensor& output_view = output;
    const float* output_data = output_view.data<float>();
    require(NearlyEqual(output_data[0], 10.0f) && NearlyEqual(output_data[1], 1.0f),
            "causal attention first token head 0 must only see key 0");
    require(NearlyEqual(output_data[2], 10.0f) && NearlyEqual(output_data[3], 1.0f),
            "grouped query attention must share kv head 0 across query heads at token 0");

    const float scale = 1.0f / std::sqrt(2.0f);
    const float score00 = (0.5f * 1.0f + 0.5f * 0.0f) * scale;
    const float score01 = (0.5f * 0.0f + 0.5f * 1.0f) * scale;
    const float max0 = std::max(score00, score01);
    const float weight00 = std::exp(score00 - max0);
    const float weight01 = std::exp(score01 - max0);
    const float norm0 = weight00 + weight01;
    const float expected_head0_x = (weight00 * 10.0f + weight01 * 2.0f) / norm0;
    const float expected_head0_y = (weight00 * 1.0f + weight01 * 20.0f) / norm0;
    require(NearlyEqual(output_data[4], expected_head0_x), "attention must compute softmax-weighted output for token 1 head 0 x");
    require(NearlyEqual(output_data[5], expected_head0_y), "attention must compute softmax-weighted output for token 1 head 0 y");

    const float score10 = (1.0f * 1.0f + 0.0f * 0.0f) * scale;
    const float score11 = (1.0f * 0.0f + 0.0f * 1.0f) * scale;
    const float max1 = std::max(score10, score11);
    const float weight10 = std::exp(score10 - max1);
    const float weight11 = std::exp(score11 - max1);
    const float norm1 = weight10 + weight11;
    const float expected_head1_x = (weight10 * 10.0f + weight11 * 2.0f) / norm1;
    const float expected_head1_y = (weight10 * 1.0f + weight11 * 20.0f) / norm1;
    require(NearlyEqual(output_data[6], expected_head1_x), "attention must compute softmax-weighted output for token 1 head 1 x");
    require(NearlyEqual(output_data[7], expected_head1_y), "attention must compute softmax-weighted output for token 1 head 1 y");

    bool threw = false;
    try {
        const GroupedQueryAttention invalid(3, 2, 2, true);
        static_cast<void>(invalid);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw, "attention must reject query head counts that do not divide kv heads evenly");
}

void test_qwen_attention_module() {
    std::cout << "=== Testing QwenAttention Module ===" << std::endl;

    const Linear q_proj(MakeFloatTensor({
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    }, {4, 4}));
    const Linear k_proj(MakeFloatTensor({
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
    }, {2, 4}));
    const Linear v_proj(MakeFloatTensor({
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
    }, {2, 4}));
    const Linear o_proj(MakeFloatTensor({
        2.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 2.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 2.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 2.0f,
    }, {4, 4}));
    const QwenAttention attention(
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        RoPE(2, 10000.0),
        GroupedQueryAttention(2, 1, 2, true));

    const Tensor hidden_states = MakeFloatTensor({
        1.0f, 2.0f, 3.0f, 4.0f,
    }, {1, 1, 4});
    const Tensor output = attention.Forward(hidden_states, 0);

    require(output.shape() == std::vector<std::size_t>({1, 1, 4}), "qwen attention output must preserve [batch, seq, hidden]");
    const float* output_data = output.data<const float>();
    require(NearlyEqual(output_data[0], 2.0f), "qwen attention must project head 0 dim 0 through o_proj");
    require(NearlyEqual(output_data[1], 4.0f), "qwen attention must project head 0 dim 1 through o_proj");
    require(NearlyEqual(output_data[2], 2.0f), "qwen attention must broadcast kv output to head 1 dim 0");
    require(NearlyEqual(output_data[3], 4.0f), "qwen attention must broadcast kv output to head 1 dim 1");
}

void test_qwen_attention_cached_module() {
    std::cout << "=== Testing QwenAttention Cached Module ===" << std::endl;

    const QwenAttention attention(
        Linear(MakeFloatTensor({
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f,
        }, {4, 4})),
        Linear(MakeFloatTensor({
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
        }, {2, 4})),
        Linear(MakeFloatTensor({
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
        }, {2, 4})),
        Linear(MakeFloatTensor({
            2.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 2.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 2.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 2.0f,
        }, {4, 4})),
        RoPE(2, 10000.0),
        GroupedQueryAttention(2, 1, 2, true));

    const Tensor hidden_states = MakeFloatTensor({
        1.0f, 2.0f, 3.0f, 4.0f,
        2.0f, 1.0f, 0.0f, 1.0f,
    }, {1, 2, 4});
    const Tensor full_output = attention.Forward(hidden_states, 0);

    TensorKVCache cache(1, 1, 1, 2, 4);
    static_cast<void>(attention.ForwardCached(MakeFloatTensor({1.0f, 2.0f, 3.0f, 4.0f}, {1, 1, 4}), cache, 0, 0));
    const Tensor cached_output = attention.ForwardCached(MakeFloatTensor({2.0f, 1.0f, 0.0f, 1.0f}, {1, 1, 4}), cache, 0, 1);

    const float* full_data = full_output.data<const float>();
    const float* cached_data = cached_output.data<const float>();
    require(NearlyEqual(cached_data[0], full_data[4]), "cached attention must match full forward for token 1 dim 0");
    require(NearlyEqual(cached_data[1], full_data[5]), "cached attention must match full forward for token 1 dim 1");
    require(NearlyEqual(cached_data[2], full_data[6]), "cached attention must match full forward for token 1 dim 2");
    require(NearlyEqual(cached_data[3], full_data[7]), "cached attention must match full forward for token 1 dim 3");
}

void test_qwen_mlp_module() {
    std::cout << "=== Testing QwenMLP Module ===" << std::endl;

    const QwenMLP mlp(
        Linear(MakeFloatTensor({
            1.0f, 0.0f,
            0.0f, 1.0f,
        }, {2, 2})),
        Linear(MakeFloatTensor({
            2.0f, 0.0f,
            0.0f, 3.0f,
        }, {2, 2})),
        Linear(MakeFloatTensor({
            1.0f, 0.0f,
            0.0f, 1.0f,
        }, {2, 2})));

    const Tensor hidden_states = MakeFloatTensor({
        1.0f, -1.0f,
    }, {1, 2});
    const Tensor output = mlp.Forward(hidden_states);

    const float silu_pos = 1.0f / (1.0f + std::exp(-1.0f));
    const float silu_neg = -1.0f / (1.0f + std::exp(1.0f));
    const float* output_data = output.data<const float>();
    require(NearlyEqual(output_data[0], silu_pos * 2.0f), "qwen mlp must apply SiLU to gate path before Hadamard product");
    require(NearlyEqual(output_data[1], silu_neg * -3.0f), "qwen mlp must apply gated up path before down projection");
}

void test_tensor_kv_cache_module() {
    std::cout << "=== Testing Tensor KV Cache Module ===" << std::endl;

    TensorKVCache cache(2, 1, 1, 2, 4);
    require(cache.length(0) == 0, "new kv cache must start empty");

    cache.Write(0, 0, 0, MakeFloatTensor({1.0f, 2.0f}, {1, 2}), MakeFloatTensor({10.0f, 20.0f}, {1, 2}));
    cache.Write(0, 0, 1, MakeFloatTensor({3.0f, 4.0f}, {1, 2}), MakeFloatTensor({30.0f, 40.0f}, {1, 2}));
    cache.Write(1, 0, 0, MakeFloatTensor({5.0f, 6.0f}, {1, 2}), MakeFloatTensor({50.0f, 60.0f}, {1, 2}));
    require(cache.length(0) == 2, "kv cache length must advance to the furthest written position");

    const auto layer0 = cache.Read(0, 0, 0, 2);
    const Tensor& keys0 = layer0.first;
    const Tensor& values0 = layer0.second;
    require(keys0.shape() == std::vector<std::size_t>({2, 1, 2}), "kv cache read must return [seq, kv_heads, head_dim]");
    const float* keys0_data = keys0.data<float>();
    const float* values0_data = values0.data<float>();
    require(NearlyEqual(keys0_data[0], 1.0f) && NearlyEqual(keys0_data[1], 2.0f), "kv cache must preserve first written key vector");
    require(NearlyEqual(keys0_data[2], 3.0f) && NearlyEqual(keys0_data[3], 4.0f), "kv cache must preserve second written key vector");
    require(NearlyEqual(values0_data[0], 10.0f) && NearlyEqual(values0_data[1], 20.0f), "kv cache must preserve first written value vector");
    require(NearlyEqual(values0_data[2], 30.0f) && NearlyEqual(values0_data[3], 40.0f), "kv cache must preserve second written value vector");

    const auto layer1 = cache.Read(1, 0, 0, 1);
    const float* keys1_data = layer1.first.data<float>();
    require(NearlyEqual(keys1_data[0], 5.0f) && NearlyEqual(keys1_data[1], 6.0f), "kv cache must store data independently per layer");

    cache.ClearSequence(0);
    require(cache.length(0) == 0, "kv cache clear must reset the tracked sequence length");

    cache.Reserve(1, 6);
    require(cache.max_sequence_length() == 6, "kv cache reserve must update sequence capacity");
    require(cache.length(0) == 0, "kv cache reserve must reset cached lengths");

    bool threw = false;
    try {
        static_cast<void>(cache.Read(0, 0, 0, 1));
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw, "kv cache must reject reads past the tracked sequence length");
}

void test_qwen_block_module() {
    std::cout << "=== Testing QwenBlock Module ===" << std::endl;

    const QwenAttention attention(
        Linear(MakeFloatTensor(std::vector<float>(16, 0.0f), {4, 4})),
        Linear(MakeFloatTensor(std::vector<float>(8, 0.0f), {2, 4})),
        Linear(MakeFloatTensor(std::vector<float>(8, 0.0f), {2, 4})),
        Linear(MakeFloatTensor(std::vector<float>(16, 0.0f), {4, 4})),
        RoPE(2, 10000.0),
        GroupedQueryAttention(2, 1, 2, true));
    const QwenMLP mlp(
        Linear(MakeFloatTensor(std::vector<float>(16, 0.0f), {4, 4})),
        Linear(MakeFloatTensor(std::vector<float>(16, 0.0f), {4, 4})),
        Linear(MakeFloatTensor(std::vector<float>(16, 0.0f), {4, 4})));
    const QwenBlock block(
        RMSNormModule(MakeFloatTensor({1.0f, 1.0f, 1.0f, 1.0f}, {4})),
        attention,
        RMSNormModule(MakeFloatTensor({1.0f, 1.0f, 1.0f, 1.0f}, {4})),
        mlp);

    const Tensor hidden_states = MakeFloatTensor({
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
    }, {1, 2, 4});
    const Tensor output = block.Forward(hidden_states, 0);
    const float* output_data = output.data<const float>();
    const float* input_data = hidden_states.data<const float>();
    for (std::size_t index = 0; index < hidden_states.numel(); ++index) {
        require(NearlyEqual(output_data[index], input_data[index]), "qwen block with zero attention/mlp must preserve the residual input");
    }
}

void test_qwen_causal_lm_module() {
    std::cout << "=== Testing QwenCausalLM Module ===" << std::endl;

    const Embedding embedding(MakeFloatTensor({
        1.0f, 1.0f,
        2.0f, 2.0f,
    }, {2, 2}));
    const QwenCausalLM model(
        embedding,
        {},
        RMSNormModule(MakeFloatTensor({1.0f, 1.0f}, {2})));

    const Tensor token_ids = MakeInt32Tensor({0, 1}, {1, 2});
    const Tensor hidden = model.ForwardHidden(token_ids, 0);
    require(hidden.shape() == std::vector<std::size_t>({1, 2, 2}), "causal lm hidden output must have shape [batch, seq, hidden]");

    const Tensor logits = model.ForwardLogits(token_ids, 0);
    require(logits.shape() == std::vector<std::size_t>({1, 2, 2}), "causal lm logits must have shape [batch, seq, vocab]");
    const float* logits_data = logits.data<const float>();
    require(NearlyEqual(logits_data[0], 2.0f) && NearlyEqual(logits_data[1], 4.0f),
            "tied embedding logits must use embedding rows as lm head weights for token 0");
    require(NearlyEqual(logits_data[2], 2.0f) && NearlyEqual(logits_data[3], 4.0f),
            "tied embedding logits must use embedding rows as lm head weights for token 1");
}

void test_qwen_causal_lm_cached_module() {
    std::cout << "=== Testing QwenCausalLM Cached Module ===" << std::endl;

    const QwenAttention attention(
        Linear(MakeFloatTensor({
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f,
        }, {4, 4})),
        Linear(MakeFloatTensor({
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
        }, {2, 4})),
        Linear(MakeFloatTensor({
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
        }, {2, 4})),
        Linear(MakeFloatTensor({
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f,
        }, {4, 4})),
        RoPE(2, 10000.0),
        GroupedQueryAttention(2, 1, 2, true));
    const QwenMLP mlp(
        Linear(MakeFloatTensor(std::vector<float>(16, 0.0f), {4, 4})),
        Linear(MakeFloatTensor(std::vector<float>(16, 0.0f), {4, 4})),
        Linear(MakeFloatTensor(std::vector<float>(16, 0.0f), {4, 4})));
    const QwenBlock block(
        RMSNormModule(MakeFloatTensor({1.0f, 1.0f, 1.0f, 1.0f}, {4})),
        attention,
        RMSNormModule(MakeFloatTensor({1.0f, 1.0f, 1.0f, 1.0f}, {4})),
        mlp);
    const QwenCausalLM model(
        Embedding(MakeFloatTensor({
            1.0f, 2.0f, 3.0f, 4.0f,
            2.0f, 1.0f, 0.0f, 1.0f,
            0.5f, 0.5f, 0.5f, 0.5f,
        }, {3, 4})),
        {block},
        RMSNormModule(MakeFloatTensor({1.0f, 1.0f, 1.0f, 1.0f}, {4})));

    const Tensor full_logits = model.ForwardLogits(MakeInt32Tensor({0, 1}, {1, 2}), 0);
    TensorKVCache cache(1, 1, 1, 2, 4);
    static_cast<void>(model.ForwardLogitsCached(MakeInt32Tensor({0}, {1, 1}), cache, 0));
    const Tensor cached_logits = model.ForwardLogitsCached(MakeInt32Tensor({1}, {1, 1}), cache, 1);

    const float* full_data = full_logits.data<const float>();
    const float* cached_data = cached_logits.data<const float>();
    require(NearlyEqual(cached_data[0], full_data[3]), "cached causal lm must match full logits for token 1 vocab 0");
    require(NearlyEqual(cached_data[1], full_data[4]), "cached causal lm must match full logits for token 1 vocab 1");
    require(NearlyEqual(cached_data[2], full_data[5]), "cached causal lm must match full logits for token 1 vocab 2");
}

TokenizerRuntime BuildTestTokenizerRuntime() {
    TokenizerRuntimeData runtime;
    runtime.format = "soc.cpp.tokenizer_runtime";
    runtime.format_version = 1;
    runtime.tokenizer_class = "TestTokenizer";
    runtime.vocab_size = 4;
    runtime.model_max_length = 32;
    runtime.template_runtime = TemplateRuntimeData{
        "qwen3",
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "You are a helpful assistant.",
    };
    runtime.added_tokens = {
        {2, "A", false, false, false, false, true},
        {3, "B", false, false, false, false, true},
    };
    runtime.vocab = {
        {0, "!"},
        {1, "?"},
    };
    return TokenizerRuntime(runtime);
}

void test_sampler_module() {
    std::cout << "=== Testing Sampler Module ===" << std::endl;

    const Sampler sampler({1.0f, 2});
    const Tensor logits = MakeFloatTensor({
        0.1f, 3.0f, 2.0f,
    }, {1, 1, 3});
    const int sampled = sampler.SampleFromLogits(logits, 0, 0);
    require(sampled == 1, "sampler must select the highest logit within top_k candidates");
}

void test_generation_session_module() {
    std::cout << "=== Testing Generation Session Module ===" << std::endl;

    const Embedding embedding(MakeFloatTensor({
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f,
        -1.0f, 1.0f,
    }, {4, 2}));
    const Linear lm_head(
        MakeFloatTensor({
            0.0f, 0.0f,
            0.0f, 0.0f,
            1.0f, 0.0f,
            0.0f, 1.0f,
        }, {4, 2}),
        MakeFloatTensor({0.0f, 0.0f, 0.5f, -0.25f}, {4}));
    const QwenCausalLM model(
        embedding,
        {},
        RMSNormModule(MakeFloatTensor({1.0f, 1.0f}, {2})),
        lm_head);
    GenerationSession session(model, BuildTestTokenizerRuntime(), Sampler({1.0f, 4}));

    const std::vector<int> prefill = session.Prefill("A");
    require(prefill.size() == 1 && prefill[0] == 2, "generation session prefill must tokenize the prompt");

    const int next_token = session.DecodeNextToken(prefill);
    require(next_token == 2, "generation session decode must sample from the last-position logits");

    const GenerationResult generated = session.Generate("A", 2, -1);
    require(generated.prompt_token_ids.size() == 1, "generation session must preserve prompt token ids");
    require(generated.generated_token_ids.size() == 2, "generation session must generate the requested number of tokens when eos is disabled");
    require(generated.generated_token_ids[0] == 2 && generated.generated_token_ids[1] == 2,
            "generation session must append sampled tokens deterministically under greedy settings");
    require(generated.generated_text == "AA", "generation session must detokenize generated token ids");
}
}

int main() {
    test_embedding_module();
    test_embedding_mixed_precision_weights();
    test_linear_module();
    test_linear_mixed_precision_weights();
    test_rms_norm_module();
    test_rms_norm_mixed_precision_weight();
    test_rope_module();
    test_grouped_query_attention_module();
    test_qwen_attention_module();
    test_qwen_attention_cached_module();
    test_qwen_mlp_module();
    test_tensor_kv_cache_module();
    test_qwen_block_module();
    test_qwen_causal_lm_module();
    test_qwen_causal_lm_cached_module();
    test_sampler_module();
    test_generation_session_module();

    std::cout << "=== NN Module Tests Completed ===" << std::endl;
    return 0;
}