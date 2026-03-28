#include <iostream>
#include <string>
#include <vector>

#include "buffer/buffer_arena.h"
#include "buffer/metal_buffer.h"
#include "kernel/pipeline_cache.h"
#include "metal/metal_context.h"
#include "model/qwen_causal_lm.h"
#include "runtime/command_scheduler.h"
#include "runtime/generation_context.h"
#include "runtime/sampler.h"
#include "tensor/device_tensor.h"
#include "tensor/tensor_desc.h"

namespace {

soc::gpu::DeviceTensor MakeFloatTensor(const soc::gpu::MetalContext& context,
                                       const std::vector<std::size_t>& shape,
                                       const std::vector<float>& values,
                                       const std::string& label,
                                       std::string* error_message) {
    auto buffer = soc::gpu::MetalBuffer::CreateShared(context, values.size() * sizeof(float), label, error_message);
    if (buffer == nullptr) {
        return {};
    }
    if (!values.empty() && !buffer->Write(values.data(), values.size() * sizeof(float), 0, error_message)) {
        return {};
    }
    return soc::gpu::DeviceTensor(buffer,
                                  0,
                                  soc::gpu::TensorDesc::CreateContiguous(soc::gpu::DataType::kFloat32, shape));
}

soc::gpu::QwenCausalLM BuildIdentityModel(const soc::gpu::MetalContext& context, std::string* error_message) {
    const std::vector<float> embed = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    const std::vector<float> ones(4, 1.0f);
    const std::vector<float> zeros(16, 0.0f);

    soc::gpu::QwenAttentionWeights attention_weights;
    attention_weights.q_proj_weight = MakeFloatTensor(context, {4, 4}, zeros, "q_proj", error_message);
    attention_weights.k_proj_weight = MakeFloatTensor(context, {4, 4}, zeros, "k_proj", error_message);
    attention_weights.v_proj_weight = MakeFloatTensor(context, {4, 4}, zeros, "v_proj", error_message);
    attention_weights.o_proj_weight = MakeFloatTensor(context, {4, 4}, zeros, "o_proj", error_message);
    attention_weights.q_norm_weight = MakeFloatTensor(context, {4}, ones, "q_norm", error_message);
    attention_weights.k_norm_weight = MakeFloatTensor(context, {4}, ones, "k_norm", error_message);

    soc::gpu::QwenMlpWeights mlp_weights;
    mlp_weights.gate_proj_weight = MakeFloatTensor(context, {4, 4}, zeros, "gate_proj", error_message);
    mlp_weights.up_proj_weight = MakeFloatTensor(context, {4, 4}, zeros, "up_proj", error_message);
    mlp_weights.down_proj_weight = MakeFloatTensor(context, {4, 4}, zeros, "down_proj", error_message);

    soc::gpu::QwenBlockWeights block_weights;
    block_weights.input_layernorm_weight = MakeFloatTensor(context, {4}, ones, "input_ln", error_message);
    block_weights.attention = attention_weights;
    block_weights.post_attention_layernorm_weight = MakeFloatTensor(context, {4}, ones, "post_attn_ln", error_message);
    block_weights.mlp = mlp_weights;

    soc::gpu::QwenCausalLMWeights model_weights;
    model_weights.embed_tokens_weight = MakeFloatTensor(context, {4, 4}, embed, "embed", error_message);
    model_weights.blocks = {block_weights};
    model_weights.final_norm_weight = MakeFloatTensor(context, {4}, ones, "final_norm", error_message);
    model_weights.lm_head_weight = model_weights.embed_tokens_weight;
    model_weights.tie_word_embeddings = true;

    soc::gpu::QwenCausalLMParams params;
    params.vocab_size = 4;
    params.hidden_size = 4;
    params.num_hidden_layers = 1;
    params.num_attention_heads = 1;
    params.num_key_value_heads = 1;
    params.head_dim = 4;
    params.intermediate_size = 4;
    params.max_position_embeddings = 8;
    params.rms_norm_eps = 1.0e-6f;

    return soc::gpu::QwenCausalLM(std::move(model_weights), params);
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

    auto temporary_arena = soc::gpu::BufferArena::CreateShared(*context, 1 << 20, "temp", &error_message);
    if (temporary_arena == nullptr) {
        std::cerr << "failed to create temporary arena: " << error_message << '\n';
        return 1;
    }

    soc::gpu::QwenCausalLM model = BuildIdentityModel(*context, &error_message);
    if (!model.weights().embed_tokens_weight.IsValid() || !model.weights().final_norm_weight.IsValid()) {
        std::cerr << "failed to build identity model: " << error_message << '\n';
        return 1;
    }

    soc::gpu::PipelineCache pipeline_cache(*context);
    soc::gpu::GenerationContext generation_context(std::move(model),
                                                   soc::gpu::Sampler({1.0f, 2}),
                                                   soc::gpu::CommandScheduler(),
                                                   8);

    const std::vector<int> prompt_token_ids = {1, 2};
    std::vector<int> generated_token_ids;
    if (!generation_context.Generate(*context,
                                     &pipeline_cache,
                                     prompt_token_ids,
                                     3,
                                     -1,
                                     temporary_arena.get(),
                                     &generated_token_ids,
                                     &error_message)) {
        std::cerr << "generation failed: " << error_message << '\n';
        return 1;
    }

    if (generated_token_ids != std::vector<int>({2, 2, 2})) {
        std::cerr << "unexpected generated token sequence\n";
        return 1;
    }
    if (generation_context.prompt_token_ids() != prompt_token_ids) {
        std::cerr << "prompt token ids were not preserved\n";
        return 1;
    }
    if (generation_context.running_token_ids() != std::vector<int>({1, 2, 2, 2, 2})) {
        std::cerr << "running token ids do not reflect decode progression\n";
        return 1;
    }

    generation_context.Reset();
    if (!generation_context.prompt_token_ids().empty() || !generation_context.running_token_ids().empty()) {
        std::cerr << "reset did not clear generation state\n";
        return 1;
    }

    std::cout << "test_generation_context passed\n";
    return 0;
}