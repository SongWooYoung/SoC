#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <cstdint>
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

soc::gpu::QwenCausalLM BuildIdentityModel(const soc::gpu::MetalContext& context,
                                         std::size_t layer_count,
                                         std::string* error_message) {
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
    model_weights.blocks = std::vector<soc::gpu::QwenBlockWeights>(layer_count, block_weights);
    model_weights.final_norm_weight = MakeFloatTensor(context, {4}, ones, "final_norm", error_message);
    model_weights.lm_head_weight = model_weights.embed_tokens_weight;
    model_weights.tie_word_embeddings = true;

    soc::gpu::QwenCausalLMParams params;
    params.vocab_size = 4;
    params.hidden_size = 4;
    params.num_hidden_layers = layer_count;
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

    soc::gpu::QwenCausalLM model = BuildIdentityModel(*context, 1, &error_message);
    if (!model.weights().embed_tokens_weight.IsValid() || !model.weights().final_norm_weight.IsValid()) {
        std::cerr << "failed to build identity model: " << error_message << '\n';
        return 1;
    }

    soc::gpu::PipelineCache pipeline_cache(*context);
    const std::vector<float> sampler_input = {
        0.1f, 3.5f, 2.0f, 1.0f,
    };
    const soc::gpu::DeviceTensor sampler_logits = MakeFloatTensor(*context,
                                                                  {1, 4},
                                                                  sampler_input,
                                                                  "sampler_logits",
                                                                  &error_message);
    soc::gpu::Sampler sampler({1.0f, 3});
    int sampled_token_id = -1;
    std::vector<float> sampled_top_logits;
    std::vector<int> sampled_top_ids;
    if (!sampler.SampleFromLogits(*context,
                                  &pipeline_cache,
                                  sampler_logits,
                                  0,
                                  &sampled_token_id,
                                  &sampled_top_logits,
                                  &sampled_top_ids,
                                  temporary_arena.get(),
                                  &error_message)) {
        std::cerr << "sampler reduction failed: " << error_message << '\n';
        return 1;
    }
    if (sampled_token_id != 1 || sampled_top_ids != std::vector<int>({1, 2, 3}) || sampled_top_logits.size() != 3 || std::fabs(sampled_top_logits[0] - 3.5f) > 1.0e-5f) {
        std::cerr << "unexpected sampler top-k reduction result\n";
        return 1;
    }

    soc::gpu::GenerationContext generation_context(std::move(model),
                                                   soc::gpu::Sampler({1.0f, 2}),
                                                   soc::gpu::CommandScheduler(),
                                                   8);
    soc::gpu::QwenCausalLM stepped_model = BuildIdentityModel(*context, 1, &error_message);
    if (!stepped_model.weights().embed_tokens_weight.IsValid() || !stepped_model.weights().final_norm_weight.IsValid()) {
        std::cerr << "failed to build stepped identity model: " << error_message << '\n';
        return 1;
    }
    soc::gpu::GenerationContext stepped_generation_context(std::move(stepped_model),
                                                           soc::gpu::Sampler({1.0f, 2}),
                                                           soc::gpu::CommandScheduler(),
                                                           8,
                                                           1);

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
    std::vector<int> stepped_generated_token_ids;
    if (!stepped_generation_context.Generate(*context,
                                             &pipeline_cache,
                                             prompt_token_ids,
                                             3,
                                             -1,
                                             temporary_arena.get(),
                                             &stepped_generated_token_ids,
                                             &error_message)) {
        std::cerr << "stepped generation failed: " << error_message << '\n';
        return 1;
    }

    if (generated_token_ids != std::vector<int>({2, 2, 2})) {
        std::cerr << "unexpected generated token sequence\n";
        return 1;
    }
    if (stepped_generated_token_ids != generated_token_ids) {
        std::cerr << "stepped prefill changed generated token sequence\n";
        return 1;
    }
    if (generation_context.prompt_token_ids() != prompt_token_ids) {
        std::cerr << "prompt token ids were not preserved\n";
        return 1;
    }
    if (stepped_generation_context.prompt_token_ids() != prompt_token_ids) {
        std::cerr << "stepped prompt token ids were not preserved\n";
        return 1;
    }
    if (generation_context.running_token_ids() != std::vector<int>({1, 2, 2, 2, 2})) {
        std::cerr << "running token ids do not reflect decode progression\n";
        return 1;
    }
    if (stepped_generation_context.running_token_ids() != generation_context.running_token_ids()) {
        std::cerr << "stepped running token ids do not match baseline\n";
        return 1;
    }

    soc::gpu::QwenCausalLM cached_model = BuildIdentityModel(*context, 1, &error_message);
    soc::gpu::GenerationContext cached_generation_context(std::move(cached_model),
                                                          soc::gpu::Sampler({1.0f, 2}),
                                                          soc::gpu::CommandScheduler(),
                                                          8);
    if (!cached_generation_context.Prefill(*context,
                                           &pipeline_cache,
                                           prompt_token_ids,
                                           temporary_arena.get(),
                                           &error_message)) {
        std::cerr << "prefill for prompt cache artifact failed: " << error_message << '\n';
        return 1;
    }
    const std::filesystem::path artifact_path = std::filesystem::temp_directory_path() / "soc_gpu_prompt_cache_test.bin";
    std::filesystem::remove(artifact_path);
    if (!cached_generation_context.SavePromptCacheArtifact(*context, artifact_path.string(), &error_message)) {
        std::cerr << "save prompt cache artifact failed: " << error_message << '\n';
        return 1;
    }
    soc::gpu::QwenCausalLM loaded_model = BuildIdentityModel(*context, 1, &error_message);
    soc::gpu::GenerationContext loaded_generation_context(std::move(loaded_model),
                                                          soc::gpu::Sampler({1.0f, 2}),
                                                          soc::gpu::CommandScheduler(),
                                                          8);
    if (!loaded_generation_context.LoadPromptCacheArtifact(*context, artifact_path.string(), &error_message)) {
        std::cerr << "load prompt cache artifact failed: " << error_message << '\n';
        return 1;
    }
    std::vector<int> cached_generated_token_ids;
    if (!loaded_generation_context.GenerateFromLoadedPromptCache(*context,
                                                                 &pipeline_cache,
                                                                 3,
                                                                 -1,
                                                                 temporary_arena.get(),
                                                                 &cached_generated_token_ids,
                                                                 &error_message)) {
        std::cerr << "generation from prompt cache artifact failed: " << error_message << '\n';
        return 1;
    }
    if (cached_generated_token_ids != generated_token_ids) {
        std::cerr << "prompt cache artifact generation changed generated token sequence\n";
        return 1;
    }
    if (loaded_generation_context.prompt_token_ids() != prompt_token_ids) {
        std::cerr << "loaded prompt cache artifact prompt ids do not match\n";
        return 1;
    }
    if (loaded_generation_context.running_token_ids() != generation_context.running_token_ids()) {
        std::cerr << "loaded prompt cache artifact running ids do not match\n";
        return 1;
    }
    std::filesystem::remove(artifact_path);

    soc::gpu::QwenCausalLM plan_model = BuildIdentityModel(*context, 2, &error_message);
    soc::gpu::CommandScheduler plan_scheduler;
    auto plan_kv_cache = soc::gpu::KVCache::CreateShared(*context,
                                                         plan_model.num_layers(),
                                                         plan_model.num_key_value_heads(),
                                                         plan_model.head_dim(),
                                                         8,
                                                         "plan_kv",
                                                         &error_message);
    if (plan_kv_cache == nullptr) {
        std::cerr << "failed to create plan kv cache: " << error_message << '\n';
        return 1;
    }
    auto plan_token_buffer = soc::gpu::MetalBuffer::CreateShared(*context, sizeof(std::int32_t), "plan_token", &error_message);
    auto plan_logits_buffer = soc::gpu::MetalBuffer::CreateShared(*context, 4 * sizeof(float), "plan_logits", &error_message);
    if (plan_token_buffer == nullptr || plan_logits_buffer == nullptr) {
        std::cerr << "failed to create plan buffers: " << error_message << '\n';
        return 1;
    }
    const std::int32_t plan_token_value = 1;
    if (!plan_token_buffer->Write(&plan_token_value, sizeof(plan_token_value), 0, &error_message)) {
        std::cerr << "failed to write plan token buffer: " << error_message << '\n';
        return 1;
    }
    const soc::gpu::DeviceTensor plan_token_tensor(plan_token_buffer,
                                                   0,
                                                   soc::gpu::TensorDesc::CreateContiguous(soc::gpu::DataType::kInt32, {1}));
    const soc::gpu::DeviceTensor plan_logits_tensor(plan_logits_buffer,
                                                    0,
                                                    soc::gpu::TensorDesc::CreateContiguous(soc::gpu::DataType::kFloat32, {1, 4}));
    if (!plan_scheduler.RunPrefill(*context,
                                   &pipeline_cache,
                                   plan_model,
                                   plan_token_tensor,
                                   plan_kv_cache.get(),
                                   plan_logits_tensor,
                                   temporary_arena.get(),
                                   0,
                                   &error_message)) {
        std::cerr << "plan prefill failed: " << error_message << '\n';
        return 1;
    }
    setenv("SOC_GPU_ENABLE_EXPERIMENTAL_PREBUILT_DECODE_PLAN", "1", 1);
    if (!plan_scheduler.RunDecode(*context,
                                  &pipeline_cache,
                                  plan_model,
                                  plan_token_tensor,
                                  plan_kv_cache.get(),
                                  plan_logits_tensor,
                                  temporary_arena.get(),
                                  0,
                                  &error_message)) {
        unsetenv("SOC_GPU_ENABLE_EXPERIMENTAL_PREBUILT_DECODE_PLAN");
        std::cerr << "plan decode failed: " << error_message << '\n';
        return 1;
    }
    unsetenv("SOC_GPU_ENABLE_EXPERIMENTAL_PREBUILT_DECODE_PLAN");
    const soc::gpu::DecodeExecutionPlan* decode_plan = plan_scheduler.decode_plan();
    if (decode_plan == nullptr || decode_plan->stages.size() != 3) {
        std::cerr << "unexpected decode plan stage count\n";
        return 1;
    }
    const std::size_t layer_span_bytes = plan_kv_cache->GetLayerSpanBytes();
    const std::size_t token_row_bytes = plan_model.num_key_value_heads() * plan_model.head_dim() * sizeof(float);
    if (decode_plan->stages[0].accesses.size() != 4 || decode_plan->stages[1].accesses.size() != 4) {
        std::cerr << "decode plan access metadata missing\n";
        return 1;
    }
    if (decode_plan->stages[0].accesses[2].buffer_kind != soc::gpu::DecodePlanBufferKind::kKvKeys ||
        decode_plan->stages[0].accesses[2].byte_offset != token_row_bytes ||
        decode_plan->stages[1].accesses[2].byte_offset != layer_span_bytes + token_row_bytes) {
        std::cerr << "decode plan KV key access offsets are incorrect\n";
        return 1;
    }
    if (decode_plan->stages[0].accesses[3].buffer_kind != soc::gpu::DecodePlanBufferKind::kKvValues ||
        decode_plan->stages[0].accesses[3].byte_offset != token_row_bytes ||
        decode_plan->stages[1].accesses[3].byte_offset != layer_span_bytes + token_row_bytes) {
        std::cerr << "decode plan KV value access offsets are incorrect\n";
        return 1;
    }
    if (decode_plan->stages[1].batch_id <= decode_plan->stages[0].batch_id ||
        decode_plan->stages[2].batch_id <= decode_plan->stages[1].batch_id) {
        std::cerr << "decode plan batch ids did not respect hazards\n";
        return 1;
    }

    generation_context.Reset();
    if (!generation_context.prompt_token_ids().empty() || !generation_context.running_token_ids().empty()) {
        std::cerr << "reset did not clear generation state\n";
        return 1;
    }
    stepped_generation_context.Reset();
    if (!stepped_generation_context.prompt_token_ids().empty() || !stepped_generation_context.running_token_ids().empty()) {
        std::cerr << "stepped reset did not clear generation state\n";
        return 1;
    }

    std::cout << "test_generation_context passed\n";
    return 0;
}