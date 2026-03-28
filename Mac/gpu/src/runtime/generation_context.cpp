#include "runtime/generation_context.h"

#include "buffer/buffer_arena.h"
#include "buffer/metal_buffer.h"

namespace soc::gpu {

GenerationContext::GenerationContext(QwenCausalLM model,
                                     Sampler sampler,
                                     CommandScheduler scheduler,
                                     std::size_t max_sequence_length)
    : model_(std::move(model)),
      sampler_(std::move(sampler)),
      scheduler_(std::move(scheduler)),
      max_sequence_length_(max_sequence_length) {}

bool GenerationContext::Prefill(const MetalContext& context,
                                PipelineCache* pipeline_cache,
                                const std::vector<int>& token_ids,
                                BufferArena* temporary_arena,
                                std::string* error_message) {
    if (token_ids.empty()) {
        if (error_message != nullptr) {
            *error_message = "GenerationContext prefill requires at least one token";
        }
        return false;
    }
    Reset();
    prompt_token_ids_ = token_ids;
    running_token_ids_ = token_ids;
    kv_cache_ = KVCache::CreateShared(context,
                                      model_.num_layers(),
                                      model_.num_key_value_heads(),
                                      model_.head_dim(),
                                      max_sequence_length_,
                                      "generation_kv_cache",
                                      error_message);
    if (kv_cache_ == nullptr) {
        return false;
    }
    if (!EnsureLogitsBuffer(context, token_ids.size(), error_message)) {
        return false;
    }

    DeviceTensor token_tensor;
    if (!UploadTokenIds(context, token_ids, &token_tensor, error_message)) {
        return false;
    }
    const DeviceTensor logits_tensor(logits_buffer_, 0, TensorDesc::CreateContiguous(DataType::kFloat32, {token_ids.size(), model_.vocab_size()}));
    return scheduler_.RunPrefill(context,
                                 pipeline_cache,
                                 model_,
                                 token_tensor,
                                 kv_cache_.get(),
                                 logits_tensor,
                                 temporary_arena,
                                 0,
                                 error_message);
}

bool GenerationContext::DecodeNextToken(const MetalContext& context,
                                        PipelineCache* pipeline_cache,
                                        int last_token_id,
                                        BufferArena* temporary_arena,
                                        GenerationStepResult* result,
                                        std::string* error_message) {
    if (result == nullptr) {
        if (error_message != nullptr) {
            *error_message = "GenerationContext decode requires a non-null result";
        }
        return false;
    }
    if (kv_cache_ == nullptr) {
        if (error_message != nullptr) {
            *error_message = "GenerationContext decode requires a prefilled KV cache";
        }
        return false;
    }
    if (running_token_ids_.empty() || running_token_ids_.back() != last_token_id) {
        if (error_message != nullptr) {
            *error_message = "GenerationContext decode requires last_token_id to match the current sequence tail";
        }
        return false;
    }
    if (running_token_ids_.size() >= max_sequence_length_) {
        if (error_message != nullptr) {
            *error_message = "GenerationContext sequence length would exceed max_sequence_length";
        }
        return false;
    }
    if (!EnsureLogitsBuffer(context, 1, error_message)) {
        return false;
    }

    DeviceTensor token_tensor;
    if (!UploadTokenIds(context, std::vector<int>{last_token_id}, &token_tensor, error_message)) {
        return false;
    }
    const DeviceTensor logits_tensor(logits_buffer_, 0, TensorDesc::CreateContiguous(DataType::kFloat32, {1, model_.vocab_size()}));
    if (!scheduler_.RunDecode(context,
                              pipeline_cache,
                              model_,
                              token_tensor,
                              kv_cache_.get(),
                              logits_tensor,
                              temporary_arena,
                              running_token_ids_.size() - 1,
                              error_message)) {
        return false;
    }
    if (!sampler_.SampleFromLogits(logits_tensor, 0, &result->token_id, &result->top_logits, &result->top_token_ids, error_message)) {
        return false;
    }
    running_token_ids_.push_back(result->token_id);
    return true;
}

bool GenerationContext::Generate(const MetalContext& context,
                                 PipelineCache* pipeline_cache,
                                 const std::vector<int>& prompt_token_ids,
                                 std::size_t max_new_tokens,
                                 int eos_token_id,
                                 BufferArena* temporary_arena,
                                 std::vector<int>* generated_token_ids,
                                 std::string* error_message) {
    if (generated_token_ids == nullptr) {
        if (error_message != nullptr) {
            *error_message = "GenerationContext generate requires a non-null generated_token_ids output";
        }
        return false;
    }
    generated_token_ids->clear();
    if (!Prefill(context, pipeline_cache, prompt_token_ids, temporary_arena, error_message)) {
        return false;
    }
    if (max_new_tokens == 0) {
        return true;
    }

    GenerationStepResult first_step_result;
    const DeviceTensor prefill_logits(logits_buffer_,
                                      0,
                                      TensorDesc::CreateContiguous(DataType::kFloat32,
                                                                   {prompt_token_ids.size(), model_.vocab_size()}));
    if (!sampler_.SampleFromLogits(prefill_logits,
                                   prompt_token_ids.size() - 1,
                                   &first_step_result.token_id,
                                   &first_step_result.top_logits,
                                   &first_step_result.top_token_ids,
                                   error_message)) {
        return false;
    }
    generated_token_ids->push_back(first_step_result.token_id);
    running_token_ids_.push_back(first_step_result.token_id);
    if (eos_token_id >= 0 && first_step_result.token_id == eos_token_id) {
        return true;
    }

    int last_token_id = first_step_result.token_id;
    for (std::size_t step = 1; step < max_new_tokens; ++step) {
        GenerationStepResult step_result;
        if (!DecodeNextToken(context, pipeline_cache, last_token_id, temporary_arena, &step_result, error_message)) {
            return false;
        }
        generated_token_ids->push_back(step_result.token_id);
        last_token_id = step_result.token_id;
        if (eos_token_id >= 0 && step_result.token_id == eos_token_id) {
            break;
        }
    }
    return true;
}

void GenerationContext::Reset() {
    kv_cache_.reset();
    prompt_token_ids_.clear();
    running_token_ids_.clear();
    logits_buffer_.reset();
}

const std::vector<int>& GenerationContext::prompt_token_ids() const { return prompt_token_ids_; }
const std::vector<int>& GenerationContext::running_token_ids() const { return running_token_ids_; }
const QwenCausalLM& GenerationContext::model() const { return model_; }

bool GenerationContext::EnsureLogitsBuffer(const MetalContext& context,
                                           std::size_t row_count,
                                           std::string* error_message) {
    const std::size_t required_size = row_count * model_.vocab_size() * sizeof(float);
    if (logits_buffer_ != nullptr && logits_buffer_->GetSizeBytes() >= required_size) {
        return true;
    }
    logits_buffer_ = MetalBuffer::CreateShared(context, required_size, "generation_logits", error_message);
    return logits_buffer_ != nullptr;
}

bool GenerationContext::UploadTokenIds(const MetalContext& context,
                                       const std::vector<int>& token_ids,
                                       DeviceTensor* token_tensor,
                                       std::string* error_message) {
    auto token_buffer = MetalBuffer::CreateShared(context, token_ids.size() * sizeof(std::int32_t), "generation_tokens", error_message);
    if (token_buffer == nullptr) {
        return false;
    }
    if (!token_buffer->Write(token_ids.data(), token_ids.size() * sizeof(std::int32_t), 0, error_message)) {
        return false;
    }
    *token_tensor = DeviceTensor(token_buffer, 0, TensorDesc::CreateContiguous(DataType::kInt32, {token_ids.size()}));
    return true;
}

}  // namespace soc::gpu