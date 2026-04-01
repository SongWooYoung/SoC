#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "runtime/command_scheduler.h"
#include "runtime/model_runner.h"
#include "runtime/runtime_policy.h"
#include "runtime/sampler.h"

namespace soc::gpu {

class BufferArena;
class MetalContext;

struct GenerationStepResult {
    int token_id = -1;
    std::vector<float> top_logits;
    std::vector<int> top_token_ids;
};

class GenerationContext {
public:
    GenerationContext(std::shared_ptr<ModelRunner> model,
                      Sampler sampler,
                      CommandScheduler scheduler,
                      std::size_t max_sequence_length,
                      std::size_t prefill_step_size = 0);
    ~GenerationContext();

    bool Prefill(const MetalContext& context,
                 PipelineCache* pipeline_cache,
                 const std::vector<int>& token_ids,
                 BufferArena* temporary_arena,
                 std::string* error_message);

    bool DecodeNextToken(const MetalContext& context,
                         PipelineCache* pipeline_cache,
                         int last_token_id,
                         BufferArena* temporary_arena,
                         GenerationStepResult* result,
                         std::string* error_message);

    bool Generate(const MetalContext& context,
                  PipelineCache* pipeline_cache,
                  const std::vector<int>& prompt_token_ids,
                  std::size_t max_new_tokens,
                  int eos_token_id,
                  BufferArena* temporary_arena,
                  std::vector<int>* generated_token_ids,
                  std::string* error_message);

    bool GenerateFromLoadedPromptCache(const MetalContext& context,
                                       PipelineCache* pipeline_cache,
                                       std::size_t max_new_tokens,
                                       int eos_token_id,
                                       BufferArena* temporary_arena,
                                       std::vector<int>* generated_token_ids,
                                       std::string* error_message);

    bool SavePromptCacheArtifact(const MetalContext& context,
                                 const std::string& artifact_path,
                                 std::string* error_message) const;
    bool LoadPromptCacheArtifact(const MetalContext& context,
                                 const std::string& artifact_path,
                                 std::string* error_message);

    void Reset();
    const std::vector<int>& prompt_token_ids() const;
    const std::vector<int>& running_token_ids() const;
    const ModelRunner& model() const;
    const CommandScheduler& scheduler() const;
    const RuntimePolicy& runtime_policy() const;

private:
    void EnsureRuntimePolicyResolved(const MetalContext& context);
    bool EnsureLogitsBuffer(const MetalContext& context,
                            std::size_t row_count,
                            std::string* error_message);
    bool EnsureTokenBuffer(const MetalContext& context,
                           std::size_t token_count,
                           std::string* error_message);
    bool UploadTokenIds(const MetalContext& context,
                        const std::vector<int>& token_ids,
                        DeviceTensor* token_tensor,
                        std::string* error_message);

    std::shared_ptr<ModelRunner> model_;
    Sampler sampler_;
    CommandScheduler scheduler_;
    std::size_t max_sequence_length_ = 0;
    std::size_t prefill_step_size_ = 0;
    RuntimePolicy runtime_policy_;
    std::unique_ptr<KVCache> kv_cache_;
    std::vector<int> prompt_token_ids_;
    std::vector<int> running_token_ids_;
    std::shared_ptr<class MetalBuffer> logits_buffer_;
    std::shared_ptr<class MetalBuffer> token_buffer_;
};

}  // namespace soc::gpu
