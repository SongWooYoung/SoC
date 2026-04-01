#include "runtime/generation_context.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>

#include "buffer/buffer_arena.h"
#include "buffer/metal_buffer.h"
#include "runtime/kv_cache.h"
#include "runtime/runtime_policy.h"

namespace soc::gpu {
namespace {

struct PromptCacheArtifactHeader {
    char magic[8];
    std::uint32_t version = 1;
    std::uint32_t reserved = 0;
    std::uint64_t vocab_size = 0;
    std::uint64_t prompt_token_count = 0;
    std::uint64_t logits_bytes = 0;
    std::uint64_t layer_count = 0;
    std::uint64_t key_value_head_count = 0;
    std::uint64_t head_dim = 0;
    std::uint64_t max_sequence_length = 0;
    std::uint64_t key_bytes = 0;
    std::uint64_t value_bytes = 0;
};

constexpr char kPromptCacheArtifactMagic[8] = {'S', 'O', 'C', 'P', 'C', 'A', 'C', 'H'};

std::size_t ResolvePrefillStepSize(std::size_t configured_step_size) {
    if (configured_step_size > 0) {
        return configured_step_size;
    }

    const char* value = std::getenv("SOC_GPU_PREFILL_STEP_SIZE");
    if (value == nullptr || value[0] == '\0') {
        return 0;
    }

    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || end == value || (end != nullptr && *end != '\0') || parsed == 0) {
        return 0;
    }
    return static_cast<std::size_t>(parsed);
}

template <typename T>
bool WriteBinary(std::ofstream* stream, const T& value, std::string* error_message) {
    stream->write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!stream->good()) {
        if (error_message != nullptr) {
            *error_message = "Failed to write prompt cache artifact";
        }
        return false;
    }
    return true;
}

template <typename T>
bool ReadBinary(std::ifstream* stream, T* value, std::string* error_message) {
    stream->read(reinterpret_cast<char*>(value), sizeof(T));
    if (!stream->good()) {
        if (error_message != nullptr) {
            *error_message = "Failed to read prompt cache artifact";
        }
        return false;
    }
    return true;
}

bool WriteBytes(std::ofstream* stream,
                const void* data,
                std::size_t size_bytes,
                std::string* error_message) {
    if (size_bytes == 0) {
        return true;
    }
    stream->write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size_bytes));
    if (!stream->good()) {
        if (error_message != nullptr) {
            *error_message = "Failed to write prompt cache artifact payload";
        }
        return false;
    }
    return true;
}

bool ReadBytes(std::ifstream* stream,
               void* data,
               std::size_t size_bytes,
               std::string* error_message) {
    if (size_bytes == 0) {
        return true;
    }
    stream->read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(size_bytes));
    if (!stream->good()) {
        if (error_message != nullptr) {
            *error_message = "Failed to read prompt cache artifact payload";
        }
        return false;
    }
    return true;
}

}  // namespace

GenerationContext::GenerationContext(std::shared_ptr<ModelRunner> model,
                                     Sampler sampler,
                                     CommandScheduler scheduler,
                                     std::size_t max_sequence_length,
                                     std::size_t prefill_step_size)
    : model_(std::move(model)),
      sampler_(std::move(sampler)),
      scheduler_(std::move(scheduler)),
      max_sequence_length_(max_sequence_length),
      prefill_step_size_(ResolvePrefillStepSize(prefill_step_size)) {}

GenerationContext::~GenerationContext() = default;

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
                                      model_->num_layers(),
                                      model_->num_key_value_heads(),
                                      model_->head_dim(),
                                      max_sequence_length_,
                                      "generation_kv_cache",
                                      error_message);
    if (kv_cache_ == nullptr) {
        return false;
    }
    if (!EnsureLogitsBuffer(context, token_ids.size(), error_message)) {
        return false;
    }

    EnsureRuntimePolicyResolved(context);

    const std::size_t configured_step_size = runtime_policy_.prefill_step_size;
    const std::size_t step_size = configured_step_size == 0 ? token_ids.size() : std::min(configured_step_size, token_ids.size());
    const BufferArenaMark prefill_mark = temporary_arena == nullptr ? 0 : temporary_arena->GetMark();

    for (std::size_t chunk_start = 0; chunk_start < token_ids.size(); chunk_start += step_size) {
        const std::size_t chunk_size = std::min(step_size, token_ids.size() - chunk_start);
        const std::vector<int> chunk_token_ids(token_ids.begin() + static_cast<std::ptrdiff_t>(chunk_start),
                                               token_ids.begin() + static_cast<std::ptrdiff_t>(chunk_start + chunk_size));
        DeviceTensor token_tensor;
        if (!UploadTokenIds(context, chunk_token_ids, &token_tensor, error_message)) {
            return false;
        }

        const DeviceTensor logits_tensor(logits_buffer_,
                                         chunk_start * model_->vocab_size() * sizeof(float),
                                         TensorDesc::CreateContiguous(DataType::kFloat32,
                                                                      {chunk_size, model_->vocab_size()}));
        if (!scheduler_.RunPrefill(context,
                                   pipeline_cache,
                                   *model_,
                                   token_tensor,
                                   kv_cache_.get(),
                                   logits_tensor,
                                   temporary_arena,
                                   chunk_start,
                                   error_message)) {
            return false;
        }
        if (temporary_arena != nullptr && !temporary_arena->ResetToMark(prefill_mark, error_message)) {
            return false;
        }
    }

    return true;
}

bool GenerationContext::DecodeNextToken(const MetalContext& context,
                                        PipelineCache* pipeline_cache,
                                        int last_token_id,
                                        BufferArena* temporary_arena,
                                        GenerationStepResult* result,
                                        std::string* error_message) {
    EnsureRuntimePolicyResolved(context);
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
    const DeviceTensor logits_tensor(logits_buffer_, 0, TensorDesc::CreateContiguous(DataType::kFloat32, {1, model_->vocab_size()}));
    if (!scheduler_.RunDecode(context,
                              pipeline_cache,
                              *model_,
                              token_tensor,
                              kv_cache_.get(),
                              logits_tensor,
                              temporary_arena,
                              running_token_ids_.size() - 1,
                              error_message)) {
        return false;
    }
    if (!sampler_.SampleFromLogits(context,
                                   pipeline_cache,
                                   logits_tensor,
                                   0,
                                   &result->token_id,
                                   &result->top_logits,
                                   &result->top_token_ids,
                                   temporary_arena,
                                   error_message)) {
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

    return GenerateFromLoadedPromptCache(context,
                                         pipeline_cache,
                                         max_new_tokens,
                                         eos_token_id,
                                         temporary_arena,
                                         generated_token_ids,
                                         error_message);
}

bool GenerationContext::GenerateFromLoadedPromptCache(const MetalContext& context,
                                                      PipelineCache* pipeline_cache,
                                                      std::size_t max_new_tokens,
                                                      int eos_token_id,
                                                      BufferArena* temporary_arena,
                                                      std::vector<int>* generated_token_ids,
                                                      std::string* error_message) {
    EnsureRuntimePolicyResolved(context);
    if (generated_token_ids == nullptr) {
        if (error_message != nullptr) {
            *error_message = "GenerationContext generate-from-cache requires a non-null generated_token_ids output";
        }
        return false;
    }
    generated_token_ids->clear();
    if (kv_cache_ == nullptr || prompt_token_ids_.empty() || logits_buffer_ == nullptr) {
        if (error_message != nullptr) {
            *error_message = "GenerationContext generate-from-cache requires a loaded prompt cache state";
        }
        return false;
    }
    if (max_new_tokens == 0) {
        return true;
    }

    GenerationStepResult first_step_result;
    const DeviceTensor prefill_logits(logits_buffer_,
                                      0,
                                      TensorDesc::CreateContiguous(DataType::kFloat32,
                                                                   {prompt_token_ids_.size(), model_->vocab_size()}));
    if (!sampler_.SampleFromLogits(context,
                                   pipeline_cache,
                                   prefill_logits,
                                   prompt_token_ids_.size() - 1,
                                   &first_step_result.token_id,
                                   &first_step_result.top_logits,
                                   &first_step_result.top_token_ids,
                                   temporary_arena,
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

bool GenerationContext::SavePromptCacheArtifact(const MetalContext& context,
                                                const std::string& artifact_path,
                                                std::string* error_message) const {
    if (kv_cache_ == nullptr || prompt_token_ids_.empty() || logits_buffer_ == nullptr) {
        if (error_message != nullptr) {
            *error_message = "GenerationContext save prompt cache requires a prefetched state";
        }
        return false;
    }
    if (running_token_ids_ != prompt_token_ids_) {
        if (error_message != nullptr) {
            *error_message = "GenerationContext save prompt cache only supports pre-decode prompt state";
        }
        return false;
    }

    KVCacheSerializedState kv_state;
    if (!kv_cache_->Serialize(context, &kv_state, error_message)) {
        return false;
    }

    const std::size_t logits_bytes = prompt_token_ids_.size() * model_->vocab_size() * sizeof(float);
    std::vector<std::uint8_t> logits_data(logits_bytes, 0);
    if (!logits_buffer_->Read(logits_data.data(), logits_data.size(), 0, error_message)) {
        return false;
    }

    const std::filesystem::path artifact_fs_path(artifact_path);
    if (artifact_fs_path.has_parent_path()) {
        std::error_code create_error;
        std::filesystem::create_directories(artifact_fs_path.parent_path(), create_error);
        if (create_error) {
            if (error_message != nullptr) {
                *error_message = "Failed to create prompt cache artifact directory";
            }
            return false;
        }
    }

    std::ofstream stream(artifact_path, std::ios::binary | std::ios::trunc);
    if (!stream.is_open()) {
        if (error_message != nullptr) {
            *error_message = "Failed to open prompt cache artifact for writing";
        }
        return false;
    }

    PromptCacheArtifactHeader header{};
    std::memcpy(header.magic, kPromptCacheArtifactMagic, sizeof(header.magic));
    header.vocab_size = model_->vocab_size();
    header.prompt_token_count = prompt_token_ids_.size();
    header.logits_bytes = logits_data.size();
    header.layer_count = kv_state.layer_count;
    header.key_value_head_count = kv_state.key_value_head_count;
    header.head_dim = kv_state.head_dim;
    header.max_sequence_length = kv_state.max_sequence_length;
    header.key_bytes = kv_state.key_bytes.size();
    header.value_bytes = kv_state.value_bytes.size();

    if (!WriteBinary(&stream, header, error_message) ||
        !WriteBytes(&stream, prompt_token_ids_.data(), prompt_token_ids_.size() * sizeof(int), error_message) ||
        !WriteBytes(&stream, kv_state.sequence_lengths.data(), kv_state.sequence_lengths.size() * sizeof(std::size_t), error_message) ||
        !WriteBytes(&stream, kv_state.key_bytes.data(), kv_state.key_bytes.size(), error_message) ||
        !WriteBytes(&stream, kv_state.value_bytes.data(), kv_state.value_bytes.size(), error_message) ||
        !WriteBytes(&stream, logits_data.data(), logits_data.size(), error_message)) {
        return false;
    }
    return true;
}

bool GenerationContext::LoadPromptCacheArtifact(const MetalContext& context,
                                                const std::string& artifact_path,
                                                std::string* error_message) {
    EnsureRuntimePolicyResolved(context);
    std::ifstream stream(artifact_path, std::ios::binary);
    if (!stream.is_open()) {
        if (error_message != nullptr) {
            *error_message = "Failed to open prompt cache artifact for reading";
        }
        return false;
    }

    PromptCacheArtifactHeader header{};
    if (!ReadBinary(&stream, &header, error_message)) {
        return false;
    }
    if (std::memcmp(header.magic, kPromptCacheArtifactMagic, sizeof(header.magic)) != 0 || header.version != 1) {
        if (error_message != nullptr) {
            *error_message = "Prompt cache artifact header is invalid";
        }
        return false;
    }
    if (header.vocab_size != model_->vocab_size() || header.layer_count != model_->num_layers() ||
        header.key_value_head_count != model_->num_key_value_heads() || header.head_dim != model_->head_dim()) {
        if (error_message != nullptr) {
            *error_message = "Prompt cache artifact does not match model configuration";
        }
        return false;
    }
    if (header.prompt_token_count == 0 || header.prompt_token_count > max_sequence_length_ ||
        header.max_sequence_length > max_sequence_length_) {
        if (error_message != nullptr) {
            *error_message = "Prompt cache artifact exceeds GenerationContext sequence capacity";
        }
        return false;
    }

    std::vector<int> prompt_token_ids(static_cast<std::size_t>(header.prompt_token_count), 0);
    std::vector<std::size_t> sequence_lengths(static_cast<std::size_t>(header.layer_count), 0);
    std::vector<std::uint8_t> key_bytes(static_cast<std::size_t>(header.key_bytes), 0);
    std::vector<std::uint8_t> value_bytes(static_cast<std::size_t>(header.value_bytes), 0);
    std::vector<std::uint8_t> logits_data(static_cast<std::size_t>(header.logits_bytes), 0);
    if (!ReadBytes(&stream, prompt_token_ids.data(), prompt_token_ids.size() * sizeof(int), error_message) ||
        !ReadBytes(&stream, sequence_lengths.data(), sequence_lengths.size() * sizeof(std::size_t), error_message) ||
        !ReadBytes(&stream, key_bytes.data(), key_bytes.size(), error_message) ||
        !ReadBytes(&stream, value_bytes.data(), value_bytes.size(), error_message) ||
        !ReadBytes(&stream, logits_data.data(), logits_data.size(), error_message)) {
        return false;
    }

    KVCacheSerializedState kv_state;
    kv_state.layer_count = static_cast<std::size_t>(header.layer_count);
    kv_state.key_value_head_count = static_cast<std::size_t>(header.key_value_head_count);
    kv_state.head_dim = static_cast<std::size_t>(header.head_dim);
    kv_state.max_sequence_length = static_cast<std::size_t>(header.max_sequence_length);
    kv_state.sequence_lengths = std::move(sequence_lengths);
    kv_state.key_bytes = std::move(key_bytes);
    kv_state.value_bytes = std::move(value_bytes);

    auto cache = KVCache::Deserialize(context, kv_state, "generation_kv_cache_artifact", error_message);
    if (cache == nullptr) {
        return false;
    }

    Reset();
    kv_cache_ = std::move(cache);
    prompt_token_ids_ = std::move(prompt_token_ids);
    running_token_ids_ = prompt_token_ids_;
    if (!EnsureLogitsBuffer(context, prompt_token_ids_.size(), error_message)) {
        return false;
    }
    return logits_buffer_->Write(logits_data.data(), logits_data.size(), 0, error_message);
}

void GenerationContext::Reset() {
    kv_cache_.reset();
    prompt_token_ids_.clear();
    running_token_ids_.clear();
    logits_buffer_.reset();
    token_buffer_.reset();
}

const std::vector<int>& GenerationContext::prompt_token_ids() const { return prompt_token_ids_; }
const std::vector<int>& GenerationContext::running_token_ids() const { return running_token_ids_; }
const ModelRunner& GenerationContext::model() const { return *model_; }
const CommandScheduler& GenerationContext::scheduler() const { return scheduler_; }
const RuntimePolicy& GenerationContext::runtime_policy() const { return runtime_policy_; }

void GenerationContext::EnsureRuntimePolicyResolved(const MetalContext& context) {
    runtime_policy_ = ResolveRuntimePolicy(context, prefill_step_size_);
}

bool GenerationContext::EnsureLogitsBuffer(const MetalContext& context,
                                           std::size_t row_count,
                                           std::string* error_message) {
    const std::size_t required_size = row_count * model_->vocab_size() * sizeof(float);
    if (logits_buffer_ != nullptr && logits_buffer_->GetSizeBytes() >= required_size) {
        return true;
    }
    logits_buffer_ = MetalBuffer::CreateForTensorClass(context,
                                                       required_size,
                                                       "generation_logits",
                                                       TensorClass::kTokenMetadata,
                                                       error_message);
    return logits_buffer_ != nullptr;
}

bool GenerationContext::EnsureTokenBuffer(const MetalContext& context,
                                          std::size_t token_count,
                                          std::string* error_message) {
    const std::size_t required_size = token_count * sizeof(std::int32_t);
    if (token_buffer_ != nullptr && token_buffer_->GetSizeBytes() >= required_size) {
        return true;
    }
    token_buffer_ = MetalBuffer::CreateForTensorClass(context,
                                                      required_size,
                                                      "generation_tokens",
                                                      TensorClass::kTokenMetadata,
                                                      error_message);
    return token_buffer_ != nullptr;
}

bool GenerationContext::UploadTokenIds(const MetalContext& context,
                                       const std::vector<int>& token_ids,
                                       DeviceTensor* token_tensor,
                                       std::string* error_message) {
    if (!EnsureTokenBuffer(context, token_ids.size(), error_message)) {
        return false;
    }
    if (!token_buffer_->Write(token_ids.data(), token_ids.size() * sizeof(std::int32_t), 0, error_message)) {
        return false;
    }
    *token_tensor = DeviceTensor(token_buffer_, 0, TensorDesc::CreateContiguous(DataType::kInt32, {token_ids.size()}));
    return true;
}

}  // namespace soc::gpu
