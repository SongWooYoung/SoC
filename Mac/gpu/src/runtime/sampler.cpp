#include "runtime/sampler.h"

#include <algorithm>
#include <cstdint>

#include "buffer/buffer_arena.h"
#include "buffer/metal_buffer.h"
#include "metal/metal_context.h"
#include "op/sampler_topk_op.h"

namespace soc::gpu {

namespace {

bool SampleFromLogitsCpuFallback(const DeviceTensor& logits,
                                 std::size_t row_index,
                                 std::size_t top_k,
                                 int* token_id,
                                 std::vector<float>* top_logits,
                                 std::vector<int>* top_token_ids,
                                 std::string* error_message) {
    const std::size_t vocab_size = logits.GetDesc().GetShape()[1];
    std::vector<float> row(vocab_size, 0.0f);
    if (!logits.GetBuffer()->Read(row.data(), sizeof(float) * vocab_size, logits.GetByteOffset() + row_index * sizeof(float) * vocab_size, error_message)) {
        return false;
    }

    std::vector<std::size_t> candidates(vocab_size);
    for (std::size_t index = 0; index < vocab_size; ++index) {
        candidates[index] = index;
    }
    std::partial_sort(candidates.begin(),
                      candidates.begin() + static_cast<std::ptrdiff_t>(top_k),
                      candidates.end(),
                      [&row](std::size_t lhs, std::size_t rhs) {
                          return row[lhs] > row[rhs];
                      });

    *token_id = static_cast<int>(candidates[0]);
    if (top_logits != nullptr) {
        top_logits->clear();
        for (std::size_t rank = 0; rank < top_k; ++rank) {
            top_logits->push_back(row[candidates[rank]]);
        }
    }
    if (top_token_ids != nullptr) {
        top_token_ids->clear();
        for (std::size_t rank = 0; rank < top_k; ++rank) {
            top_token_ids->push_back(static_cast<int>(candidates[rank]));
        }
    }
    return true;
}

}  // namespace

Sampler::Sampler(SamplerConfig config) : config_(config) {}

const SamplerConfig& Sampler::config() const { return config_; }

bool Sampler::SampleFromLogits(const MetalContext& context,
                               PipelineCache* pipeline_cache,
                               const DeviceTensor& logits,
                               std::size_t row_index,
                               int* token_id,
                               std::vector<float>* top_logits,
                               std::vector<int>* top_token_ids,
                               BufferArena* temporary_arena,
                               std::string* error_message) const {
    if (token_id == nullptr) {
        if (error_message != nullptr) {
            *error_message = "Sampler requires a non-null token_id output";
        }
        return false;
    }
    if (!logits.IsValid() || logits.GetDesc().GetDataType() != DataType::kFloat32 || logits.GetDesc().Rank() != 2) {
        if (error_message != nullptr) {
            *error_message = "Sampler expects a valid rank-2 float32 logits tensor";
        }
        return false;
    }
    if (row_index >= logits.GetDesc().GetShape()[0]) {
        if (error_message != nullptr) {
            *error_message = "Sampler row_index is out of range";
        }
        return false;
    }
    const std::size_t vocab_size = logits.GetDesc().GetShape()[1];
    const std::size_t top_k = std::max<std::size_t>(1, std::min<std::size_t>(config_.top_k, vocab_size));

    if (pipeline_cache == nullptr || top_k > SamplerTopKParams::kMaxTopK) {
        return SampleFromLogitsCpuFallback(logits,
                                           row_index,
                                           top_k,
                                           token_id,
                                           top_logits,
                                           top_token_ids,
                                           error_message);
    }

    auto top_values_buffer = MetalBuffer::CreateShared(context, sizeof(float) * top_k, "sampler_top_values", error_message);
    auto top_indices_buffer = MetalBuffer::CreateShared(context, sizeof(std::int32_t) * top_k, "sampler_top_indices", error_message);
    if (top_values_buffer == nullptr || top_indices_buffer == nullptr) {
        return false;
    }

    const DeviceTensor top_values_tensor(top_values_buffer,
                                         0,
                                         TensorDesc::CreateContiguous(DataType::kFloat32, {1, top_k}));
    const DeviceTensor top_indices_tensor(top_indices_buffer,
                                          0,
                                          TensorDesc::CreateContiguous(DataType::kInt32, {1, top_k}));
    SamplerTopKParams params;
    params.row_count = 1;
    params.row_size = static_cast<std::uint32_t>(vocab_size);
    params.top_k = static_cast<std::uint32_t>(top_k);
    const DeviceTensor row_logits(logits.GetBuffer(),
                                  logits.GetByteOffset() + row_index * sizeof(float) * vocab_size,
                                  TensorDesc::CreateContiguous(DataType::kFloat32, {1, vocab_size}));
    if (!SamplerTopKOp::Run(context,
                            pipeline_cache,
                            row_logits,
                            top_values_tensor,
                            top_indices_tensor,
                            params,
                            temporary_arena,
                            error_message)) {
        return false;
    }

    std::vector<float> reduced_logits(top_k, 0.0f);
    std::vector<std::int32_t> reduced_indices(top_k, -1);
    if (!top_values_buffer->Read(reduced_logits.data(), sizeof(float) * top_k, 0, error_message) ||
        !top_indices_buffer->Read(reduced_indices.data(), sizeof(std::int32_t) * top_k, 0, error_message)) {
        return false;
    }

    *token_id = reduced_indices.empty() ? -1 : reduced_indices[0];
    if (top_logits != nullptr) {
        top_logits->assign(reduced_logits.begin(), reduced_logits.end());
    }
    if (top_token_ids != nullptr) {
        top_token_ids->clear();
        top_token_ids->reserve(reduced_indices.size());
        for (std::int32_t index : reduced_indices) {
            top_token_ids->push_back(index);
        }
    }
    return true;
}

}  // namespace soc::gpu