#include "runtime/sampler.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <random>

#include "buffer/buffer_arena.h"
#include "buffer/metal_buffer.h"
#include "metal/metal_context.h"
#include "op/sampler_topk_op.h"

namespace soc::gpu {

namespace {

bool UseExperimentalGpuSampler() {
    const char* value = std::getenv("SOC_GPU_ENABLE_EXPERIMENTAL_GPU_SAMPLER");
    return value != nullptr && std::string(value) == "1";
}

bool SampleFromLogitsCpuFallback(const DeviceTensor& logits,
                                 std::size_t row_index,
                                 float temperature,
                                 std::size_t top_k,
                                 std::mt19937_64* rng,
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

    if (top_k == 1) {
        *token_id = static_cast<int>(candidates[0]);
    } else {
        if (rng == nullptr) {
            if (error_message != nullptr) {
                *error_message = "Sampler RNG must not be null";
            }
            return false;
        }
        std::vector<double> scaled_logits(top_k, 0.0);
        double max_score = -std::numeric_limits<double>::infinity();
        for (std::size_t rank = 0; rank < top_k; ++rank) {
            const double score = static_cast<double>(row[candidates[rank]]) / static_cast<double>(temperature);
            scaled_logits[rank] = score;
            if (score > max_score) {
                max_score = score;
            }
        }
        std::vector<double> weights(top_k, 0.0);
        for (std::size_t rank = 0; rank < top_k; ++rank) {
            weights[rank] = std::exp(scaled_logits[rank] - max_score);
        }
        std::discrete_distribution<std::size_t> distribution(weights.begin(), weights.end());
        *token_id = static_cast<int>(candidates[distribution(*rng)]);
    }
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

bool EnsureTopKBuffers(const MetalContext& context,
                       std::size_t top_k,
                       std::shared_ptr<MetalBuffer>* top_values_buffer,
                       std::shared_ptr<MetalBuffer>* top_indices_buffer,
                       std::string* error_message) {
    if (top_values_buffer == nullptr || top_indices_buffer == nullptr) {
        if (error_message != nullptr) {
            *error_message = "Sampler buffer outputs must not be null";
        }
        return false;
    }

    const std::size_t values_bytes = sizeof(float) * top_k;
    if (*top_values_buffer == nullptr || (*top_values_buffer)->GetSizeBytes() < values_bytes) {
        *top_values_buffer = MetalBuffer::CreateForTensorClass(context,
                                                               values_bytes,
                                                               "sampler_top_values",
                                                               TensorClass::kTokenMetadata,
                                                               error_message);
        if (*top_values_buffer == nullptr) {
            return false;
        }
    }

    const std::size_t indices_bytes = sizeof(std::int32_t) * top_k;
    if (*top_indices_buffer == nullptr || (*top_indices_buffer)->GetSizeBytes() < indices_bytes) {
        *top_indices_buffer = MetalBuffer::CreateForTensorClass(context,
                                                                indices_bytes,
                                                                "sampler_top_indices",
                                                                TensorClass::kTokenMetadata,
                                                                error_message);
        if (*top_indices_buffer == nullptr) {
            return false;
        }
    }

    return true;
}

}  // namespace

Sampler::Sampler(SamplerConfig config) : config_(config), rng_(config.seed) {}

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
    const float temperature = std::max(config_.temperature, 1e-5f);
    const std::size_t top_k = std::max<std::size_t>(1, std::min<std::size_t>(config_.top_k, vocab_size));

    // The current GPU top-k kernel is a single-row scalar scan and measured
    // substantially slower than reading logits back to CPU on Apple M4.
    const bool should_use_cpu_sampler =
        pipeline_cache == nullptr ||
        top_k > SamplerTopKParams::kMaxTopK ||
        !UseExperimentalGpuSampler();
    if (should_use_cpu_sampler) {
        return SampleFromLogitsCpuFallback(logits,
                                           row_index,
                                           temperature,
                                           top_k,
                                           &rng_,
                                           token_id,
                                           top_logits,
                                           top_token_ids,
                                           error_message);
    }

    if (!EnsureTopKBuffers(context,
                           top_k,
                           &top_values_buffer_,
                           &top_indices_buffer_,
                           error_message)) {
        return false;
    }

    const DeviceTensor top_values_tensor(top_values_buffer_,
                                         0,
                                         TensorDesc::CreateContiguous(DataType::kFloat32, {1, top_k}));
    const DeviceTensor top_indices_tensor(top_indices_buffer_,
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
                            nullptr,
                            error_message)) {
        return false;
    }

    std::vector<float> reduced_logits(top_k, 0.0f);
    std::vector<std::int32_t> reduced_indices(top_k, -1);
    if (!top_values_buffer_->Read(reduced_logits.data(), sizeof(float) * top_k, 0, error_message) ||
        !top_indices_buffer_->Read(reduced_indices.data(), sizeof(std::int32_t) * top_k, 0, error_message)) {
        return false;
    }

    if (reduced_indices.empty()) {
        *token_id = -1;
    } else if (reduced_indices.size() == 1) {
        *token_id = reduced_indices[0];
    } else {
        std::vector<double> scaled_logits(reduced_logits.size(), 0.0);
        double max_score = -std::numeric_limits<double>::infinity();
        for (std::size_t rank = 0; rank < reduced_logits.size(); ++rank) {
            const double score = static_cast<double>(reduced_logits[rank]) / static_cast<double>(temperature);
            scaled_logits[rank] = score;
            if (score > max_score) {
                max_score = score;
            }
        }
        std::vector<double> weights(reduced_logits.size(), 0.0);
        for (std::size_t rank = 0; rank < reduced_logits.size(); ++rank) {
            weights[rank] = std::exp(scaled_logits[rank] - max_score);
        }
        std::discrete_distribution<std::size_t> distribution(weights.begin(), weights.end());
        *token_id = reduced_indices[distribution(rng_)];
    }
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
