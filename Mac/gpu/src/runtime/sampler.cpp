#include "runtime/sampler.h"

#include <algorithm>
#include <cmath>

#include "buffer/metal_buffer.h"

namespace soc::gpu {

Sampler::Sampler(SamplerConfig config) : config_(config) {}

const SamplerConfig& Sampler::config() const { return config_; }

bool Sampler::SampleFromLogits(const DeviceTensor& logits,
                               std::size_t row_index,
                               int* token_id,
                               std::vector<float>* top_logits,
                               std::vector<int>* top_token_ids,
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
    std::vector<float> row(vocab_size, 0.0f);
    if (!logits.GetBuffer()->Read(row.data(), sizeof(float) * vocab_size, logits.GetByteOffset() + row_index * sizeof(float) * vocab_size, error_message)) {
        return false;
    }

    std::vector<std::size_t> candidates(vocab_size);
    for (std::size_t index = 0; index < vocab_size; ++index) {
        candidates[index] = index;
    }
    const std::size_t top_k = std::max<std::size_t>(1, std::min<std::size_t>(config_.top_k, vocab_size));
    const float temperature = config_.temperature <= 0.0f ? 1.0f : config_.temperature;
    std::partial_sort(candidates.begin(),
                      candidates.begin() + static_cast<std::ptrdiff_t>(top_k),
                      candidates.end(),
                      [&row, temperature](std::size_t lhs, std::size_t rhs) {
                          return row[lhs] / temperature > row[rhs] / temperature;
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

}  // namespace soc::gpu