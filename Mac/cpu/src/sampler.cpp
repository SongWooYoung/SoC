#include "header/sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void Require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}
}

Sampler::Sampler(SamplerConfig config) : config_(config) {
    Require(config_.temperature > 0.0f, "sampler temperature must be positive");
    Require(config_.top_k != 0, "sampler top_k must be positive");
}

const SamplerConfig& Sampler::config() const {
    return config_;
}

int Sampler::SampleFromLogits(const Tensor& logits, std::size_t batch_index, std::size_t position_index) const {
    Require(logits.dtype() == DType::Float32, "sampler logits must be float32");
    Require(logits.dim() == 3, "sampler logits must have shape [batch, seq, vocab]");
    Require(batch_index < logits.shape()[0], "sampler batch_index out of range");
    Require(position_index < logits.shape()[1], "sampler position_index out of range");

    const std::size_t vocab_size = logits.shape()[2];
    const float* logits_data = logits.data<const float>();
    const float* logits_row = logits_data + (batch_index * logits.shape()[1] + position_index) * vocab_size;

    std::vector<std::size_t> candidates(vocab_size);
    for (std::size_t index = 0; index < vocab_size; ++index) {
        candidates[index] = index;
    }

    const std::size_t top_k = std::min(config_.top_k, vocab_size);
    std::partial_sort(
        candidates.begin(),
        candidates.begin() + static_cast<std::ptrdiff_t>(top_k),
        candidates.end(),
        [logits_row, this](std::size_t left, std::size_t right) {
            return logits_row[left] / config_.temperature > logits_row[right] / config_.temperature;
        });

    std::size_t best_index = candidates[0];
    float best_score = logits_row[best_index] / config_.temperature;
    for (std::size_t rank = 1; rank < top_k; ++rank) {
        const std::size_t candidate = candidates[rank];
        const float score = logits_row[candidate] / config_.temperature;
        if (score > best_score) {
            best_score = score;
            best_index = candidate;
        }
    }

    return static_cast<int>(best_index);
}