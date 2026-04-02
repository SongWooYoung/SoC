#include "header/sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
void Require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}
}

Sampler::Sampler(SamplerConfig config) : config_(config), rng_(config.seed) {
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

    if (top_k == 1) {
        return static_cast<int>(candidates[0]);
    }

    std::vector<double> scaled_logits(top_k, 0.0);
    double max_score = -std::numeric_limits<double>::infinity();
    for (std::size_t rank = 0; rank < top_k; ++rank) {
        const double score = static_cast<double>(logits_row[candidates[rank]]) / static_cast<double>(config_.temperature);
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
    const std::size_t sampled_rank = distribution(rng_);
    return static_cast<int>(candidates[sampled_rank]);
}
