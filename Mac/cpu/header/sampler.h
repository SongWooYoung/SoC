#ifndef SAMPLER_H
#define SAMPLER_H

#include <cstddef>
#include <vector>

#include "header/tensor.h"

struct SamplerConfig {
    float temperature = 1.0f;
    std::size_t top_k = 1;
};

class Sampler {
public:
    explicit Sampler(SamplerConfig config = {});

    const SamplerConfig& config() const;
    int SampleFromLogits(const Tensor& logits, std::size_t batch_index = 0, std::size_t position_index = 0) const;

private:
    SamplerConfig config_;
};

#endif