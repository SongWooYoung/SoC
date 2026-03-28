#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "tensor/device_tensor.h"

namespace soc::gpu {

struct SamplerConfig {
    float temperature = 1.0f;
    std::size_t top_k = 1;
};

class Sampler {
public:
    explicit Sampler(SamplerConfig config = {});

    const SamplerConfig& config() const;
    bool SampleFromLogits(const DeviceTensor& logits,
                          std::size_t row_index,
                          int* token_id,
                          std::vector<float>* top_logits,
                          std::vector<int>* top_token_ids,
                          std::string* error_message) const;

private:
    SamplerConfig config_;
};

}  // namespace soc::gpu