#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "kernel/pipeline_cache.h"
#include "tensor/device_tensor.h"

namespace soc::gpu {

class BufferArena;
class MetalContext;

struct SamplerConfig {
    float temperature = 1.0f;
    std::size_t top_k = 1;
};

class Sampler {
public:
    explicit Sampler(SamplerConfig config = {});

    const SamplerConfig& config() const;
    bool SampleFromLogits(const MetalContext& context,
                          PipelineCache* pipeline_cache,
                          const DeviceTensor& logits,
                          std::size_t row_index,
                          int* token_id,
                          std::vector<float>* top_logits,
                          std::vector<int>* top_token_ids,
                          BufferArena* temporary_arena,
                          std::string* error_message) const;

private:
    SamplerConfig config_;
};

}  // namespace soc::gpu