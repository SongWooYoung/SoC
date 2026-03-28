#pragma once

#include <memory>
#include <string>

#include "kernel/kernel_key.h"

namespace soc::gpu {

class MetalContext;

class PipelineCache {
public:
    explicit PipelineCache(const MetalContext& context);
    ~PipelineCache();

    bool WarmPipeline(const KernelKey& key, std::string* error_message);
    bool HasPipeline(const KernelKey& key) const;
    const void* GetOrCreatePipeline(const KernelKey& key, std::string* error_message);

private:
    struct Impl;

    std::unique_ptr<Impl> impl_;
};

}  // namespace soc::gpu