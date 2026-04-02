#include "models/qwen3_5/qwen3_5_architecture.h"

namespace soc::gpu::models::qwen3_5 {

Qwen3_5ArchitectureSpec BuildQwen3_5_9BReferenceSpec() {
    Qwen3_5ArchitectureSpec spec;
    spec.layer_types.reserve(spec.num_hidden_layers);
    for (std::size_t layer_index = 0; layer_index < spec.num_hidden_layers; ++layer_index) {
        spec.layer_types.push_back((layer_index % 4) == 3 ? Qwen3_5LayerType::kGatedAttention
                                                          : Qwen3_5LayerType::kGatedDeltaNet);
    }
    return spec;
}

}  // namespace soc::gpu::models::qwen3_5
