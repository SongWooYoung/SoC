#pragma once

#include <cstdint>
#include <string>

namespace soc::gpu {

enum class KernelKind {
    kBootstrap,
    kEmbedding,
    kElementwiseMul,
    kRmsNorm,
    kRoPE,
    kSoftmax,
    kAttentionScores,
    kAttentionValues,
    kMatMul,
};

struct KernelSpecializationKey {
    bool use_bias = false;
    bool decode_mode = false;
    bool transpose_rhs = false;
    bool enable_silu = false;
    bool enable_residual = false;
    std::uint32_t tile_columns = 1;
    std::uint32_t tile_rows = 1;

    bool operator<(const KernelSpecializationKey& other) const;
    bool operator==(const KernelSpecializationKey& other) const;
};

struct KernelKey {
    KernelKind kind = KernelKind::kBootstrap;
    std::string function_name;
    std::uint32_t threadgroup_width = 1;
    std::uint32_t threadgroup_height = 1;
    KernelSpecializationKey specialization;

    bool operator<(const KernelKey& other) const;
    bool operator==(const KernelKey& other) const;
};

}  // namespace soc::gpu