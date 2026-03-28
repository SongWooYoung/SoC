#include "kernel/kernel_key.h"

#include <tuple>

namespace soc::gpu {

bool KernelSpecializationKey::operator<(const KernelSpecializationKey& other) const {
    return std::tie(use_bias, decode_mode, transpose_rhs, enable_silu, enable_residual, tile_columns, tile_rows) <
           std::tie(other.use_bias,
                    other.decode_mode,
                    other.transpose_rhs,
                    other.enable_silu,
                    other.enable_residual,
                    other.tile_columns,
                    other.tile_rows);
}

bool KernelSpecializationKey::operator==(const KernelSpecializationKey& other) const {
    return use_bias == other.use_bias && decode_mode == other.decode_mode &&
           transpose_rhs == other.transpose_rhs &&
           enable_silu == other.enable_silu && enable_residual == other.enable_residual &&
           tile_columns == other.tile_columns && tile_rows == other.tile_rows;
}

bool KernelKey::operator<(const KernelKey& other) const {
    return std::tie(kind, function_name, threadgroup_width, threadgroup_height, specialization) <
           std::tie(other.kind,
                    other.function_name,
                    other.threadgroup_width,
                    other.threadgroup_height,
                    other.specialization);
}

bool KernelKey::operator==(const KernelKey& other) const {
    return kind == other.kind && function_name == other.function_name &&
           threadgroup_width == other.threadgroup_width &&
           threadgroup_height == other.threadgroup_height &&
           specialization == other.specialization;
}

}  // namespace soc::gpu