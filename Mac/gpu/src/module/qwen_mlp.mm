#include "module/qwen_mlp.h"

#include "buffer/buffer_arena.h"
#include "op/elementwise_mul_op.h"
#include "op/linear_op.h"

namespace soc::gpu {
namespace {

bool AllocateTemporaryTensor(BufferArena* arena,
                             const TensorDesc& desc,
                             DeviceTensor* tensor,
                             std::string* error_message) {
    if (arena == nullptr) {
        if (error_message != nullptr) {
            *error_message = "Temporary arena is required for QwenMLP";
        }
        return false;
    }
    BufferArenaSlice slice;
    if (!arena->Allocate(desc.ByteSize(), 256, &slice, error_message)) {
        return false;
    }
    *tensor = DeviceTensor(slice.buffer, slice.offset_bytes, desc);
    return true;
}

bool ValidateMlpIO(const DeviceTensor& input,
                   const DeviceTensor& output,
                   const QwenMlpWeights& weights,
                   std::size_t* row_count,
                   std::size_t* hidden_size,
                   std::size_t* intermediate_size,
                   std::string* error_message) {
    if (!input.IsValid() || !output.IsValid() || !weights.gate_proj_weight.IsValid() ||
        !weights.up_proj_weight.IsValid() || !weights.down_proj_weight.IsValid()) {
        if (error_message != nullptr) {
            *error_message = "QwenMLP tensors must be valid";
        }
        return false;
    }
    if (input.GetDesc().GetDataType() != DataType::kFloat32 ||
        output.GetDesc().GetDataType() != DataType::kFloat32 ||
        weights.gate_proj_weight.GetDesc().GetDataType() != DataType::kFloat32 ||
        weights.up_proj_weight.GetDesc().GetDataType() != DataType::kFloat32 ||
        weights.down_proj_weight.GetDesc().GetDataType() != DataType::kFloat32) {
        if (error_message != nullptr) {
            *error_message = "QwenMLP baseline currently supports float32 tensors only";
        }
        return false;
    }
    if (input.GetDesc().Rank() != 2 || output.GetDesc().Rank() != 2 ||
        weights.gate_proj_weight.GetDesc().Rank() != 2 ||
        weights.up_proj_weight.GetDesc().Rank() != 2 ||
        weights.down_proj_weight.GetDesc().Rank() != 2) {
        if (error_message != nullptr) {
            *error_message = "QwenMLP expects rank-2 tensors";
        }
        return false;
    }

    const auto& input_shape = input.GetDesc().GetShape();
    const auto& output_shape = output.GetDesc().GetShape();
    const auto& gate_shape = weights.gate_proj_weight.GetDesc().GetShape();
    const auto& up_shape = weights.up_proj_weight.GetDesc().GetShape();
    const auto& down_shape = weights.down_proj_weight.GetDesc().GetShape();
    if (output_shape != input_shape) {
        if (error_message != nullptr) {
            *error_message = "QwenMLP output shape must match input shape";
        }
        return false;
    }
    if (gate_shape[1] != input_shape[1] || up_shape[1] != input_shape[1] || gate_shape[0] != up_shape[0]) {
        if (error_message != nullptr) {
            *error_message = "QwenMLP gate/up projection weights do not match input hidden size";
        }
        return false;
    }
    if (down_shape[0] != input_shape[1] || down_shape[1] != gate_shape[0]) {
        if (error_message != nullptr) {
            *error_message = "QwenMLP down projection weight shape does not match gate/up output size";
        }
        return false;
    }

    *row_count = input_shape[0];
    *hidden_size = input_shape[1];
    *intermediate_size = gate_shape[0];
    return true;
}

}  // namespace

bool QwenMLP::Run(const MetalContext& context,
                  PipelineCache* pipeline_cache,
                  const DeviceTensor& input,
                  const DeviceTensor* residual,
                  const QwenMlpWeights& weights,
                  const DeviceTensor& output,
                  const QwenMlpParams& params,
                  BufferArena* temporary_arena,
                  std::string* error_message) {
    if (pipeline_cache == nullptr) {
        if (error_message != nullptr) {
            *error_message = "Pipeline cache must not be null";
        }
        return false;
    }

    std::size_t row_count = 0;
    std::size_t hidden_size = 0;
    std::size_t inferred_intermediate_size = 0;
    if (!ValidateMlpIO(input, output, weights, &row_count, &hidden_size, &inferred_intermediate_size, error_message)) {
        return false;
    }
    if (params.intermediate_size != 0 && params.intermediate_size != inferred_intermediate_size) {
        if (error_message != nullptr) {
            *error_message = "QwenMLP intermediate size does not match weight shape";
        }
        return false;
    }
    if (params.add_residual && residual != nullptr) {
        if (!residual->IsValid() || residual->GetDesc().GetDataType() != DataType::kFloat32 ||
            residual->GetDesc().Rank() != 2 || residual->GetDesc().GetShape() != output.GetDesc().GetShape()) {
            if (error_message != nullptr) {
                *error_message = "QwenMLP residual must match output shape";
            }
            return false;
        }
    }

    BufferArenaMarkGuard arena_mark(temporary_arena, "QwenMLP");

    DeviceTensor gate_tensor;
    DeviceTensor up_tensor;
    DeviceTensor fused_tensor;
    const TensorDesc intermediate_desc =
        TensorDesc::CreateContiguous(DataType::kFloat32, {row_count, inferred_intermediate_size});
    if (!AllocateTemporaryTensor(temporary_arena, intermediate_desc, &gate_tensor, error_message) ||
        !AllocateTemporaryTensor(temporary_arena, intermediate_desc, &up_tensor, error_message) ||
        !AllocateTemporaryTensor(temporary_arena, intermediate_desc, &fused_tensor, error_message)) {
        return false;
    }

    LinearParams gate_params;
    gate_params.activation = LinearActivation::kSiLU;
    gate_params.matmul.row_count = static_cast<std::uint32_t>(row_count);
    gate_params.matmul.inner_dim = static_cast<std::uint32_t>(hidden_size);
    gate_params.matmul.column_count = static_cast<std::uint32_t>(inferred_intermediate_size);
    gate_params.matmul.decode_mode = row_count == 1;
    gate_params.matmul.transpose_rhs = true;
    if (!LinearOp::Run(context,
                       pipeline_cache,
                       input,
                       weights.gate_proj_weight,
                       nullptr,
                       nullptr,
                       gate_tensor,
                       gate_params,
                       temporary_arena,
                       error_message)) {
        return false;
    }

    LinearParams up_params;
    up_params.matmul.row_count = static_cast<std::uint32_t>(row_count);
    up_params.matmul.inner_dim = static_cast<std::uint32_t>(hidden_size);
    up_params.matmul.column_count = static_cast<std::uint32_t>(inferred_intermediate_size);
    up_params.matmul.decode_mode = row_count == 1;
    up_params.matmul.transpose_rhs = true;
    if (!LinearOp::Run(context,
                       pipeline_cache,
                       input,
                       weights.up_proj_weight,
                       nullptr,
                       nullptr,
                       up_tensor,
                       up_params,
                       temporary_arena,
                       error_message)) {
        return false;
    }

    ElementwiseMulParams mul_params;
    mul_params.row_count = static_cast<std::uint32_t>(row_count);
    mul_params.row_size = static_cast<std::uint32_t>(inferred_intermediate_size);
    if (!ElementwiseMulOp::Run(context,
                               pipeline_cache,
                               gate_tensor,
                               up_tensor,
                               fused_tensor,
                               mul_params,
                               temporary_arena,
                               error_message)) {
        return false;
    }

    LinearParams down_params;
    down_params.add_residual = params.add_residual;
    down_params.matmul.row_count = static_cast<std::uint32_t>(row_count);
    down_params.matmul.inner_dim = static_cast<std::uint32_t>(inferred_intermediate_size);
    down_params.matmul.column_count = static_cast<std::uint32_t>(hidden_size);
    down_params.matmul.decode_mode = row_count == 1;
    down_params.matmul.transpose_rhs = true;
    if (!LinearOp::Run(context,
                       pipeline_cache,
                       fused_tensor,
                       weights.down_proj_weight,
                       nullptr,
                       params.add_residual ? (residual == nullptr ? &input : residual) : nullptr,
                       output,
                       down_params,
                       temporary_arena,
                       error_message)) {
        return false;
    }

    return true;
}

}  // namespace soc::gpu