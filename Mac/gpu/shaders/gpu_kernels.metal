#include <metal_stdlib>

using namespace metal;

constant bool kMatMulUseBias [[function_constant(0)]];
constant bool kMatMulDecodeMode [[function_constant(1)]];
constant bool kMatMulTransposeRhs [[function_constant(2)]];
constant uint kMatMulTileColumns [[function_constant(3)]];
constant uint kMatMulTileRows [[function_constant(4)]];
constant bool kEnableSiLU [[function_constant(5)]];
constant bool kEnableResidual [[function_constant(6)]];

kernel void bootstrap_copy(const device uint* input_values [[buffer(0)]],
						   device uint* output_values [[buffer(1)]],
						   uint gid [[thread_position_in_grid]]) {
	output_values[gid] = input_values[gid] + 1;
}

struct RmsNormParams {
	uint row_size;
	uint row_count;
	float epsilon;
	uint padding;
};

struct EmbeddingParams {
	uint token_count;
	uint hidden_size;
	uint vocab_size;
	uint padding;
};

struct RopeParams {
	uint row_count;
	uint head_count;
	uint head_dim;
	uint rotary_dim;
	uint position_offset;
	float rope_theta;
};

struct SoftmaxParams {
	uint row_count;
	uint row_size;
};

struct ElementwiseMulParams {
	uint row_count;
	uint row_size;
};

struct AttentionScoreParams {
	uint query_row_count;
	uint key_row_count;
	uint query_head_count;
	uint key_value_head_count;
	uint head_dim;
	uint head_group_size;
	uint query_position_base;
	uint causal_mask;
	float scale;
};

struct AttentionValueParams {
	uint query_row_count;
	uint key_row_count;
	uint query_head_count;
	uint key_value_head_count;
	uint head_dim;
	uint head_group_size;
};

struct MatMulParams {
	uint row_count;
	uint inner_dim;
	uint column_count;
	uint lhs_row_stride;
	uint rhs_row_stride;
	uint output_row_stride;
};

kernel void rms_norm_f32_rowwise(const device float* input_values [[buffer(0)]],
								 const device float* weight_values [[buffer(1)]],
								 device float* output_values [[buffer(2)]],
								 constant RmsNormParams& params [[buffer(3)]],
								 uint gid [[thread_position_in_grid]]) {
	if (gid >= params.row_count) {
		return;
	}

	const uint base = gid * params.row_size;
	float sum_squares = 0.0f;
	for (uint index = 0; index < params.row_size; ++index) {
		const float value = input_values[base + index];
		sum_squares += value * value;
	}

	const float inv_rms = rsqrt(sum_squares / static_cast<float>(params.row_size) + params.epsilon);
	for (uint index = 0; index < params.row_size; ++index) {
		output_values[base + index] = input_values[base + index] * inv_rms * weight_values[index];
	}
}

kernel void embedding_f32_lookup(const device int* token_ids [[buffer(0)]],
						 const device float* table_values [[buffer(1)]],
						 device float* output_values [[buffer(2)]],
						 constant EmbeddingParams& params [[buffer(3)]],
						 uint2 gid [[thread_position_in_grid]]) {
	if (gid.y >= params.token_count || gid.x >= params.hidden_size) {
		return;
	}
	const int token_id = token_ids[gid.y];
	const uint output_index = gid.y * params.hidden_size + gid.x;
	if (token_id < 0 || static_cast<uint>(token_id) >= params.vocab_size) {
		output_values[output_index] = 0.0f;
		return;
	}
	output_values[output_index] = table_values[static_cast<uint>(token_id) * params.hidden_size + gid.x];
}

kernel void embedding_f16_lookup(const device int* token_ids [[buffer(0)]],
						 const device half* table_values [[buffer(1)]],
						 device float* output_values [[buffer(2)]],
						 constant EmbeddingParams& params [[buffer(3)]],
						 uint2 gid [[thread_position_in_grid]]) {
	if (gid.y >= params.token_count || gid.x >= params.hidden_size) {
		return;
	}
	const int token_id = token_ids[gid.y];
	const uint output_index = gid.y * params.hidden_size + gid.x;
	if (token_id < 0 || static_cast<uint>(token_id) >= params.vocab_size) {
		output_values[output_index] = 0.0f;
		return;
	}
	output_values[output_index] = static_cast<float>(table_values[static_cast<uint>(token_id) * params.hidden_size + gid.x]);
}

kernel void rope_f32_qwen(const device float* input_values [[buffer(0)]],
					device float* output_values [[buffer(1)]],
					constant RopeParams& params [[buffer(2)]],
					uint2 gid [[thread_position_in_grid]]) {
	const uint pair_count_per_head = params.rotary_dim / 2;
	if (gid.y >= params.row_count || gid.x >= params.head_count * pair_count_per_head) {
		return;
	}
	const uint head_index = gid.x / pair_count_per_head;
	const uint pair_index = gid.x % pair_count_per_head;
	const uint row_index = gid.y;
	const uint hidden_stride = params.head_count * params.head_dim;
	const uint base_index = row_index * hidden_stride + head_index * params.head_dim;
	const uint left_index = base_index + pair_index;
	const uint right_index = base_index + pair_index + pair_count_per_head;
	const float position = static_cast<float>(row_index + params.position_offset);
	const float exponent = (2.0f * static_cast<float>(pair_index)) / static_cast<float>(params.rotary_dim);
	const float inv_freq = pow(params.rope_theta, -exponent);
	const float angle = position * inv_freq;
	const float cosine = cos(angle);
	const float sine = sin(angle);
	const float left_value = input_values[left_index];
	const float right_value = input_values[right_index];
	output_values[left_index] = left_value * cosine - right_value * sine;
	output_values[right_index] = left_value * sine + right_value * cosine;

	for (uint passthrough = params.rotary_dim; passthrough < params.head_dim; ++passthrough) {
		if (pair_index == 0) {
			const uint passthrough_index = base_index + passthrough;
			output_values[passthrough_index] = input_values[passthrough_index];
		}
	}
}

kernel void softmax_f32_rowwise(const device float* input_values [[buffer(0)]],
					device float* output_values [[buffer(1)]],
					constant SoftmaxParams& params [[buffer(2)]],
					uint gid [[thread_position_in_grid]]) {
	if (gid >= params.row_count) {
		return;
	}
	const uint base = gid * params.row_size;
	float max_value = input_values[base];
	for (uint index = 1; index < params.row_size; ++index) {
		max_value = max(max_value, input_values[base + index]);
	}
	float sum_value = 0.0f;
	for (uint index = 0; index < params.row_size; ++index) {
		const float exponent = exp(input_values[base + index] - max_value);
		output_values[base + index] = exponent;
		sum_value += exponent;
	}
	for (uint index = 0; index < params.row_size; ++index) {
		output_values[base + index] = output_values[base + index] / sum_value;
	}
}

kernel void elementwise_mul_f32(const device float* lhs_values [[buffer(0)]],
					const device float* rhs_values [[buffer(1)]],
					device float* output_values [[buffer(2)]],
					constant ElementwiseMulParams& params [[buffer(3)]],
					uint2 gid [[thread_position_in_grid]]) {
	if (gid.y >= params.row_count || gid.x >= params.row_size) {
		return;
	}
	const uint index = gid.y * params.row_size + gid.x;
	output_values[index] = lhs_values[index] * rhs_values[index];
}

kernel void attention_scores_f32_qwen(const device float* query_values [[buffer(0)]],
					  const device float* key_values [[buffer(1)]],
					  device float* score_values [[buffer(2)]],
					  constant AttentionScoreParams& params [[buffer(3)]],
					  uint2 gid [[thread_position_in_grid]]) {
	if (gid.y >= params.query_row_count * params.query_head_count || gid.x >= params.key_row_count) {
		return;
	}
	const uint query_row = gid.y / params.query_head_count;
	const uint query_head = gid.y % params.query_head_count;
	const uint key_row = gid.x;
	const uint kv_head = query_head / params.head_group_size;
	if (params.causal_mask != 0 && key_row > params.query_position_base + query_row) {
		score_values[gid.y * params.key_row_count + gid.x] = -INFINITY;
		return;
	}
	const uint query_base = query_row * (params.query_head_count * params.head_dim) + query_head * params.head_dim;
	const uint key_base = key_row * (params.key_value_head_count * params.head_dim) + kv_head * params.head_dim;
	float accumulator = 0.0f;
	for (uint dim = 0; dim < params.head_dim; ++dim) {
		accumulator += query_values[query_base + dim] * key_values[key_base + dim];
	}
	score_values[gid.y * params.key_row_count + gid.x] = accumulator * params.scale;
}

kernel void attention_values_f32_qwen(const device float* probability_values [[buffer(0)]],
					  const device float* value_values [[buffer(1)]],
					  device float* output_values [[buffer(2)]],
					  constant AttentionValueParams& params [[buffer(3)]],
					  uint2 gid [[thread_position_in_grid]]) {
	if (gid.y >= params.query_row_count * params.query_head_count || gid.x >= params.head_dim) {
		return;
	}
	const uint query_row = gid.y / params.query_head_count;
	const uint query_head = gid.y % params.query_head_count;
	const uint kv_head = query_head / params.head_group_size;
	float accumulator = 0.0f;
	for (uint key_row = 0; key_row < params.key_row_count; ++key_row) {
		const float probability = probability_values[gid.y * params.key_row_count + key_row];
		const uint value_base = key_row * (params.key_value_head_count * params.head_dim) + kv_head * params.head_dim;
		accumulator += probability * value_values[value_base + gid.x];
	}
	const uint output_base = query_row * (params.query_head_count * params.head_dim) + query_head * params.head_dim;
	output_values[output_base + gid.x] = accumulator;
}

kernel void matmul_f32_basic(const device float* lhs_values [[buffer(0)]],
						 const device float* rhs_values [[buffer(1)]],
						 const device float* bias_values [[buffer(2)]],
						 device float* output_values [[buffer(3)]],
						 constant MatMulParams& params [[buffer(4)]],
						 const device float* residual_values [[buffer(5)]],
						 uint2 gid [[thread_position_in_grid]]) {
	if (gid.y >= params.row_count || gid.x >= params.column_count) {
		return;
	}
	if (kMatMulTileColumns == 0 || kMatMulTileRows == 0) {
		return;
	}

	const uint row_index = gid.y;
	const uint lhs_base = kMatMulDecodeMode ? 0u : row_index * params.lhs_row_stride;
	float accumulator = 0.0f;
	for (uint inner_index = 0; inner_index < params.inner_dim; ++inner_index) {
		const uint rhs_index = kMatMulTransposeRhs
			? gid.x * params.rhs_row_stride + inner_index
			: inner_index * params.rhs_row_stride + gid.x;
		accumulator += lhs_values[lhs_base + inner_index] * rhs_values[rhs_index];
	}
	if (kMatMulUseBias) {
		accumulator += bias_values[gid.x];
	}
	if (kEnableResidual) {
		const uint residual_index = kMatMulDecodeMode ? gid.x : row_index * params.output_row_stride + gid.x;
		accumulator += residual_values[residual_index];
	}
	if (kEnableSiLU) {
		accumulator = accumulator / (1.0f + exp(-accumulator));
	}

	const uint output_index = kMatMulDecodeMode ? gid.x : row_index * params.output_row_stride + gid.x;
	output_values[output_index] = accumulator;
}

kernel void matmul_f32_tiled(const device float* lhs_values [[buffer(0)]],
					 const device float* rhs_values [[buffer(1)]],
					 const device float* bias_values [[buffer(2)]],
					 device float* output_values [[buffer(3)]],
					 constant MatMulParams& params [[buffer(4)]],
					 const device float* residual_values [[buffer(5)]],
					 uint2 tid [[thread_position_in_threadgroup]],
					 uint2 tgid [[threadgroup_position_in_grid]]) {
	if (kMatMulTileColumns == 0 || kMatMulTileRows == 0 || kMatMulTileColumns > 32 || kMatMulTileRows > 4) {
		return;
	}

	threadgroup float lhs_tile[4][16];
	threadgroup float rhs_tile[16][32];

	const uint row_index = tgid.y * kMatMulTileRows + tid.y;
	const uint col_index = tgid.x * kMatMulTileColumns + tid.x;
	const uint linear_tid = tid.y * kMatMulTileColumns + tid.x;
	const uint threads_per_group = kMatMulTileColumns * kMatMulTileRows;
	float accumulator = 0.0f;

	for (uint tile_start = 0; tile_start < params.inner_dim; tile_start += 16) {
		const uint lhs_load_count = kMatMulTileRows * 16;
		for (uint load_index = linear_tid; load_index < lhs_load_count; load_index += threads_per_group) {
			const uint local_row = load_index / 16;
			const uint local_k = load_index % 16;
			const uint global_row = tgid.y * kMatMulTileRows + local_row;
			const uint global_k = tile_start + local_k;
			lhs_tile[local_row][local_k] = (global_row < params.row_count && global_k < params.inner_dim)
				? lhs_values[global_row * params.lhs_row_stride + global_k]
				: 0.0f;
		}

		const uint rhs_load_count = 16 * kMatMulTileColumns;
		for (uint load_index = linear_tid; load_index < rhs_load_count; load_index += threads_per_group) {
			const uint local_k = load_index / kMatMulTileColumns;
			const uint local_col = load_index % kMatMulTileColumns;
			const uint global_k = tile_start + local_k;
			const uint global_col = tgid.x * kMatMulTileColumns + local_col;
			const uint rhs_index = kMatMulTransposeRhs
				? global_col * params.rhs_row_stride + global_k
				: global_k * params.rhs_row_stride + global_col;
			rhs_tile[local_k][local_col] = (global_k < params.inner_dim && global_col < params.column_count)
				? rhs_values[rhs_index]
				: 0.0f;
		}

		threadgroup_barrier(mem_flags::mem_threadgroup);
		if (row_index < params.row_count && col_index < params.column_count) {
			const uint tile_extent = min(16u, params.inner_dim - tile_start);
			for (uint local_k = 0; local_k < tile_extent; ++local_k) {
				accumulator += lhs_tile[tid.y][local_k] * rhs_tile[local_k][tid.x];
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}

	if (row_index >= params.row_count || col_index >= params.column_count) {
		return;
	}
	if (kMatMulUseBias) {
		accumulator += bias_values[col_index];
	}
	if (kEnableResidual) {
		accumulator += residual_values[row_index * params.output_row_stride + col_index];
	}
	if (kEnableSiLU) {
		accumulator = accumulator / (1.0f + exp(-accumulator));
	}
	output_values[row_index * params.output_row_stride + col_index] = accumulator;
}

kernel void matmul_f32_decode_tiled(const device float* lhs_values [[buffer(0)]],
					    const device float* rhs_values [[buffer(1)]],
					    const device float* bias_values [[buffer(2)]],
					    device float* output_values [[buffer(3)]],
					    constant MatMulParams& params [[buffer(4)]],
					    const device float* residual_values [[buffer(5)]],
					    uint tid [[thread_index_in_threadgroup]],
					    uint tgid [[threadgroup_position_in_grid]]) {
	if (kMatMulTileColumns == 0 || kMatMulTileColumns > 32) {
		return;
	}

	threadgroup float lhs_tile[16];
	threadgroup float rhs_tile[16][32];
	const uint col_index = tgid * kMatMulTileColumns + tid;
	const uint threads_per_group = kMatMulTileColumns;
	float accumulator = 0.0f;

	for (uint tile_start = 0; tile_start < params.inner_dim; tile_start += 16) {
		for (uint load_index = tid; load_index < 16; load_index += threads_per_group) {
			const uint global_k = tile_start + load_index;
			lhs_tile[load_index] = global_k < params.inner_dim ? lhs_values[global_k] : 0.0f;
		}
		for (uint load_index = tid; load_index < 16 * kMatMulTileColumns; load_index += threads_per_group) {
			const uint local_k = load_index / kMatMulTileColumns;
			const uint local_col = load_index % kMatMulTileColumns;
			const uint global_k = tile_start + local_k;
			const uint global_col = tgid * kMatMulTileColumns + local_col;
			const uint rhs_index = kMatMulTransposeRhs
				? global_col * params.rhs_row_stride + global_k
				: global_k * params.rhs_row_stride + global_col;
			rhs_tile[local_k][local_col] = (global_col < params.column_count && global_k < params.inner_dim)
				? rhs_values[rhs_index]
				: 0.0f;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if (tid < kMatMulTileColumns && col_index < params.column_count) {
			const uint tile_extent = min(16u, params.inner_dim - tile_start);
			for (uint local_k = 0; local_k < tile_extent; ++local_k) {
				accumulator += lhs_tile[local_k] * rhs_tile[local_k][tid];
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}

	if (tid >= kMatMulTileColumns || col_index >= params.column_count) {
		return;
	}
	if (kMatMulUseBias) {
		accumulator += bias_values[col_index];
	}
	if (kEnableResidual) {
		accumulator += residual_values[col_index];
	}
	if (kEnableSiLU) {
		accumulator = accumulator / (1.0f + exp(-accumulator));
	}
	output_values[col_index] = accumulator;
}