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

struct SamplerTopKParams {
	uint row_count;
	uint row_size;
	uint top_k;
	uint padding;
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

struct AffineQmmParams {
	uint row_count;
	uint inner_dim;
	uint column_count;
	uint output_row_stride;
	uint packed_inner_dim;
	uint groups_per_row;
	uint group_size;
	uint bits;
	uint enable_silu;
	uint add_residual;
	uint padding;
};

inline float4 unpack_nibbles4(uint packed, uint shift_base, float scale, float bias) {
	const float4 nibble_values = float4(float((packed >> shift_base) & 0xFu),
	                                   float((packed >> (shift_base + 4)) & 0xFu),
	                                   float((packed >> (shift_base + 8)) & 0xFu),
	                                   float((packed >> (shift_base + 12)) & 0xFu));
	return nibble_values * scale + bias;
}

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

kernel void rms_norm_f32_rowwise_simd(const device float* input_values [[buffer(0)]],
									  const device float* weight_values [[buffer(1)]],
									  device float* output_values [[buffer(2)]],
									  constant RmsNormParams& params [[buffer(3)]],
									  uint lane [[thread_index_in_threadgroup]],
									  uint row_index [[threadgroup_position_in_grid]]) {
	if (row_index >= params.row_count) {
		return;
	}

	const uint base = row_index * params.row_size;
	float partial_sum = 0.0f;
	for (uint index = lane; index < params.row_size; index += 32) {
		const float value = input_values[base + index];
		partial_sum += value * value;
	}
	const float sum_squares = simd_sum(partial_sum);
	const float inv_rms = rsqrt(sum_squares / static_cast<float>(params.row_size) + params.epsilon);
	for (uint index = lane; index < params.row_size; index += 32) {
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

kernel void sampler_topk_f32_rowwise(const device float* input_values [[buffer(0)]],
					 device float* top_values [[buffer(1)]],
					 device int* top_indices [[buffer(2)]],
					 constant SamplerTopKParams& params [[buffer(3)]],
					 uint gid [[thread_position_in_grid]]) {
	constexpr uint kMaxTopK = 64;
	if (gid >= params.row_count || params.top_k == 0 || params.top_k > kMaxTopK) {
		return;
	}

	float best_values[kMaxTopK];
	int best_indices[kMaxTopK];
	for (uint rank = 0; rank < params.top_k; ++rank) {
		best_values[rank] = -INFINITY;
		best_indices[rank] = -1;
	}

	const uint row_base = gid * params.row_size;
	for (uint vocab_index = 0; vocab_index < params.row_size; ++vocab_index) {
		const float value = input_values[row_base + vocab_index];
		for (uint rank = 0; rank < params.top_k; ++rank) {
			if (value > best_values[rank]) {
				for (uint shift = params.top_k - 1; shift > rank; --shift) {
					best_values[shift] = best_values[shift - 1];
					best_indices[shift] = best_indices[shift - 1];
				}
				best_values[rank] = value;
				best_indices[rank] = static_cast<int>(vocab_index);
				break;
			}
		}
	}

	const uint output_base = gid * params.top_k;
	for (uint rank = 0; rank < params.top_k; ++rank) {
		top_values[output_base + rank] = best_values[rank];
		top_indices[output_base + rank] = best_indices[rank];
	}
}

kernel void affine_qmm_t_4bit(const device float* lhs_values [[buffer(0)]],
			      const device uint* qweight_values [[buffer(1)]],
			      const device float* scale_values [[buffer(2)]],
			      const device float* qbias_values [[buffer(3)]],
			      const device float* residual_values [[buffer(4)]],
			      device float* output_values [[buffer(5)]],
			      constant AffineQmmParams& params [[buffer(6)]],
			      uint tid [[thread_index_in_threadgroup]],
			      uint2 tgid [[threadgroup_position_in_grid]]) {
	if (params.bits != 4 || params.group_size == 0) {
		return;
	}
	const uint row_index = tgid.y;
	const uint col_index = tgid.x * 32 + tid;
	if (row_index >= params.row_count || col_index >= params.column_count) {
		return;
	}

	threadgroup float4 lhs_tile[1024];
	const uint lhs_vec_count = params.inner_dim / 4;
	for (uint index = tid; index < lhs_vec_count; index += 32) {
		const uint lhs_offset = row_index * params.inner_dim + index * 4;
		lhs_tile[index] = *reinterpret_cast<const device float4*>(lhs_values + lhs_offset);
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);

	const uint packed_per_group = params.group_size / 8;
	float accumulator = 0.0f;
	for (uint group_index = 0; group_index < params.groups_per_row; ++group_index) {
		const float scale = scale_values[col_index * params.groups_per_row + group_index];
		const float bias = qbias_values[col_index * params.groups_per_row + group_index];
		const uint packed_base = group_index * packed_per_group;
		const uint x_base = group_index * params.group_size;
		for (uint packed_index = 0; packed_index < packed_per_group; ++packed_index) {
			const uint packed = qweight_values[col_index * params.packed_inner_dim + packed_base + packed_index];
			const uint lhs_base = (x_base + packed_index * 8) / 4;
			const float4 lhs0 = lhs_tile[lhs_base];
			const float4 lhs1 = lhs_tile[lhs_base + 1];
			const float4 q0 = unpack_nibbles4(packed, 0, scale, bias);
			const float4 q1 = unpack_nibbles4(packed, 16, scale, bias);
			accumulator += dot(lhs0, q0);
			accumulator += dot(lhs1, q1);
		}
	}

	if (params.add_residual != 0) {
		accumulator += residual_values[row_index * params.output_row_stride + col_index];
	}
	if (params.enable_silu != 0) {
		accumulator = accumulator / (1.0f + exp(-accumulator));
	}
	output_values[row_index * params.output_row_stride + col_index] = accumulator;
}

kernel void affine_qmm_t_4bit_mlp2(const device float* lhs_values [[buffer(0)]],
				   const device uint* qweight_values [[buffer(1)]],
				   const device float* scale_values [[buffer(2)]],
				   const device float* qbias_values [[buffer(3)]],
				   const device float* residual_values [[buffer(4)]],
				   device float* output_values [[buffer(5)]],
				   constant AffineQmmParams& params [[buffer(6)]],
				   uint tid [[thread_index_in_threadgroup]],
				   uint2 tgid [[threadgroup_position_in_grid]]) {
	if (params.bits != 4 || params.group_size == 0) {
		return;
	}
	constexpr uint kOutputsPerThread = 2;
	const uint row_index = tgid.y;
	const uint col_base = tgid.x * 32 + tid * kOutputsPerThread;
	if (row_index >= params.row_count) {
		return;
	}

	threadgroup float4 lhs_tile[1024];
	const uint lhs_vec_count = params.inner_dim / 4;
	for (uint index = tid; index < lhs_vec_count; index += 16) {
		const uint lhs_offset = row_index * params.inner_dim + index * 4;
		lhs_tile[index] = *reinterpret_cast<const device float4*>(lhs_values + lhs_offset);
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);

	float2 accumulators = float2(0.0f);
	const uint packed_per_group = params.group_size / 8;
	for (uint group_index = 0; group_index < params.groups_per_row; ++group_index) {
		const uint packed_base = group_index * packed_per_group;
		const uint x_base = (group_index * params.group_size) / 4;
		for (uint packed_index = 0; packed_index < packed_per_group; ++packed_index) {
			const uint packed_row_index = packed_base + packed_index;
			const float4 lhs0 = lhs_tile[x_base + packed_index * 2];
			const float4 lhs1 = lhs_tile[x_base + packed_index * 2 + 1];
			for (uint lane = 0; lane < kOutputsPerThread; ++lane) {
				const uint col_index = col_base + lane;
				if (col_index >= params.column_count) {
					continue;
				}
				const float scale = scale_values[col_index * params.groups_per_row + group_index];
				const float bias = qbias_values[col_index * params.groups_per_row + group_index];
				const uint packed = qweight_values[col_index * params.packed_inner_dim + packed_row_index];
				const float4 q0 = unpack_nibbles4(packed, 0, scale, bias);
				const float4 q1 = unpack_nibbles4(packed, 16, scale, bias);
				accumulators[lane] += dot(lhs0, q0);
				accumulators[lane] += dot(lhs1, q1);
			}
		}
	}

	for (uint lane = 0; lane < kOutputsPerThread; ++lane) {
		const uint col_index = col_base + lane;
		if (col_index >= params.column_count) {
			continue;
		}
		float accumulator = accumulators[lane];
		if (params.add_residual != 0) {
			accumulator += residual_values[row_index * params.output_row_stride + col_index];
		}
		if (params.enable_silu != 0) {
			accumulator = accumulator / (1.0f + exp(-accumulator));
		}
		output_values[row_index * params.output_row_stride + col_index] = accumulator;
	}
}

kernel void affine_qmm_t_4bit_lmhead2(const device float* lhs_values [[buffer(0)]],
				      const device uint* qweight_values [[buffer(1)]],
				      const device float* scale_values [[buffer(2)]],
				      const device float* qbias_values [[buffer(3)]],
				      const device float* residual_values [[buffer(4)]],
				      device float* output_values [[buffer(5)]],
				      constant AffineQmmParams& params [[buffer(6)]],
				      uint tid [[thread_index_in_threadgroup]],
				      uint2 tgid [[threadgroup_position_in_grid]]) {
	if (params.bits != 4 || params.group_size == 0) {
		return;
	}
	constexpr uint kOutputsPerThread = 2;
	const uint row_index = tgid.y;
	const uint col_base = tgid.x * 64 + tid * kOutputsPerThread;
	if (row_index >= params.row_count) {
		return;
	}

	threadgroup float4 lhs_tile[1024];
	const uint lhs_vec_count = params.inner_dim / 4;
	for (uint index = tid; index < lhs_vec_count; index += 32) {
		const uint lhs_offset = row_index * params.inner_dim + index * 4;
		lhs_tile[index] = *reinterpret_cast<const device float4*>(lhs_values + lhs_offset);
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);

	float2 accumulators = float2(0.0f);
	const uint packed_per_group = params.group_size / 8;
	for (uint group_index = 0; group_index < params.groups_per_row; ++group_index) {
		const float2 scales = float2(scale_values[(col_base + 0) * params.groups_per_row + group_index],
					     scale_values[(col_base + 1) * params.groups_per_row + group_index]);
		const float2 biases = float2(qbias_values[(col_base + 0) * params.groups_per_row + group_index],
					     qbias_values[(col_base + 1) * params.groups_per_row + group_index]);
		const uint packed_base = group_index * packed_per_group;
		const uint x_base = (group_index * params.group_size) / 4;
		for (uint packed_index = 0; packed_index < packed_per_group; ++packed_index) {
			const uint packed_row_index = packed_base + packed_index;
			const float4 lhs0 = lhs_tile[x_base + packed_index * 2];
			const float4 lhs1 = lhs_tile[x_base + packed_index * 2 + 1];
			const uint packed0 = qweight_values[(col_base + 0) * params.packed_inner_dim + packed_row_index];
			const uint packed1 = qweight_values[(col_base + 1) * params.packed_inner_dim + packed_row_index];
			const float4 q00 = unpack_nibbles4(packed0, 0, scales[0], biases[0]);
			const float4 q01 = unpack_nibbles4(packed0, 16, scales[0], biases[0]);
			const float4 q10 = unpack_nibbles4(packed1, 0, scales[1], biases[1]);
			const float4 q11 = unpack_nibbles4(packed1, 16, scales[1], biases[1]);
			accumulators[0] += dot(lhs0, q00) + dot(lhs1, q01);
			accumulators[1] += dot(lhs0, q10) + dot(lhs1, q11);
		}
	}

	output_values[row_index * params.output_row_stride + col_base + 0] = accumulators[0];
	output_values[row_index * params.output_row_stride + col_base + 1] = accumulators[1];
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

	constexpr uint kMatMulTileDepth = 32;
	threadgroup float lhs_tile[4][kMatMulTileDepth];
	threadgroup float rhs_tile[kMatMulTileDepth][32];

	const uint row_index = tgid.y * kMatMulTileRows + tid.y;
	const uint col_index = tgid.x * kMatMulTileColumns + tid.x;
	const uint linear_tid = tid.y * kMatMulTileColumns + tid.x;
	const uint threads_per_group = kMatMulTileColumns * kMatMulTileRows;
	float accumulator = 0.0f;

	for (uint tile_start = 0; tile_start < params.inner_dim; tile_start += kMatMulTileDepth) {
		const uint lhs_load_count = kMatMulTileRows * kMatMulTileDepth;
		for (uint load_index = linear_tid; load_index < lhs_load_count; load_index += threads_per_group) {
			const uint local_row = load_index / kMatMulTileDepth;
			const uint local_k = load_index % kMatMulTileDepth;
			const uint global_row = tgid.y * kMatMulTileRows + local_row;
			const uint global_k = tile_start + local_k;
			lhs_tile[local_row][local_k] = (global_row < params.row_count && global_k < params.inner_dim)
				? lhs_values[global_row * params.lhs_row_stride + global_k]
				: 0.0f;
		}

		const uint rhs_load_count = kMatMulTileDepth * kMatMulTileColumns;
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
			const uint tile_extent = min(kMatMulTileDepth, params.inner_dim - tile_start);
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

	constexpr uint kMatMulTileDepth = 32;
	threadgroup float lhs_tile[kMatMulTileDepth];
	threadgroup float rhs_tile[kMatMulTileDepth][32];
	const uint col_index = tgid * kMatMulTileColumns + tid;
	const uint threads_per_group = kMatMulTileColumns;
	float accumulator = 0.0f;

	for (uint tile_start = 0; tile_start < params.inner_dim; tile_start += kMatMulTileDepth) {
		for (uint load_index = tid; load_index < kMatMulTileDepth; load_index += threads_per_group) {
			const uint global_k = tile_start + load_index;
			lhs_tile[load_index] = global_k < params.inner_dim ? lhs_values[global_k] : 0.0f;
		}
		for (uint load_index = tid; load_index < kMatMulTileDepth * kMatMulTileColumns; load_index += threads_per_group) {
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
			const uint tile_extent = min(kMatMulTileDepth, params.inner_dim - tile_start);
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

kernel void matmul_f32_decode_tiled_vec4(const device float* lhs_values [[buffer(0)]],
					 const device float* rhs_values [[buffer(1)]],
					 const device float* bias_values [[buffer(2)]],
					 device float* output_values [[buffer(3)]],
					 constant MatMulParams& params [[buffer(4)]],
					 const device float* residual_values [[buffer(5)]],
					 uint tid [[thread_index_in_threadgroup]],
					 uint tgid [[threadgroup_position_in_grid]]) {
	if (kMatMulTileColumns == 0 || kMatMulTileColumns > 32 || !kMatMulTransposeRhs) {
		return;
	}

	constexpr uint kMatMulTileDepth = 32;
	constexpr uint kVecWidth = 4;
	constexpr uint kTileVecCount = kMatMulTileDepth / kVecWidth;
	threadgroup float4 lhs_tile[kTileVecCount];
	threadgroup float4 rhs_tile[kTileVecCount][32];
	const uint col_index = tgid * kMatMulTileColumns + tid;
	const uint threads_per_group = kMatMulTileColumns;
	float accumulator = 0.0f;

	for (uint tile_start = 0; tile_start < params.inner_dim; tile_start += kMatMulTileDepth) {
		for (uint load_index = tid; load_index < kTileVecCount; load_index += threads_per_group) {
			const uint global_k = tile_start + load_index * kVecWidth;
			if (global_k + 3 < params.inner_dim) {
				lhs_tile[load_index] = *reinterpret_cast<const device float4*>(lhs_values + global_k);
			} else {
				float4 tail = float4(0.0f);
				for (uint lane = 0; lane < kVecWidth; ++lane) {
					const uint tail_k = global_k + lane;
					tail[lane] = tail_k < params.inner_dim ? lhs_values[tail_k] : 0.0f;
				}
				lhs_tile[load_index] = tail;
			}
		}
		for (uint load_index = tid; load_index < kTileVecCount * kMatMulTileColumns; load_index += threads_per_group) {
			const uint local_vec = load_index / kMatMulTileColumns;
			const uint local_col = load_index % kMatMulTileColumns;
			const uint global_k = tile_start + local_vec * kVecWidth;
			const uint global_col = tgid * kMatMulTileColumns + local_col;
			float4 rhs_vec = float4(0.0f);
			if (global_col < params.column_count) {
				const uint rhs_index = global_col * params.rhs_row_stride + global_k;
				if (global_k + 3 < params.inner_dim) {
					rhs_vec = *reinterpret_cast<const device float4*>(rhs_values + rhs_index);
				} else {
					for (uint lane = 0; lane < kVecWidth; ++lane) {
						const uint tail_k = global_k + lane;
						rhs_vec[lane] = tail_k < params.inner_dim ? rhs_values[rhs_index + lane] : 0.0f;
					}
				}
			}
			rhs_tile[local_vec][local_col] = rhs_vec;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if (tid < kMatMulTileColumns && col_index < params.column_count) {
			const uint tile_extent = min(kMatMulTileDepth, params.inner_dim - tile_start);
			const uint full_vec_count = tile_extent / kVecWidth;
			for (uint local_vec = 0; local_vec < full_vec_count; ++local_vec) {
				accumulator += dot(lhs_tile[local_vec], rhs_tile[local_vec][tid]);
			}
			const uint tail_start = full_vec_count * kVecWidth;
			if (tail_start < tile_extent) {
				const float4 lhs_vec = lhs_tile[full_vec_count];
				const float4 rhs_vec = rhs_tile[full_vec_count][tid];
				for (uint lane = tail_start; lane < tile_extent; ++lane) {
					const uint tail_lane = lane - tail_start;
					accumulator += lhs_vec[tail_lane] * rhs_vec[tail_lane];
				}
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

kernel void matmul_f32_decode_lmhead_vec4(const device float* lhs_values [[buffer(0)]],
					  const device float* rhs_values [[buffer(1)]],
					  const device float* bias_values [[buffer(2)]],
					  device float* output_values [[buffer(3)]],
					  constant MatMulParams& params [[buffer(4)]],
					  const device float* residual_values [[buffer(5)]],
					  uint tid [[thread_index_in_threadgroup]],
					  uint tgid [[threadgroup_position_in_grid]]) {
	constexpr uint kOutputsPerThread = 4;
	constexpr uint kMatMulTileDepth = 32;
	constexpr uint kVecWidth = 4;
	constexpr uint kTileVecCount = kMatMulTileDepth / kVecWidth;
	if (kMatMulTileColumns == 0 || (kMatMulTileColumns % kOutputsPerThread) != 0 || !kMatMulTransposeRhs) {
		return;
	}

	threadgroup float4 lhs_tile[kTileVecCount];
	const uint col_base = tgid * kMatMulTileColumns + tid * kOutputsPerThread;
	const uint threads_per_group = kMatMulTileColumns / kOutputsPerThread;
	float4 accumulators = float4(0.0f);

	for (uint tile_start = 0; tile_start < params.inner_dim; tile_start += kMatMulTileDepth) {
		for (uint load_index = tid; load_index < kTileVecCount; load_index += threads_per_group) {
			const uint global_k = tile_start + load_index * kVecWidth;
			if (global_k + 3 < params.inner_dim) {
				lhs_tile[load_index] = *reinterpret_cast<const device float4*>(lhs_values + global_k);
			} else {
				float4 tail = float4(0.0f);
				for (uint lane = 0; lane < kVecWidth; ++lane) {
					const uint tail_k = global_k + lane;
					tail[lane] = tail_k < params.inner_dim ? lhs_values[tail_k] : 0.0f;
				}
				lhs_tile[load_index] = tail;
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);

		const uint tile_extent = min(kMatMulTileDepth, params.inner_dim - tile_start);
		const uint full_vec_count = tile_extent / kVecWidth;
		for (uint local_vec = 0; local_vec < full_vec_count; ++local_vec) {
			const uint global_k = tile_start + local_vec * kVecWidth;
			const float4 lhs_vec = lhs_tile[local_vec];
			for (uint output_lane = 0; output_lane < kOutputsPerThread; ++output_lane) {
				const uint global_col = col_base + output_lane;
				if (global_col >= params.column_count) {
					continue;
				}
				const uint rhs_index = global_col * params.rhs_row_stride + global_k;
				const float4 rhs_vec = *reinterpret_cast<const device float4*>(rhs_values + rhs_index);
				accumulators[output_lane] += dot(lhs_vec, rhs_vec);
			}
		}
		const uint tail_start = full_vec_count * kVecWidth;
		if (tail_start < tile_extent) {
			const float4 lhs_vec = lhs_tile[full_vec_count];
			for (uint output_lane = 0; output_lane < kOutputsPerThread; ++output_lane) {
				const uint global_col = col_base + output_lane;
				if (global_col >= params.column_count) {
					continue;
				}
				const uint rhs_index = global_col * params.rhs_row_stride + tile_start + tail_start;
				for (uint lane = 0; lane < tile_extent - tail_start; ++lane) {
					accumulators[output_lane] += lhs_vec[lane] * rhs_values[rhs_index + lane];
				}
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}

	for (uint output_lane = 0; output_lane < kOutputsPerThread; ++output_lane) {
		const uint global_col = col_base + output_lane;
		if (global_col >= params.column_count) {
			continue;
		}
		float accumulator = accumulators[output_lane];
		if (kMatMulUseBias) {
			accumulator += bias_values[global_col];
		}
		if (kEnableResidual) {
			accumulator += residual_values[global_col];
		}
		if (kEnableSiLU) {
			accumulator = accumulator / (1.0f + exp(-accumulator));
		}
		output_values[global_col] = accumulator;
	}
}

kernel void matmul_f32_f16rhs_basic(const device float* lhs_values [[buffer(0)]],
						 const device half* rhs_values [[buffer(1)]],
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
		accumulator += lhs_values[lhs_base + inner_index] * static_cast<float>(rhs_values[rhs_index]);
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

kernel void matmul_f32_f16rhs_tiled(const device float* lhs_values [[buffer(0)]],
					 const device half* rhs_values [[buffer(1)]],
					 const device float* bias_values [[buffer(2)]],
					 device float* output_values [[buffer(3)]],
					 constant MatMulParams& params [[buffer(4)]],
					 const device float* residual_values [[buffer(5)]],
					 uint2 tid [[thread_position_in_threadgroup]],
					 uint2 tgid [[threadgroup_position_in_grid]]) {
	if (kMatMulTileColumns == 0 || kMatMulTileRows == 0 || kMatMulTileColumns > 32 || kMatMulTileRows > 4) {
		return;
	}

	constexpr uint kMatMulTileDepth = 32;
	threadgroup float lhs_tile[4][kMatMulTileDepth];
	threadgroup float rhs_tile[kMatMulTileDepth][32];

	const uint row_index = tgid.y * kMatMulTileRows + tid.y;
	const uint col_index = tgid.x * kMatMulTileColumns + tid.x;
	const uint linear_tid = tid.y * kMatMulTileColumns + tid.x;
	const uint threads_per_group = kMatMulTileColumns * kMatMulTileRows;
	float accumulator = 0.0f;

	for (uint tile_start = 0; tile_start < params.inner_dim; tile_start += kMatMulTileDepth) {
		const uint lhs_load_count = kMatMulTileRows * kMatMulTileDepth;
		for (uint load_index = linear_tid; load_index < lhs_load_count; load_index += threads_per_group) {
			const uint local_row = load_index / kMatMulTileDepth;
			const uint local_k = load_index % kMatMulTileDepth;
			const uint global_row = tgid.y * kMatMulTileRows + local_row;
			const uint global_k = tile_start + local_k;
			lhs_tile[local_row][local_k] = (global_row < params.row_count && global_k < params.inner_dim)
				? lhs_values[global_row * params.lhs_row_stride + global_k]
				: 0.0f;
		}

		const uint rhs_load_count = kMatMulTileDepth * kMatMulTileColumns;
		for (uint load_index = linear_tid; load_index < rhs_load_count; load_index += threads_per_group) {
			const uint local_k = load_index / kMatMulTileColumns;
			const uint local_col = load_index % kMatMulTileColumns;
			const uint global_k = tile_start + local_k;
			const uint global_col = tgid.x * kMatMulTileColumns + local_col;
			const uint rhs_index = kMatMulTransposeRhs
				? global_col * params.rhs_row_stride + global_k
				: global_k * params.rhs_row_stride + global_col;
			rhs_tile[local_k][local_col] = (global_k < params.inner_dim && global_col < params.column_count)
				? static_cast<float>(rhs_values[rhs_index])
				: 0.0f;
		}

		threadgroup_barrier(mem_flags::mem_threadgroup);
		if (row_index < params.row_count && col_index < params.column_count) {
			const uint tile_extent = min(kMatMulTileDepth, params.inner_dim - tile_start);
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

kernel void matmul_f32_f16rhs_decode_tiled(const device float* lhs_values [[buffer(0)]],
					    const device half* rhs_values [[buffer(1)]],
					    const device float* bias_values [[buffer(2)]],
					    device float* output_values [[buffer(3)]],
					    constant MatMulParams& params [[buffer(4)]],
					    const device float* residual_values [[buffer(5)]],
					    uint tid [[thread_index_in_threadgroup]],
					    uint tgid [[threadgroup_position_in_grid]]) {
	if (kMatMulTileColumns == 0 || kMatMulTileColumns > 32) {
		return;
	}

	constexpr uint kMatMulTileDepth = 32;
	threadgroup float lhs_tile[kMatMulTileDepth];
	threadgroup float rhs_tile[kMatMulTileDepth][32];
	const uint col_index = tgid * kMatMulTileColumns + tid;
	const uint threads_per_group = kMatMulTileColumns;
	float accumulator = 0.0f;

	for (uint tile_start = 0; tile_start < params.inner_dim; tile_start += kMatMulTileDepth) {
		for (uint load_index = tid; load_index < kMatMulTileDepth; load_index += threads_per_group) {
			const uint global_k = tile_start + load_index;
			lhs_tile[load_index] = global_k < params.inner_dim ? lhs_values[global_k] : 0.0f;
		}
		for (uint load_index = tid; load_index < kMatMulTileDepth * kMatMulTileColumns; load_index += threads_per_group) {
			const uint local_k = load_index / kMatMulTileColumns;
			const uint local_col = load_index % kMatMulTileColumns;
			const uint global_k = tile_start + local_k;
			const uint global_col = tgid * kMatMulTileColumns + local_col;
			const uint rhs_index = kMatMulTransposeRhs
				? global_col * params.rhs_row_stride + global_k
				: global_k * params.rhs_row_stride + global_col;
			rhs_tile[local_k][local_col] = (global_col < params.column_count && global_k < params.inner_dim)
				? static_cast<float>(rhs_values[rhs_index])
				: 0.0f;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if (tid < kMatMulTileColumns && col_index < params.column_count) {
			const uint tile_extent = min(kMatMulTileDepth, params.inner_dim - tile_start);
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

kernel void matmul_f32_f16rhs_decode_tiled_vec4(const device float* lhs_values [[buffer(0)]],
						const device half* rhs_values [[buffer(1)]],
						const device float* bias_values [[buffer(2)]],
						device float* output_values [[buffer(3)]],
						constant MatMulParams& params [[buffer(4)]],
						const device float* residual_values [[buffer(5)]],
						uint tid [[thread_index_in_threadgroup]],
						uint tgid [[threadgroup_position_in_grid]]) {
	if (kMatMulTileColumns == 0 || kMatMulTileColumns > 32 || !kMatMulTransposeRhs) {
		return;
	}

	constexpr uint kMatMulTileDepth = 32;
	constexpr uint kVecWidth = 4;
	constexpr uint kTileVecCount = kMatMulTileDepth / kVecWidth;
	threadgroup float4 lhs_tile[kTileVecCount];
	threadgroup float4 rhs_tile[kTileVecCount][32];
	const uint col_index = tgid * kMatMulTileColumns + tid;
	const uint threads_per_group = kMatMulTileColumns;
	float accumulator = 0.0f;

	for (uint tile_start = 0; tile_start < params.inner_dim; tile_start += kMatMulTileDepth) {
		for (uint load_index = tid; load_index < kTileVecCount; load_index += threads_per_group) {
			const uint global_k = tile_start + load_index * kVecWidth;
			if (global_k + 3 < params.inner_dim) {
				lhs_tile[load_index] = *reinterpret_cast<const device float4*>(lhs_values + global_k);
			} else {
				float4 tail = float4(0.0f);
				for (uint lane = 0; lane < kVecWidth; ++lane) {
					const uint tail_k = global_k + lane;
					tail[lane] = tail_k < params.inner_dim ? lhs_values[tail_k] : 0.0f;
				}
				lhs_tile[load_index] = tail;
			}
		}
		for (uint load_index = tid; load_index < kTileVecCount * kMatMulTileColumns; load_index += threads_per_group) {
			const uint local_vec = load_index / kMatMulTileColumns;
			const uint local_col = load_index % kMatMulTileColumns;
			const uint global_k = tile_start + local_vec * kVecWidth;
			const uint global_col = tgid * kMatMulTileColumns + local_col;
			float4 rhs_vec = float4(0.0f);
			if (global_col < params.column_count) {
				const uint rhs_index = global_col * params.rhs_row_stride + global_k;
				if (global_k + 3 < params.inner_dim) {
					const half4 packed = *reinterpret_cast<const device half4*>(rhs_values + rhs_index);
					rhs_vec = float4(packed);
				} else {
					for (uint lane = 0; lane < kVecWidth; ++lane) {
						const uint tail_k = global_k + lane;
						rhs_vec[lane] = tail_k < params.inner_dim ? static_cast<float>(rhs_values[rhs_index + lane]) : 0.0f;
					}
				}
			}
			rhs_tile[local_vec][local_col] = rhs_vec;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if (tid < kMatMulTileColumns && col_index < params.column_count) {
			const uint tile_extent = min(kMatMulTileDepth, params.inner_dim - tile_start);
			const uint full_vec_count = tile_extent / kVecWidth;
			for (uint local_vec = 0; local_vec < full_vec_count; ++local_vec) {
				accumulator += dot(lhs_tile[local_vec], rhs_tile[local_vec][tid]);
			}
			const uint tail_start = full_vec_count * kVecWidth;
			if (tail_start < tile_extent) {
				const float4 lhs_vec = lhs_tile[full_vec_count];
				const float4 rhs_vec = rhs_tile[full_vec_count][tid];
				for (uint lane = tail_start; lane < tile_extent; ++lane) {
					const uint tail_lane = lane - tail_start;
					accumulator += lhs_vec[tail_lane] * rhs_vec[tail_lane];
				}
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

kernel void matmul_f32_f16rhs_decode_lmhead_vec4(const device float* lhs_values [[buffer(0)]],
						 const device half* rhs_values [[buffer(1)]],
						 const device float* bias_values [[buffer(2)]],
						 device float* output_values [[buffer(3)]],
						 constant MatMulParams& params [[buffer(4)]],
						 const device float* residual_values [[buffer(5)]],
						 uint tid [[thread_index_in_threadgroup]],
						 uint tgid [[threadgroup_position_in_grid]]) {
	constexpr uint kOutputsPerThread = 4;
	constexpr uint kMatMulTileDepth = 32;
	constexpr uint kVecWidth = 4;
	constexpr uint kTileVecCount = kMatMulTileDepth / kVecWidth;
	if (kMatMulTileColumns == 0 || (kMatMulTileColumns % kOutputsPerThread) != 0 || !kMatMulTransposeRhs) {
		return;
	}

	threadgroup float4 lhs_tile[kTileVecCount];
	const uint col_base = tgid * kMatMulTileColumns + tid * kOutputsPerThread;
	const uint threads_per_group = kMatMulTileColumns / kOutputsPerThread;
	float4 accumulators = float4(0.0f);

	for (uint tile_start = 0; tile_start < params.inner_dim; tile_start += kMatMulTileDepth) {
		for (uint load_index = tid; load_index < kTileVecCount; load_index += threads_per_group) {
			const uint global_k = tile_start + load_index * kVecWidth;
			if (global_k + 3 < params.inner_dim) {
				lhs_tile[load_index] = *reinterpret_cast<const device float4*>(lhs_values + global_k);
			} else {
				float4 tail = float4(0.0f);
				for (uint lane = 0; lane < kVecWidth; ++lane) {
					const uint tail_k = global_k + lane;
					tail[lane] = tail_k < params.inner_dim ? lhs_values[tail_k] : 0.0f;
				}
				lhs_tile[load_index] = tail;
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);

		const uint tile_extent = min(kMatMulTileDepth, params.inner_dim - tile_start);
		const uint full_vec_count = tile_extent / kVecWidth;
		for (uint local_vec = 0; local_vec < full_vec_count; ++local_vec) {
			const uint global_k = tile_start + local_vec * kVecWidth;
			const float4 lhs_vec = lhs_tile[local_vec];
			for (uint output_lane = 0; output_lane < kOutputsPerThread; ++output_lane) {
				const uint global_col = col_base + output_lane;
				if (global_col >= params.column_count) {
					continue;
				}
				const uint rhs_index = global_col * params.rhs_row_stride + global_k;
				const half4 packed = *reinterpret_cast<const device half4*>(rhs_values + rhs_index);
				accumulators[output_lane] += dot(lhs_vec, float4(packed));
			}
		}
		const uint tail_start = full_vec_count * kVecWidth;
		if (tail_start < tile_extent) {
			const float4 lhs_vec = lhs_tile[full_vec_count];
			for (uint output_lane = 0; output_lane < kOutputsPerThread; ++output_lane) {
				const uint global_col = col_base + output_lane;
				if (global_col >= params.column_count) {
					continue;
				}
				const uint rhs_index = global_col * params.rhs_row_stride + tile_start + tail_start;
				for (uint lane = 0; lane < tile_extent - tail_start; ++lane) {
					accumulators[output_lane] += lhs_vec[lane] * static_cast<float>(rhs_values[rhs_index + lane]);
				}
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}

	for (uint output_lane = 0; output_lane < kOutputsPerThread; ++output_lane) {
		const uint global_col = col_base + output_lane;
		if (global_col >= params.column_count) {
			continue;
		}
		float accumulator = accumulators[output_lane];
		if (kMatMulUseBias) {
			accumulator += bias_values[global_col];
		}
		if (kEnableResidual) {
			accumulator += residual_values[global_col];
		}
		if (kEnableSiLU) {
			accumulator = accumulator / (1.0f + exp(-accumulator));
		}
		output_values[global_col] = accumulator;
	}
}

kernel void dual_matmul_f32_decode_vec4(const device float* lhs_values [[buffer(0)]],
					const device float* rhs0_values [[buffer(1)]],
					const device float* rhs1_values [[buffer(2)]],
					device float* output0_values [[buffer(3)]],
					device float* output1_values [[buffer(4)]],
					constant MatMulParams& params [[buffer(5)]],
					uint tid [[thread_index_in_threadgroup]],
					uint tgid [[threadgroup_position_in_grid]]) {
	constexpr uint kMatMulTileDepth = 32;
	constexpr uint kVecWidth = 4;
	constexpr uint kTileVecCount = kMatMulTileDepth / kVecWidth;
	if (params.column_count == 0) {
		return;
	}

	threadgroup float4 lhs_tile[kTileVecCount];
	const uint col_index = tgid * 32 + tid;
	float accumulator0 = 0.0f;
	float accumulator1 = 0.0f;

	for (uint tile_start = 0; tile_start < params.inner_dim; tile_start += kMatMulTileDepth) {
		for (uint load_index = tid; load_index < kTileVecCount; load_index += 32) {
			const uint global_k = tile_start + load_index * kVecWidth;
			if (global_k + 3 < params.inner_dim) {
				lhs_tile[load_index] = *reinterpret_cast<const device float4*>(lhs_values + global_k);
			} else {
				float4 tail = float4(0.0f);
				for (uint lane = 0; lane < kVecWidth; ++lane) {
					const uint tail_k = global_k + lane;
					tail[lane] = tail_k < params.inner_dim ? lhs_values[tail_k] : 0.0f;
				}
				lhs_tile[load_index] = tail;
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if (col_index < params.column_count) {
			const uint tile_extent = min(kMatMulTileDepth, params.inner_dim - tile_start);
			const uint full_vec_count = tile_extent / kVecWidth;
			for (uint local_vec = 0; local_vec < full_vec_count; ++local_vec) {
				const uint global_k = tile_start + local_vec * kVecWidth;
				const float4 lhs_vec = lhs_tile[local_vec];
				const uint rhs_index = col_index * params.rhs_row_stride + global_k;
				accumulator0 += dot(lhs_vec, *reinterpret_cast<const device float4*>(rhs0_values + rhs_index));
				accumulator1 += dot(lhs_vec, *reinterpret_cast<const device float4*>(rhs1_values + rhs_index));
			}
			const uint tail_start = full_vec_count * kVecWidth;
			if (tail_start < tile_extent) {
				const float4 lhs_vec = lhs_tile[full_vec_count];
				const uint rhs_index = col_index * params.rhs_row_stride + tile_start + tail_start;
				for (uint lane = 0; lane < tile_extent - tail_start; ++lane) {
					accumulator0 += lhs_vec[lane] * rhs0_values[rhs_index + lane];
					accumulator1 += lhs_vec[lane] * rhs1_values[rhs_index + lane];
				}
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}

	if (col_index >= params.column_count) {
		return;
	}
	output0_values[col_index] = accumulator0 / (1.0f + exp(-accumulator0));
	output1_values[col_index] = accumulator1;
}

kernel void dual_matmul_f32_f16rhs_decode_vec4(const device float* lhs_values [[buffer(0)]],
					       const device half* rhs0_values [[buffer(1)]],
					       const device half* rhs1_values [[buffer(2)]],
					       device float* output0_values [[buffer(3)]],
					       device float* output1_values [[buffer(4)]],
					       constant MatMulParams& params [[buffer(5)]],
					       uint tid [[thread_index_in_threadgroup]],
					       uint tgid [[threadgroup_position_in_grid]]) {
	constexpr uint kMatMulTileDepth = 32;
	constexpr uint kVecWidth = 4;
	constexpr uint kTileVecCount = kMatMulTileDepth / kVecWidth;
	if (params.column_count == 0) {
		return;
	}

	threadgroup float4 lhs_tile[kTileVecCount];
	const uint col_index = tgid * 32 + tid;
	float accumulator0 = 0.0f;
	float accumulator1 = 0.0f;

	for (uint tile_start = 0; tile_start < params.inner_dim; tile_start += kMatMulTileDepth) {
		for (uint load_index = tid; load_index < kTileVecCount; load_index += 32) {
			const uint global_k = tile_start + load_index * kVecWidth;
			if (global_k + 3 < params.inner_dim) {
				lhs_tile[load_index] = *reinterpret_cast<const device float4*>(lhs_values + global_k);
			} else {
				float4 tail = float4(0.0f);
				for (uint lane = 0; lane < kVecWidth; ++lane) {
					const uint tail_k = global_k + lane;
					tail[lane] = tail_k < params.inner_dim ? lhs_values[tail_k] : 0.0f;
				}
				lhs_tile[load_index] = tail;
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if (col_index < params.column_count) {
			const uint tile_extent = min(kMatMulTileDepth, params.inner_dim - tile_start);
			const uint full_vec_count = tile_extent / kVecWidth;
			for (uint local_vec = 0; local_vec < full_vec_count; ++local_vec) {
				const uint global_k = tile_start + local_vec * kVecWidth;
				const float4 lhs_vec = lhs_tile[local_vec];
				const uint rhs_index = col_index * params.rhs_row_stride + global_k;
				accumulator0 += dot(lhs_vec, float4(*reinterpret_cast<const device half4*>(rhs0_values + rhs_index)));
				accumulator1 += dot(lhs_vec, float4(*reinterpret_cast<const device half4*>(rhs1_values + rhs_index)));
			}
			const uint tail_start = full_vec_count * kVecWidth;
			if (tail_start < tile_extent) {
				const float4 lhs_vec = lhs_tile[full_vec_count];
				const uint rhs_index = col_index * params.rhs_row_stride + tile_start + tail_start;
				for (uint lane = 0; lane < tile_extent - tail_start; ++lane) {
					accumulator0 += lhs_vec[lane] * static_cast<float>(rhs0_values[rhs_index + lane]);
					accumulator1 += lhs_vec[lane] * static_cast<float>(rhs1_values[rhs_index + lane]);
				}
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}

	if (col_index >= params.column_count) {
		return;
	}
	output0_values[col_index] = accumulator0 / (1.0f + exp(-accumulator0));
	output1_values[col_index] = accumulator1;
}
