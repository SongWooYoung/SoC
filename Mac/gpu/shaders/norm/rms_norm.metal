#include <metal_stdlib>

using namespace metal;

struct RmsNormParams {
    uint row_size;
    uint row_count;
    float epsilon;
    uint padding;
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