#include <metal_stdlib>

using namespace metal;

kernel void bootstrap_copy(const device uint* input_values [[buffer(0)]],
                           device uint* output_values [[buffer(1)]],
                           uint gid [[thread_position_in_grid]]) {
    output_values[gid] = input_values[gid] + 1;
}