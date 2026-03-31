// GPU Optimization Benchmark: Baseline + Phase Measurements
// This test measures per-operation command buffer overhead and overall decode throughput.
// Run this BEFORE and AFTER each optimization phase.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "metal/metal_context.h"
#include "buffer/metal_buffer.h"
#include "buffer/buffer_arena.h"
#include "kernel/pipeline_cache.h"
#include "op/matmul_op.h"
#include "op/rms_norm_op.h"
#include "op/softmax_op.h"
#include "op/rope_op.h"
#include "op/elementwise_mul_op.h"
#include "op/embedding_op.h"
#include "tensor/device_tensor.h"
#include "tensor/tensor_desc.h"

using Clock = std::chrono::steady_clock;

static double DurationUs(const Clock::time_point& start, const Clock::time_point& end) {
    return std::chrono::duration<double, std::micro>(end - start).count();
}

struct BenchmarkResult {
    std::string name;
    double avg_us;
    double min_us;
    double max_us;
    int iterations;
};

static void PrintResult(const BenchmarkResult& r) {
    std::cout << std::left << std::setw(40) << r.name
              << " avg=" << std::right << std::setw(10) << std::fixed << std::setprecision(1) << r.avg_us << " µs"
              << "  min=" << std::setw(10) << r.min_us
              << "  max=" << std::setw(10) << r.max_us
              << "  (n=" << r.iterations << ")\n";
}

static soc::gpu::DeviceTensor MakeFloatTensor(const soc::gpu::MetalContext& ctx,
                                               const std::vector<std::size_t>& shape,
                                               const std::string& label) {
    std::size_t count = 1;
    for (auto d : shape) count *= d;
    std::string err;
    auto buf = soc::gpu::MetalBuffer::CreateShared(ctx, count * sizeof(float), label, &err);
    if (!buf) {
        std::cerr << "Failed to create buffer: " << err << "\n";
        return {};
    }
    // Initialize with small random-ish values
    std::vector<float> data(count);
    for (std::size_t i = 0; i < count; ++i) {
        data[i] = 0.01f * static_cast<float>(i % 100 - 50);
    }
    buf->Write(data.data(), count * sizeof(float), 0, &err);
    return soc::gpu::DeviceTensor(buf, 0,
                                   soc::gpu::TensorDesc::CreateContiguous(soc::gpu::DataType::kFloat32, shape));
}

// ============================================================================
// Benchmark 1: Raw command buffer overhead (empty dispatch)
// ============================================================================
static BenchmarkResult BenchCommandBufferOverhead(const soc::gpu::MetalContext& ctx, int warmup, int iterations) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)ctx.GetNativeCommandQueue();
        id<MTLDevice> device = (__bridge id<MTLDevice>)ctx.GetNativeDevice();

        // Create trivial pipeline
        NSError* nsErr = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:@"kernel void noop(uint gid [[thread_position_in_grid]]) {}"
                                                  options:nil
                                                    error:&nsErr];
        id<MTLFunction> func = [lib newFunctionWithName:@"noop"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&nsErr];

        // Warmup
        for (int i = 0; i < warmup; ++i) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }

        // Measure
        std::vector<double> times;
        for (int i = 0; i < iterations; ++i) {
            auto start = Clock::now();
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            times.push_back(DurationUs(start, Clock::now()));
        }

        double sum = std::accumulate(times.begin(), times.end(), 0.0);
        double min_v = *std::min_element(times.begin(), times.end());
        double max_v = *std::max_element(times.begin(), times.end());
        return {"command_buffer_overhead (noop)", sum / times.size(), min_v, max_v, iterations};
    }
}

// ============================================================================
// Benchmark 2: MatMul decode (1×1024) × (1024×1024) — simulates one linear layer
// ============================================================================
static BenchmarkResult BenchMatMulDecode(const soc::gpu::MetalContext& ctx,
                                          soc::gpu::PipelineCache& cache,
                                          int warmup, int iterations) {
    const std::size_t M = 1, K = 1024, N = 1024;
    auto lhs = MakeFloatTensor(ctx, {M, K}, "bench_lhs");
    auto rhs = MakeFloatTensor(ctx, {N, K}, "bench_rhs");  // transposed: (N, K)
    auto out = MakeFloatTensor(ctx, {M, N}, "bench_out");
    std::string err;

    soc::gpu::MatMulParams params;
    params.decode_mode = true;
    params.transpose_rhs = true;
    params.row_count = static_cast<uint32_t>(M);
    params.inner_dim = static_cast<uint32_t>(K);
    params.column_count = static_cast<uint32_t>(N);

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        soc::gpu::MatMulOp::Run(ctx, &cache, lhs, rhs, nullptr, nullptr, out, params, nullptr, nullptr, &err);
    }

    std::vector<double> times;
    for (int i = 0; i < iterations; ++i) {
        auto start = Clock::now();
        soc::gpu::MatMulOp::Run(ctx, &cache, lhs, rhs, nullptr, nullptr, out, params, nullptr, nullptr, &err);
        times.push_back(DurationUs(start, Clock::now()));
    }

    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double min_v = *std::min_element(times.begin(), times.end());
    double max_v = *std::max_element(times.begin(), times.end());
    return {"matmul_decode_1x1024x1024", sum / times.size(), min_v, max_v, iterations};
}

// ============================================================================
// Benchmark 3: MatMul decode (1×1024) × (1024×2816) — simulates gate/up proj
// ============================================================================
static BenchmarkResult BenchMatMulDecodeWide(const soc::gpu::MetalContext& ctx,
                                              soc::gpu::PipelineCache& cache,
                                              int warmup, int iterations) {
    const std::size_t M = 1, K = 1024, N = 2816;
    auto lhs = MakeFloatTensor(ctx, {M, K}, "bench_lhs_wide");
    auto rhs = MakeFloatTensor(ctx, {N, K}, "bench_rhs_wide");
    auto out = MakeFloatTensor(ctx, {M, N}, "bench_out_wide");
    std::string err;

    soc::gpu::MatMulParams params;
    params.decode_mode = true;
    params.transpose_rhs = true;
    params.row_count = static_cast<uint32_t>(M);
    params.inner_dim = static_cast<uint32_t>(K);
    params.column_count = static_cast<uint32_t>(N);

    for (int i = 0; i < warmup; ++i) {
        soc::gpu::MatMulOp::Run(ctx, &cache, lhs, rhs, nullptr, nullptr, out, params, nullptr, nullptr, &err);
    }

    std::vector<double> times;
    for (int i = 0; i < iterations; ++i) {
        auto start = Clock::now();
        soc::gpu::MatMulOp::Run(ctx, &cache, lhs, rhs, nullptr, nullptr, out, params, nullptr, nullptr, &err);
        times.push_back(DurationUs(start, Clock::now()));
    }

    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double min_v = *std::min_element(times.begin(), times.end());
    double max_v = *std::max_element(times.begin(), times.end());
    return {"matmul_decode_1x1024x2816", sum / times.size(), min_v, max_v, iterations};
}

// ============================================================================
// Benchmark 4: RMSNorm (1×1024) — single row, simulates decode norm
// ============================================================================
static BenchmarkResult BenchRmsNormDecode(const soc::gpu::MetalContext& ctx,
                                           soc::gpu::PipelineCache& cache,
                                           int warmup, int iterations) {
    auto input = MakeFloatTensor(ctx, {1, 1024}, "bench_norm_in");
    auto weight = MakeFloatTensor(ctx, {1024}, "bench_norm_w");
    auto output = MakeFloatTensor(ctx, {1, 1024}, "bench_norm_out");
    std::string err;

    soc::gpu::RmsNormParams params;
    params.row_count = 1;
    params.row_size = 1024;
    params.epsilon = 1e-6f;

    for (int i = 0; i < warmup; ++i) {
        soc::gpu::RmsNormOp::Run(ctx, &cache, input, weight, output, params, nullptr, nullptr, &err);
    }

    std::vector<double> times;
    for (int i = 0; i < iterations; ++i) {
        auto start = Clock::now();
        soc::gpu::RmsNormOp::Run(ctx, &cache, input, weight, output, params, nullptr, nullptr, &err);
        times.push_back(DurationUs(start, Clock::now()));
    }

    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double min_v = *std::min_element(times.begin(), times.end());
    double max_v = *std::max_element(times.begin(), times.end());
    return {"rms_norm_decode_1x1024", sum / times.size(), min_v, max_v, iterations};
}

// ============================================================================
// Benchmark 5: Batch of N command buffers (simulate full decode step overhead)
// ============================================================================
static BenchmarkResult BenchBatchedNoops(const soc::gpu::MetalContext& ctx, int batch_count, int warmup, int iterations) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)ctx.GetNativeCommandQueue();
        id<MTLDevice> device = (__bridge id<MTLDevice>)ctx.GetNativeDevice();

        NSError* nsErr = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:@"kernel void noop2(uint gid [[thread_position_in_grid]]) {}"
                                                  options:nil
                                                    error:&nsErr];
        id<MTLFunction> func = [lib newFunctionWithName:@"noop2"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&nsErr];

        // Warmup
        for (int w = 0; w < warmup; ++w) {
            for (int b = 0; b < batch_count; ++b) {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pipeline];
                [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }
        }

        // Measure
        std::vector<double> times;
        for (int i = 0; i < iterations; ++i) {
            auto start = Clock::now();
            for (int b = 0; b < batch_count; ++b) {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pipeline];
                [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }
            times.push_back(DurationUs(start, Clock::now()));
        }

        double sum = std::accumulate(times.begin(), times.end(), 0.0);
        double min_v = *std::min_element(times.begin(), times.end());
        double max_v = *std::max_element(times.begin(), times.end());
        std::string name = "batched_noop_x" + std::to_string(batch_count) + "_serial";
        return {name, sum / times.size(), min_v, max_v, iterations};
    }
}

// ============================================================================
// Benchmark 6: Single command buffer with N encoders (compare to batch serial)
// ============================================================================
static BenchmarkResult BenchSingleCBMultiEncoder(const soc::gpu::MetalContext& ctx, int dispatch_count, int warmup, int iterations) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)ctx.GetNativeCommandQueue();
        id<MTLDevice> device = (__bridge id<MTLDevice>)ctx.GetNativeDevice();

        NSError* nsErr = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:@"kernel void noop3(uint gid [[thread_position_in_grid]]) {}"
                                                  options:nil
                                                    error:&nsErr];
        id<MTLFunction> func = [lib newFunctionWithName:@"noop3"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&nsErr];

        // Warmup
        for (int w = 0; w < warmup; ++w) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            for (int d = 0; d < dispatch_count; ++d) {
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pipeline];
                [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                [enc endEncoding];
            }
            [cb commit];
            [cb waitUntilCompleted];
        }

        // Measure
        std::vector<double> times;
        for (int i = 0; i < iterations; ++i) {
            auto start = Clock::now();
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            for (int d = 0; d < dispatch_count; ++d) {
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pipeline];
                [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                [enc endEncoding];
            }
            [cb commit];
            [cb waitUntilCompleted];
            times.push_back(DurationUs(start, Clock::now()));
        }

        double sum = std::accumulate(times.begin(), times.end(), 0.0);
        double min_v = *std::min_element(times.begin(), times.end());
        double max_v = *std::max_element(times.begin(), times.end());
        std::string name = "single_cb_x" + std::to_string(dispatch_count) + "_batched";
        return {name, sum / times.size(), min_v, max_v, iterations};
    }
}

int main() {
    std::string err;
    auto ctx = soc::gpu::MetalContext::CreateDefault("build/shaders/gpu.metallib",
                                                     "shaders/gpu_kernels.metal",
                                                     &err);
    if (!ctx) {
        std::cerr << "Metal init failed: " << err << "\n";
        return 1;
    }

    const auto& info = ctx->GetDeviceInfo();
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  GPU OPTIMIZATION BASELINE BENCHMARK                        ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    std::cout << "Device: " << info.name << "\n";
    std::cout << "Unified Memory: " << (info.has_unified_memory ? "yes" : "no") << "\n";
    std::cout << "SIMD Matrix Support: " << (info.supports_simdgroup_matrix ? "yes" : "no") << "\n";
    std::cout << "Thread Execution Width: " << info.thread_execution_width << "\n";
    std::cout << "Max Threads/Threadgroup: " << info.max_threads_per_threadgroup << "\n";
    std::cout << "Recommended Working Set: " << (info.recommended_max_working_set_size / (1024*1024)) << " MB\n";
    std::cout << "\n";

    soc::gpu::PipelineCache cache(*ctx);
    constexpr int kWarmup = 10;
    constexpr int kIterations = 100;

    std::vector<BenchmarkResult> results;

    // Section 1: Command Buffer Overhead
    std::cout << "--- SECTION 1: Command Buffer Overhead ---\n";
    results.push_back(BenchCommandBufferOverhead(*ctx, kWarmup, kIterations));
    PrintResult(results.back());

    results.push_back(BenchBatchedNoops(*ctx, 535, kWarmup, 20));
    PrintResult(results.back());

    results.push_back(BenchSingleCBMultiEncoder(*ctx, 535, kWarmup, 20));
    PrintResult(results.back());

    std::cout << "\n--- SECTION 2: Per-Op Kernels (decode mode) ---\n";
    results.push_back(BenchMatMulDecode(*ctx, cache, kWarmup, kIterations));
    PrintResult(results.back());

    results.push_back(BenchMatMulDecodeWide(*ctx, cache, kWarmup, kIterations));
    PrintResult(results.back());

    results.push_back(BenchRmsNormDecode(*ctx, cache, kWarmup, kIterations));
    PrintResult(results.back());

    // Section 3: Summary
    std::cout << "\n--- SECTION 3: Throughput Estimates ---\n";
    double cb_overhead_us = results[0].avg_us;
    double serial_535_us = results[1].avg_us;
    double batched_535_us = results[2].avg_us;

    std::cout << "Single CB overhead: " << std::fixed << std::setprecision(1) << cb_overhead_us << " µs\n";
    std::cout << "535 serial CBs:     " << serial_535_us << " µs = " << serial_535_us / 1000.0 << " ms\n";
    std::cout << "535 batched in 1CB: " << batched_535_us << " µs = " << batched_535_us / 1000.0 << " ms\n";
    std::cout << "Batching speedup:   " << serial_535_us / batched_535_us << "x\n";
    std::cout << "\n";

    double serial_overhead_per_token_ms = serial_535_us / 1000.0;
    double batched_overhead_per_token_ms = batched_535_us / 1000.0;
    std::cout << "Estimated decode overhead (serial):  " << serial_overhead_per_token_ms << " ms/token"
              << " → max " << std::setprecision(0) << 1000.0 / serial_overhead_per_token_ms << " tok/s\n";
    std::cout << "Estimated decode overhead (batched): " << std::setprecision(1) << batched_overhead_per_token_ms << " ms/token"
              << " → max " << std::setprecision(0) << 1000.0 / batched_overhead_per_token_ms << " tok/s\n";

    std::cout << "\n";
    std::cout << "MatMul 1×1024×1024 avg: " << std::setprecision(1) << results[3].avg_us << " µs\n";
    std::cout << "MatMul 1×1024×2816 avg: " << results[4].avg_us << " µs\n";
    double matmul_total_per_layer = results[3].avg_us * 4 + results[4].avg_us * 3;  // q,k,v,o + gate,up,down
    double total_compute_28_layers = matmul_total_per_layer * 28.0;
    std::cout << "Estimated matmul time per token (28 layers): " << total_compute_28_layers / 1000.0 << " ms\n";
    double total_per_token = (serial_overhead_per_token_ms * 1000 + total_compute_28_layers) / 1000.0;
    std::cout << "Total estimated (serial): " << total_per_token << " ms/token"
              << " → " << std::setprecision(0) << 1000.0 / total_per_token << " tok/s\n";

    std::cout << "\nBenchmark complete!\n";
    return 0;
}
