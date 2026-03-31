// Phase 1 Experiment: Command Buffer Batching via CommandStream
// Tests that multiple Metal dispatches can be batched into a single command buffer
// and produce correct results while being dramatically faster.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <string>
#include <vector>

#include "metal/metal_context.h"
#include "metal/command_stream.h"
#include "buffer/metal_buffer.h"
#include "buffer/buffer_arena.h"
#include "kernel/pipeline_cache.h"
#include "op/matmul_op.h"
#include "tensor/device_tensor.h"
#include "tensor/tensor_desc.h"

using Clock = std::chrono::steady_clock;

static double DurationUs(const Clock::time_point& s, const Clock::time_point& e) {
    return std::chrono::duration<double, std::micro>(e - s).count();
}

// ============================================================================
// Test 1: CommandStream basic lifecycle
// ============================================================================
static bool TestCommandStreamLifecycle(const soc::gpu::MetalContext& ctx) {
    std::cout << "Test: CommandStream lifecycle... ";
    std::string err;
    soc::gpu::CommandStream stream;

    if (stream.IsActive()) {
        std::cout << "FAIL (should not be active before Begin)\n";
        return false;
    }

    if (!stream.Begin(ctx, &err)) {
        std::cout << "FAIL (Begin failed: " << err << ")\n";
        return false;
    }

    if (!stream.IsActive()) {
        std::cout << "FAIL (should be active after Begin)\n";
        return false;
    }

    // Begin a second time should fail
    if (stream.Begin(ctx, &err)) {
        std::cout << "FAIL (double Begin should fail)\n";
        return false;
    }

    if (!stream.Flush(ctx, &err)) {
        std::cout << "FAIL (Flush failed: " << err << ")\n";
        return false;
    }

    if (stream.IsActive()) {
        std::cout << "FAIL (should not be active after Flush)\n";
        return false;
    }

    std::cout << "PASS\n";
    return true;
}

// ============================================================================
// Test 2: Multiple encoders in single command buffer produce correct results
// ============================================================================
static bool TestBatchedDispatchCorrectness(const soc::gpu::MetalContext& ctx) {
    std::cout << "Test: Batched dispatch correctness... ";
    @autoreleasepool {
        std::string err;
        id<MTLDevice> device = (__bridge id<MTLDevice>)ctx.GetNativeDevice();

        // Create a simple kernel: output[gid] = input[gid] * 2.0
        NSError* nsErr = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:
            @"kernel void double_it(const device float* input [[buffer(0)]],"
             "device float* output [[buffer(1)]],"
             "uint gid [[thread_position_in_grid]]) {"
             "  output[gid] = input[gid] * 2.0f;"
             "}"
            options:nil error:&nsErr];
        if (!lib) {
            std::cout << "FAIL (shader compile)\n";
            return false;
        }
        id<MTLFunction> func = [lib newFunctionWithName:@"double_it"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&nsErr];

        const std::size_t N = 1024;
        auto buf_in = soc::gpu::MetalBuffer::CreateShared(ctx, N * sizeof(float), "in", &err);
        auto buf_mid = soc::gpu::MetalBuffer::CreateShared(ctx, N * sizeof(float), "mid", &err);
        auto buf_out = soc::gpu::MetalBuffer::CreateShared(ctx, N * sizeof(float), "out", &err);

        // Write input: [1.0, 2.0, 3.0, ...]
        std::vector<float> input(N);
        for (std::size_t i = 0; i < N; ++i) input[i] = static_cast<float>(i + 1);
        buf_in->Write(input.data(), N * sizeof(float), 0, &err);

        // Batch: dispatch 1 (input → mid, *2), dispatch 2 (mid → out, *2)
        // Expected output: input * 4
        soc::gpu::CommandStream stream;
        if (!stream.Begin(ctx, &err)) {
            std::cout << "FAIL (Begin: " << err << ")\n";
            return false;
        }

        // Dispatch 1: input → mid
        {
            const void* enc_handle = stream.BeginEncoder();
            id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)enc_handle;
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:(__bridge id<MTLBuffer>)buf_in->GetNativeHandle() offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)buf_mid->GetNativeHandle() offset:0 atIndex:1];
            [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            stream.EndEncoder();
        }

        // Dispatch 2: mid → out
        {
            const void* enc_handle = stream.BeginEncoder();
            id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)enc_handle;
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:(__bridge id<MTLBuffer>)buf_mid->GetNativeHandle() offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)buf_out->GetNativeHandle() offset:0 atIndex:1];
            [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            stream.EndEncoder();
        }

        if (!stream.Flush(ctx, &err)) {
            std::cout << "FAIL (Flush: " << err << ")\n";
            return false;
        }

        if (stream.GetEncoderCount() != 2) {
            std::cout << "FAIL (expected 2 encoders, got " << stream.GetEncoderCount() << ")\n";
            return false;
        }

        // Read back and verify
        std::vector<float> output(N);
        buf_out->Read(output.data(), N * sizeof(float), 0, &err);

        for (std::size_t i = 0; i < N; ++i) {
            float expected = static_cast<float>(i + 1) * 4.0f;
            if (std::abs(output[i] - expected) > 1e-4f) {
                std::cout << "FAIL (output[" << i << "]=" << output[i] << " expected=" << expected << ")\n";
                return false;
            }
        }

        std::cout << "PASS\n";
        return true;
    }
}

// ============================================================================
// Test 3: Performance comparison — serial vs batched command buffers
// ============================================================================
static bool TestBatchingPerformance(const soc::gpu::MetalContext& ctx) {
    std::cout << "Test: Batching performance comparison... ";
    @autoreleasepool {
        std::string err;
        id<MTLDevice> device = (__bridge id<MTLDevice>)ctx.GetNativeDevice();
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)ctx.GetNativeCommandQueue();

        NSError* nsErr = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:
            @"kernel void inc_it(device float* data [[buffer(0)]],"
             "uint gid [[thread_position_in_grid]]) {"
             "  data[gid] = data[gid] + 1.0f;"
             "}"
            options:nil error:&nsErr];
        id<MTLFunction> func = [lib newFunctionWithName:@"inc_it"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&nsErr];

        const std::size_t N = 1024;
        const int kDispatches = 535;  // Simulate full decode step
        const int kIterations = 10;

        auto buf = soc::gpu::MetalBuffer::CreateShared(ctx, N * sizeof(float), "perf_buf", &err);
        std::vector<float> zeros(N, 0.0f);

        // ---- Serial: separate command buffer per dispatch ----
        buf->Write(zeros.data(), N * sizeof(float), 0, &err);
        auto serial_start = Clock::now();
        for (int iter = 0; iter < kIterations; ++iter) {
            for (int d = 0; d < kDispatches; ++d) {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pipeline];
                [enc setBuffer:(__bridge id<MTLBuffer>)buf->GetNativeHandle() offset:0 atIndex:0];
                [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }
        }
        double serial_us = DurationUs(serial_start, Clock::now()) / kIterations;

        // ---- Batched: single command buffer with CommandStream ----
        buf->Write(zeros.data(), N * sizeof(float), 0, &err);
        auto batched_start = Clock::now();
        for (int iter = 0; iter < kIterations; ++iter) {
            soc::gpu::CommandStream stream;
            stream.Begin(ctx, &err);
            for (int d = 0; d < kDispatches; ++d) {
                const void* enc_handle = stream.BeginEncoder();
                id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)enc_handle;
                [enc setComputePipelineState:pipeline];
                [enc setBuffer:(__bridge id<MTLBuffer>)buf->GetNativeHandle() offset:0 atIndex:0];
                [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                stream.EndEncoder();
            }
            stream.Flush(ctx, &err);
        }
        double batched_us = DurationUs(batched_start, Clock::now()) / kIterations;

        // Verify correctness: buf should have value = kIterations * kDispatches
        std::vector<float> result(N);
        buf->Read(result.data(), N * sizeof(float), 0, &err);
        float expected = static_cast<float>(kIterations * kDispatches);
        bool correct = std::abs(result[0] - expected) < 1.0f;

        double speedup = serial_us / batched_us;

        std::cout << "\n";
        std::cout << "  Serial (" << kDispatches << " CBs):  " << std::fixed << std::setprecision(1)
                  << serial_us / 1000.0 << " ms\n";
        std::cout << "  Batched (1 CB):              " << batched_us / 1000.0 << " ms\n";
        std::cout << "  Speedup:                     " << std::setprecision(1) << speedup << "x\n";
        std::cout << "  Result correctness:          " << (correct ? "PASS" : "FAIL") << "\n";

        if (!correct) {
            std::cout << "  Expected " << expected << " got " << result[0] << "\n";
            return false;
        }
        if (speedup < 5.0) {
            std::cout << "  WARNING: Speedup lower than expected (" << speedup << "x)\n";
        }
        return true;
    }
}

// ============================================================================
// Test 4: Batched MatMul simulation — chain of matmuls in one command buffer
// ============================================================================
static bool TestBatchedMatMulChain(const soc::gpu::MetalContext& ctx,
                                    soc::gpu::PipelineCache& cache) {
    std::cout << "Test: Batched MatMul chain... ";
    @autoreleasepool {
        std::string err;

        // Simulate a single layer's linear ops: 7 matmuls chained
        // (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
        const std::size_t hidden = 1024;
        const std::size_t intermediate = 2816;

        // Create shared buffers
        auto make_buf = [&](std::size_t bytes, const std::string& name) {
            return soc::gpu::MetalBuffer::CreateShared(ctx, bytes, name, &err);
        };

        auto input_buf   = make_buf(hidden * sizeof(float), "chain_input");
        auto q_buf       = make_buf(hidden * sizeof(float), "chain_q");
        auto k_buf       = make_buf(512 * sizeof(float), "chain_k");
        auto v_buf       = make_buf(512 * sizeof(float), "chain_v");
        auto o_buf       = make_buf(hidden * sizeof(float), "chain_o");
        auto gate_buf    = make_buf(intermediate * sizeof(float), "chain_gate");
        auto up_buf      = make_buf(intermediate * sizeof(float), "chain_up");
        auto down_buf    = make_buf(hidden * sizeof(float), "chain_down");
        auto q_weight    = make_buf(hidden * hidden * sizeof(float), "chain_q_w");
        auto k_weight    = make_buf(512 * hidden * sizeof(float), "chain_k_w");
        auto v_weight    = make_buf(512 * hidden * sizeof(float), "chain_v_w");
        auto o_weight    = make_buf(hidden * hidden * sizeof(float), "chain_o_w");
        auto gate_weight = make_buf(intermediate * hidden * sizeof(float), "chain_gate_w");
        auto up_weight   = make_buf(intermediate * hidden * sizeof(float), "chain_up_w");
        auto down_weight = make_buf(hidden * intermediate * sizeof(float), "chain_down_w");

        // Initialize input
        std::vector<float> init_data(hidden, 1.0f);
        input_buf->Write(init_data.data(), hidden * sizeof(float), 0, &err);

        // Initialize weights with small values
        auto init_weight = [&](std::shared_ptr<soc::gpu::MetalBuffer>& buf, std::size_t rows, std::size_t cols) {
            std::vector<float> w(rows * cols);
            for (std::size_t i = 0; i < w.size(); ++i) {
                w[i] = 0.001f * static_cast<float>(i % 100);
            }
            buf->Write(w.data(), w.size() * sizeof(float), 0, &err);
        };
        init_weight(q_weight, hidden, hidden);
        init_weight(k_weight, 512, hidden);
        init_weight(v_weight, 512, hidden);
        init_weight(o_weight, hidden, hidden);
        init_weight(gate_weight, intermediate, hidden);
        init_weight(up_weight, intermediate, hidden);
        init_weight(down_weight, hidden, intermediate);

        // Create device tensors
        using DT = soc::gpu::DeviceTensor;
        using TD = soc::gpu::TensorDesc;
        using Dtype = soc::gpu::DataType;
        auto make_dt = [](std::shared_ptr<soc::gpu::MetalBuffer>& buf, std::vector<std::size_t> shape) {
            return DT(buf, 0, TD::CreateContiguous(Dtype::kFloat32, shape));
        };

        DT input_t  = make_dt(input_buf, {1, hidden});
        DT q_t      = make_dt(q_buf, {1, hidden});
        DT k_t      = make_dt(k_buf, {1, 512});
        DT v_t      = make_dt(v_buf, {1, 512});
        DT o_t      = make_dt(o_buf, {1, hidden});
        DT gate_t   = make_dt(gate_buf, {1, intermediate});
        DT up_t     = make_dt(up_buf, {1, intermediate});
        DT down_t   = make_dt(down_buf, {1, hidden});
        DT q_w_t    = make_dt(q_weight, {hidden, hidden});
        DT k_w_t    = make_dt(k_weight, {512, hidden});
        DT v_w_t    = make_dt(v_weight, {512, hidden});
        DT o_w_t    = make_dt(o_weight, {hidden, hidden});
        DT gate_w_t = make_dt(gate_weight, {intermediate, hidden});
        DT up_w_t   = make_dt(up_weight, {intermediate, hidden});
        DT down_w_t = make_dt(down_weight, {hidden, intermediate});

        soc::gpu::MatMulParams mm_params;
        mm_params.decode_mode = true;
        mm_params.transpose_rhs = true;

        // Run serial: 7 matmuls with separate command buffers
        const int kIterations = 20;
        auto serial_start = Clock::now();
        for (int i = 0; i < kIterations; ++i) {
            soc::gpu::MatMulOp::Run(ctx, &cache, input_t, q_w_t, nullptr, nullptr, q_t, mm_params, nullptr, nullptr, &err);
            soc::gpu::MatMulOp::Run(ctx, &cache, input_t, k_w_t, nullptr, nullptr, k_t, mm_params, nullptr, nullptr, &err);
            soc::gpu::MatMulOp::Run(ctx, &cache, input_t, v_w_t, nullptr, nullptr, v_t, mm_params, nullptr, nullptr, &err);
            soc::gpu::MatMulOp::Run(ctx, &cache, q_t, o_w_t, nullptr, nullptr, o_t, mm_params, nullptr, nullptr, &err);
            soc::gpu::MatMulOp::Run(ctx, &cache, input_t, gate_w_t, nullptr, nullptr, gate_t, mm_params, nullptr, nullptr, &err);
            soc::gpu::MatMulOp::Run(ctx, &cache, input_t, up_w_t, nullptr, nullptr, up_t, mm_params, nullptr, nullptr, &err);
            soc::gpu::MatMulOp::Run(ctx, &cache, gate_t, down_w_t, nullptr, nullptr, down_t, mm_params, nullptr, nullptr, &err);
        }
        double serial_us = DurationUs(serial_start, Clock::now()) / kIterations;

        std::cout << "\n";
        std::cout << "  7 MatMuls serial:     " << std::fixed << std::setprecision(1)
                  << serial_us / 1000.0 << " ms (per iteration)\n";
        std::cout << "  Per-matmul avg:       " << serial_us / 7.0 << " µs\n";
        std::cout << "  Extrapolated 28 layers: " << serial_us * 4.0 / 1000.0 << " ms\n";

        // NOTE: Batched version requires modifying MatMulOp to accept CommandStream
        // This is Phase 1 implementation work. For now, we measure the serial baseline.
        std::cout << "  [Batched test] requires Op-level CommandStream integration (Phase 1 core work)\n";

        return true;
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

    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  PHASE 1: Command Buffer Batching Experiments                ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    std::cout << "Device: " << ctx->GetDeviceInfo().name << "\n\n";

    soc::gpu::PipelineCache cache(*ctx);
    int pass = 0, fail = 0;

    if (TestCommandStreamLifecycle(*ctx)) ++pass; else ++fail;
    if (TestBatchedDispatchCorrectness(*ctx)) ++pass; else ++fail;
    if (TestBatchingPerformance(*ctx)) ++pass; else ++fail;
    if (TestBatchedMatMulChain(*ctx, cache)) ++pass; else ++fail;

    std::cout << "\n═══════════════════════════════════════\n";
    std::cout << "Results: " << pass << " passed, " << fail << " failed\n";
    return fail > 0 ? 1 : 0;
}
