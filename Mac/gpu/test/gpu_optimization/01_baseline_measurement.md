# Experiment 01: Baseline Measurement

**Date**: 2026-03-30
**Device**: Apple M4 (Mac Mini)
**GPU Cores**: 10
**Unified Memory**: 25,559 MB recommended working set
**SIMD Matrix**: YES
**Thread Execution Width**: 32

---

## Results

### Command Buffer Overhead
| Test | Avg (µs) | Min (µs) | Max (µs) |
|------|----------|----------|----------|
| Single noop command buffer | 131.4 | 92.1 | 733.3 |
| 535 serial command buffers | 70,100 | 52,750 | 77,518 |
| 535 dispatches in 1 command buffer | 433.4 | 388.4 | 542.5 |

### Batching Speedup: **161.7x**

### Per-Op Kernel Timings (decode mode)
| Op | Avg (µs) | Min (µs) | Note |
|----|----------|----------|------|
| MatMul 1×1024×1024 | 307.9 | 250.9 | q/k/v/o projections |
| MatMul 1×1024×2816 | 356.3 | 321.5 | gate/up/down projections |
| RMSNorm 1×1024 | 262.0 | 234.0 | Single-thread reduction |

### Throughput Estimates (decode)
| Scenario | ms/token | tok/s |
|----------|----------|-------|
| CB overhead only (serial) | 70.1 | 14 |
| CB overhead only (batched) | 0.4 | 2307 |
| MatMul compute (28 layers) | 64.4 | ~15 |
| Total serial (overhead + compute) | 134.5 | **7** |

---

## Analysis

### Finding 1: Command buffer overhead DOMINATES
- 70.1ms of pure overhead per token (535 serial waitUntilCompleted)
- With batching: only 0.4ms → **161.7x** improvement
- This confirms: Phase 1 (batching) is the #1 priority

### Finding 2: MatMul kernels are SLOW  
- 307.9µs per 1×1024×1024 matmul = **very slow** for a memory-bandwidth-bound op
- Theoretical: 1024×1024×4 bytes = 4MB at 100 GB/s = **40µs**
- Actual: 307.9µs → only **13% bandwidth utilization**
- This includes command buffer overhead (131µs base) + kernel inefficiency

### Finding 3: RMSNorm is embarrassingly slow
- 262µs for 1×1024 (4KB of data)
- At 100 GB/s: 4KB should take < 0.04µs
- Overhead is 99.98% command buffer + launch overhead
- With batching, this will drop to near-zero effective cost

### Finding 4: Current decode rate = ~7 tok/s
- 134.5ms per token
- Dominated by overhead, not compute

---

## Path to 100 tok/s

After Phase 1 (batching), the command buffer overhead drops from 70ms to ~0.4ms.
Remaining bottleneck: **64.4ms of actual matmul compute** (still includes per-op CB overhead).

With batching, actual per-matmul overhead should drop to ~1µs.
Pure compute estimate for (1×1024)×(1024×col):
- Memory read: 1024*col*4B / (100 GB/s) = col*40.96ns
- 1×1024×1024: ~42µs (optimistic), ~50-60µs (realistic)
- 1×1024×2816: ~115µs 

With batching + optimized compute:
- Per layer: 4×50 + 3×115 = 545µs (optimistic)
- 28 layers: 15.3ms → **65 tok/s**

To reach 100 tok/s (10ms/token budget):
- Need float16: halves memory read → ~7.7ms compute → **130 tok/s** ✓

## Updated Priority
1. **Phase 1: Command Buffer Batching** → 7 → 15-20 tok/s (remove CB overhead from total)
2. **Phase 1.5: Optimize MatMul kernel** → 20 → 40-60 tok/s 
3. **Phase 2: Float16** → 60 → 100+ tok/s

## Files Modified
- `test/gpu_optimization/bench_baseline.mm` — benchmark code
- `Makefile` — added `bench-baseline` target
