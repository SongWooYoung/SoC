# Experiment 02: Phase 1 — Command Buffer Batching

**Date**: 2026-03-30
**Device**: Apple M4 (Mac Mini)

## Hypothesis
Replacing per-op command buffer (create+commit+waitUntilCompleted) with a single batched
command buffer will reduce per-token overhead from ~70ms to <1ms.

## Implementation: CommandStream
Created `include/metal/command_stream.h` + `src/metal/command_stream.mm`:
- `Begin()`: Creates a single MTLCommandBuffer
- `BeginEncoder()` / `EndEncoder()`: Creates/ends MTLComputeCommandEncoder per op
- `Flush()`: Single commit + waitUntilCompleted at the end

## Test Results (test_phase1_batching.mm)

### Correctness Tests
| Test | Result |
|------|--------|
| CommandStream lifecycle | PASS |
| Batched dispatch correctness (chain: input×2→mid×2→output=input×4) | PASS |

### Performance Tests
| Metric | Serial | Batched | Speedup |
|--------|--------|---------|---------|
| 535 dispatches | 69.7 ms | 0.7 ms | **105x** |
| Per-dispatch overhead | 130 µs | 1.3 µs | 100x |

### MatMul Chain Baseline (7 serial matmuls per layer)
| Metric | Value |
|--------|-------|
| 7 MatMuls serial | 2.9 ms |
| Per-matmul (with CB overhead) | 419 µs |
| Extrapolated 28 layers | 11.7 ms |

## Analysis
- Serial CB overhead: 70ms = 7 tok/s (matches baseline measurement)
- With batching: 0.7ms overhead → negligible
- Remaining bottleneck: actual kernel compute time
- 7 serial MatMuls = 2.9ms → includes 7×130µs = 0.9ms CB overhead
- Pure compute per matmul: ~280µs (after removing CB overhead)
- With batching, 7 matmuls should take ~1.8ms
- 28 layers × 1.8ms = ~50ms → ~20 tok/s (still need float16 for 100)

## 2026-03-31 Safety Correction

위 수치는 `microbenchmark` 관점에서는 유효했지만, 실기기 end-to-end decode에 그대로 확대 적용하면 안 됐다. Mac mini M4 32GB에서는 `full decode step` 또는 `multi-layer range`를 하나의 giant command buffer로 묶는 방식이 `WindowServer` fault를 유발했다.

따라서 현재 해석은 다음으로 수정한다.

1. batching은 유효한 최적화 수단이다.
2. 하지만 granularity가 너무 크면 unsafe하다.
3. 재도입은 `layer` 또는 그보다 작은 bounded batching부터 해야 한다.

상세한 재현 조건과 현재 정책은 `Mac/gpu/plan/13_GPU_FAULT_POSTMORTEM.md`와 `Mac/gpu/test/errors/01_windowserver_gpu_fault.md`를 따른다.

## Next Steps
1. Integrate CommandStream into all Op::Run() methods (optional parameter)
2. Use CommandStream in QwenBlock::RunDecode() for full decode batching
3. Measure actual end-to-end improvement with real model

## Files Added
- `include/metal/command_stream.h`
- `src/metal/command_stream.mm`
- `test/gpu_optimization/test_phase1_batching.mm`
