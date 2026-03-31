# GPU Optimization Master Plan — Qwen3 0.6B on M4 Mac Mini

**Target**: 100+ tokens/sec decode throughput on M4 Mac Mini (full-GPU mode)
**Model**: Qwen3 0.6B (28 layers, hidden=1024, heads=16, kv_heads=8, head_dim=64, intermediate=2816)
**Date**: 2026-03-30

---

## Hardware Analysis: M4 Mac Mini

| Spec | M4 (base) | M4 Pro |
|------|-----------|--------|
| GPU Cores | 10 | 16 |
| Memory Bandwidth | 100 GB/s | 273 GB/s |
| Max Threadgroup Size | 1024 | 1024 |
| Thread Execution Width | 32 | 32 |
| SIMD-group Matrix | Yes (Apple8+) | Yes |
| Unified Memory | Yes | Yes |

## Decode Throughput Ceiling Analysis

For decode (batch=1), the bottleneck is **memory bandwidth** (reading weights).

**Weight Size Per Token Read (all layers):**
- Per layer: q_proj(1024×1024) + k_proj(1024×512) + v_proj(1024×512) + o_proj(1024×1024) 
  + gate(1024×2816) + up(1024×2816) + down(2816×1024) + norms
- Per layer ≈ 11.5M params
- 28 layers × 11.5M = ~322M params total (+ embedding, lm_head)

| Dtype | Weight Size | M4 Base (100 GB/s) | M4 Pro (273 GB/s) |
|-------|------------|---------------------|---------------------|
| float32 | ~1.3 GB | ~77 tok/s max | ~210 tok/s max |
| float16 | ~0.65 GB | ~154 tok/s max | ~420 tok/s max |

**Conclusion**: float32 CANNOT reach 100 tok/s on M4 base. float16 is mandatory.

---

## Bottleneck Analysis (Current Codebase)

### B1: Per-Operation Command Buffer Submission (CRITICAL)
Every single op (matmul, rms_norm, rope, softmax, elementwise_mul, embedding) creates its own:
1. `[commandQueue commandBuffer]`
2. `[commandBuffer computeCommandEncoder]`
3. Encode dispatch
4. `[encoder endEncoding]`
5. `[commandBuffer commit]`
6. `[commandBuffer waitUntilCompleted]`  ← **SYNCHRONOUS WAIT**

Per decode step: ~395 command buffer submissions (28 layers × 14 ops + embedding + norm + lm_head).
Each submission costs 10-50µs overhead → **4-20ms pure overhead** per token.
At 100 tok/s target (10ms/token budget), this alone may consume 40-200% of the budget.

### B2: Float32-Only Compute (CRITICAL)
All computation is float32. This wastes half the memory bandwidth.
The current pipeline converts float16 embedding tables to float32 at load time.
Weight storage on disk may already be float16; the loader upconverts.

### B3: Single-Threaded Reduction Kernels (HIGH)
`rms_norm_f32_rowwise`: Single thread loops over entire row (1024 elements)
`softmax_f32_rowwise`: Single thread loops over entire row for max, sum, normalize
These should use SIMD group parallel reductions (32 threads cooperative).

### B4: No Kernel Fusion (MEDIUM)
- Gate+SiLU and Up projections could share the same weight read pass
- RMSNorm→Linear could be fused (read input once)
- Attention scores → Softmax → Value aggregation could be partially fused

### B5: Per-Step Buffer Allocation (MEDIUM)
`UploadTokenIds` in `GenerationContext::DecodeNextToken` allocates a new MetalBuffer every step.
`EnsureLogitsBuffer` also potentially re-allocates.

### B6: Small Tile Sizes in Tiled MatMul (LOW-MEDIUM)
Current: tile_rows=4, tile_columns=32, inner_tile=16
Threadgroup memory: lhs_tile[4][16] + rhs_tile[16][32] = 256 + 2048 = 2304 floats = 9.2KB
M4 has 32KB threadgroup memory → could use larger tiles.

---

## Optimization Phases

### Phase 1: Command Buffer Batching (Expected: 2-5x improvement)
**Idea**: Replace per-op command buffers with a single command buffer per forward pass.
Pass a shared `MTLCommandBuffer` through the op chain, only commit once at the end.

**Changes Required**:
1. Add `id<MTLCommandBuffer>` parameter to all Op::Run() methods (or use a CommandStream abstraction)
2. Remove `[commandBuffer commit]; [commandBuffer waitUntilCompleted]` from each op
3. Single commit+wait at the end of `ForwardLogitsCached`
4. Update `PipelineCache` and `FinalizeCommandBuffer` accordingly

**Risk**: Low — purely internal plumbing change, no algorithmic change.

### Phase 2: Float16 Compute Pipeline (Expected: ~2x improvement)
**Idea**: Keep weights in float16, compute in float16/mixed precision.
- Embedding tables: already float16 on disk, keep as float16
- Weight matrices: store as float16
- Intermediate activations: float16
- Final logits: can be float32 for sampling accuracy

**Changes Required**:
1. New Metal kernels: `matmul_f16_*`, `rms_norm_f16_*`, `rope_f16_*`, etc.
2. weight_loader: skip f16→f32 conversion
3. TensorDesc/DeviceTensor: proper float16 support
4. KV cache in float16

**Risk**: Medium — need to verify numerical accuracy.

### Phase 3: SIMD-Parallel Reductions (Expected: 5-10x per kernel)
**Idea**: Use `simd_sum()`, `simd_max()` for row-wise reductions.
- RMSNorm: 32 threads cooperatively reduce 1024 elements
- Softmax: 32 threads cooperative max + sum + normalize

**Changes Required**:
1. New `rms_norm_f16_simd` kernel
2. New `softmax_f16_simd` kernel
3. Update dispatch to use threadgroup-per-row instead of thread-per-row

**Risk**: Low — well-established Metal pattern.

### Phase 4: Kernel Fusion (Expected: 10-30% improvement)
**Ideas**:
- Fused MLP: gate_proj(SiLU) * up_proj in one kernel (same input, two weight reads, one output)
- Fused attention: scores→softmax→values in one pass
- Fused RMSNorm+Linear: normalize then immediately project

**Risk**: Medium — complex kernels, harder to debug.

### Phase 5: Advanced Metal Optimizations
- `simdgroup_matrix` for 8×8 matrix multiplication (M4 Apple8+)
- Larger tile sizes exploiting 32KB threadgroup memory
- `MTLResourceStorageModePrivate` for weights (GPU-only) with staging buffer upload
- Double-buffered command submission (overlap CPU scheduling with GPU execution)

---

## Experiment Tracking

All experiments recorded in this directory:
- `01_baseline_measurement.md` — Current performance measurement
- `02_command_buffer_batching.md` — Phase 1 experiment
- `03_float16_pipeline.md` — Phase 2 experiment
- `04_simd_reductions.md` — Phase 3 experiment
- etc.

## Success Criteria

| Metric | Current (est.) | Target |
|--------|---------------|--------|
| Decode tok/s | ~10-30? | 100+ |
| Command buffers/token | ~395 | 1-3 |
| Weight dtype | float32 | float16 |
| Reduction parallelism | 1 thread/row | 32 threads/row |
