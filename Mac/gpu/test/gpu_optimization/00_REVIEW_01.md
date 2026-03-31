# Review #1: Agent Critique of Optimization Master Plan

**Date**: 2026-03-30
**Reviewer**: Explore Agent (thorough mode)

## Key Corrections

### 1. Command Buffer Count: UNDERESTIMATED
- **My estimate**: ~395/token  
- **Actual**: ~535/token (27% more)
- **Missed**: KV cache blit ops (56 buffers for key+value per 28 layers), extra attention sub-ops (q_norm, k_norm = 2 per layer)

### 2. Per-Decode Layer = 19 ops (not 14):
1. Input RMSNorm (1)
2. Q/K/V projections (3)
3. Q/K head norms (2) ← missed
4. Q/K RoPE (2)
5. KV cache append key (1) ← missed
6. KV cache append value (1) ← missed
7. Attention scores (1)
8. Softmax (1)
9. Attention values (1)
10. O projection (1)
11. Post RMSNorm (1)
12. Gate projection (1)
13. Up projection (1)
14. Elementwise mul (1)
15. Down projection (1)

**Total**: 28 × 19 + 3 (emb + final_norm + lm_head) = **535**

### 3. Bandwidth Analysis: Correct ceiling, but irrelevant until overhead fixed
- 535 × 30µs = ~16ms CPU stall per token
- Even with infinite compute speed, 16ms/token = 62 tok/s maximum
- **Phase 1 is non-negotiable**

### 4. Weight Loader: Confirmed float16→float32 conversion at load time
- `HalfToFloat()` called during model loading
- All tensors stored as kFloat32 in GPU memory
- 650MB→1.3GB memory waste

### 5. Missed Issues
- KV cache uses blit command buffers (56 extra per token)
- MTLResourceStorageModePrivate for weights (10-15% bandwidth gain)
- Phase 2-5 gains are multiplicative ONLY if Phase 1 succeeds

## Revised Priority
1. **Phase 1: Command Buffer Batching** → 10→50 tok/s (CRITICAL)
2. **Phase 2: Float16 Pipeline** → 50→100 tok/s (HIGH)
3. **Phase 3+**: Fine-tuning → 100→120+ tok/s (MEDIUM)

## Action Items
- [x] Update master plan with correct counts
- [ ] Build Phase 1: CommandStream abstraction
- [ ] Benchmark baseline before any changes
- [ ] Build Phase 1 experiment
