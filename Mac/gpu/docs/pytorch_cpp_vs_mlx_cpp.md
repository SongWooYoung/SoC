# PyTorch→C++ vs MLX→C++ 구현 비교

## 개요
동일한 Qwen3.5 모델을 두 가지 경로로 C++에 포팅한다:
1. **py_cpp**: transformers/modeling_qwen3_5.py (PyTorch) → C++ (Accelerate BLAS, CPU)
2. **mlx_cpp**: mlx-vlm/models/qwen3_5 (MLX) → C++ (Metal GPU, Apple Silicon 최적화)

이 문서는 두 구현의 구조적 차이를 정리하여 최적화 방향을 안내한다.

---

## 1. 소스 구조

| 구분 | py_cpp | mlx_cpp |
|------|--------|---------|
| 원본 파일수 | 3 (.py) + modular | 5 (.py) + gated_delta |
| 핵심 파일 | modeling_qwen3_5.py (1300줄) | language.py (650줄) + gated_delta.py (280줄) |
| .h 매핑 | modeling.h (단일 파일, 1100줄) | language.h + gated_delta.h (분리) |
| 의존성 | torch, transformers | mlx, mlx-lm |

## 2. GatedDeltaNet 구현 차이 (가장 중요)

### 2.1 Conv1d 처리

| 항목 | py_cpp | mlx |
|------|--------|-----|
| Prefill | causal_conv1d_bf16() — GEMM 기반 | nn.Conv1d — cat + conv |
| Decode step | causal_conv1d_step() — shift+insert+convolve | cat([conv_state, x]) → conv → update state |
| 상태 layout | `[conv_dim, ks-1]` (channels-first) | `[B, ks-1, conv_dim]` (batch-first, channels-last) |
| 버그 위험 | shift-then-convolve에서 double-counting 가능 | cat 방식으로 버그 불가 |

### 2.2 Delta Rule 연산

| 항목 | py_cpp | mlx |
|------|--------|-----|
| Gating | `g = -exp(A_log) * softplus(a + dt_bias)` → `exp(g)` 별도 적용 | `g = exp(-exp(A_log) * softplus(a + dt_bias))` — 직접 decay factor |
| State layout | `[Hv, Dk, Dv]` — no batch | `[B, Hv, Dv, Dk]` — batch-first, Dv/Dk 순서 반대 |
| Prefill/Decode | 별도 함수 (forward_chunk / forward_recurrent) | 통합 (gated_delta_update → kernel 또는 ops) |
| GPU 가속 | 없음 (CPU 스칼라 루프) | Metal shader (SIMD sum, threadgroup 최적화) |

### 2.3 Q/K Normalization

| 항목 | py_cpp | mlx |
|------|--------|-----|
| 방식 | L2 normalization | RMSNorm (weight=None) + scale factor |
| 수식 | `q/‖q‖₂`, `k/‖k‖₂` | `rms_norm(q)*inv_scale², rms_norm(k)*inv_scale` |
| 차이 | L2norm = x / sqrt(sum(x²)) | RMSNorm = x / sqrt(mean(x²)+eps) |
| 영향 | sqrt(dim) 차이 — 상수 스케일이므로 attention 결과 동일 | |

## 3. Attention 구현 차이

| 항목 | py_cpp | mlx |
|------|--------|-----|
| Q projection | split into query + gate (head_dim*2) | 동일 |
| K/V projection | 별도 linear | 동일 |
| Q/K norm | RMSNorm per head | 동일 |
| RoPE | pre-compute table, apply rotation | compute per call, interleaved MRoPE |
| SDPA | 수동 loop (Q@K, softmax, @V) | mx.fast.scaled_dot_product_attention (Metal) |
| Gate | sigmoid(gate) * attn_out → O proj | 동일 |

## 4. 캐시 구조

| 구분 | py_cpp | mlx |
|------|--------|-----|
| KV cache | KVCache struct (append, flat vectors) | KVCache class (update_and_fetch) |
| GDN cache | GDNCache struct (conv_state, recurrent_state, conv_weight_f32) | ArraysCache(size=2) — [conv_state, recurrent_state] |
| Position tracking | ModelCache.seq_offset | cache.offset (per-layer) |

## 5. 성능 비교 (Phase 4 baseline)

| 지표 | py_cpp (CPU Accelerate) | Python transformers (MPS) | mlx_cpp (목표) |
|------|------------------------|---------------------------|----------------|
| prefill (20 tok) | ~1100ms | ~250ms | <250ms |
| decode (ms/tok) | ~580ms | ~135ms | <100ms |
| throughput | ~1.7 tok/s | ~7 tok/s | >10 tok/s |

## 6. MLX→C++ 포팅 전략

### 방법 A: MLX C++ API 직접 링크
- mlx 라이브러리를 빌드하여 링크
- mx::array, mx::fast::* 함수를 C++에서 직접 호출
- 장점: 검증된 Metal kernel 재사용
- 단점: mlx 의존성 추가

### 방법 B: Metal kernel만 추출
- gated_delta_kernel의 Metal 소스를 .metal 파일로 분리
- MTLLibrary / MTLComputePipelineState로 직접 dispatch
- 장점: 의존성 최소화
- 단점: 버퍼 관리, synchronization 직접 구현

### 방법 C: Hybrid
- 기본 연산 (matmul, RMSNorm 등)은 Accelerate + 직접 Metal kernel
- GDN의 gated_delta_kernel만 MLX Metal 소스에서 추출/수정
- 가장 현실적인 접근

## 7. 핵심 최적화 기회

1. **GatedDeltaNet Metal kernel**: py_cpp의 CPU 루프를 Metal로 → 10-50x 가능
2. **SDPA Metal**: 수동 attention 루프를 Metal SDPA로 → 5-10x
3. **Fused RMSNorm+Gate**: SwiGLU gate와 RMSNorm을 하나의 Metal kernel로
4. **Unified memory zero-copy**: bf16 weights를 MTLBuffer(shared)로 직접 접근
5. **Batch matmul**: 여러 head의 Q@K를 하나의 batched GEMM으로
