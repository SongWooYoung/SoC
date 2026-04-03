# PyTorch MPS Backend — 연산별 Metal 디스패치 방식

> 소스: `pytorch/aten/src/ATen/native/mps/operations/`
> 커밋 기준: 2025-07 main branch

## 개요

PyTorch MPS 백엔드는 세 가지 방식으로 연산을 구현한다:

| 방식 | 특징 | 성능 |
|------|------|------|
| **Custom Metal Shader** | `.metal` 파일에서 `MTLComputePipelineState`로 직접 디스패치 | 최적화 가능, edge case 처리 |
| **MPSGraph** | Apple의 `MPSGraph` API로 그래프 기반 실행 | 범용적, Apple 최적화 내장 |
| **MPSNDArray / MPSMatrix** | MPS framework의 저수준 API 직접 호출 | 특수 케이스 (대형 bmm 등) |

공통 패턴: `LookUpOrCreateCachedGraph<CachedGraph>(key, ...)` — 한 번 빌드한 MPSGraph를 키 기반으로 캐시하여 재사용.

---

## 연산별 분석

### 1. Matrix Multiply (Linear)

**소스**: `LinearAlgebra.mm`

| 경로 | 방식 | 조건 |
|------|------|------|
| `do_metal_mm()` | Custom Metal (`matmul_` shader, TILE_DIM=16) | `use_metal_mm()` == true |
| `do_metal_bmm()` | Custom Metal (`naive_bmm_` shader) | batched mm, 결과 < 2^32 원소 |
| `do_metal_addmm()` | Custom Metal (`addmm_` shader) | addmm + use_metal_mm 조건 |
| `MPSGraph matrixMultiplication` | MPSGraph `matrixMultiplicationWithPrimaryTensor:secondaryTensor:` | 기본 경로 (대부분의 경우) |
| `MPSNDArrayMatrixMultiplication` | MPS 저수준 API (tiled) | bmm 결과 > 2^32 원소 |

**`use_metal_mm()` 결정 조건** (하나라도 참이면 custom Metal 사용):
- 환경변수 `PYTORCH_MPS_PREFER_METAL` 설정됨
- 정수형 타입 (integral types)
- complex 타입 + inner dim > 2048
- macOS < 14.0 + 큰 stride/size (알려진 MPSGraph 버그 회피)
- fp16 padding 버그 회피

**Quantized**: 미지원 (MPS 네이티브 양자화 없음)
**레이아웃**: row-major (C-contiguous) 기본. strided 입력 시 내부 contiguous() 호출 가능.

**Stage 1 전략**: MPSGraph `matrixMultiplicationWithPrimaryTensor` 사용.
**Stage 2 고려**: TILE_DIM=16 custom shader는 small matrix에서 유리. 대형 행렬은 MPSGraph가 내부적으로 SIMD-group matmul 활용.

---

### 2. Softmax

**소스**: `SoftMax.mm`

| 방식 | API |
|------|-----|
| MPSGraph | `[mpsGraph softMaxWithTensor:inputTensor axis:dim name:nil]` |

- macOS < 15에서 ChannelsLast 메모리 포맷일 경우 reshape/transpose 우회 처리
- `log_softmax` → `[mpsGraph logarithmWithTensor:[mpsGraph softMaxWithTensor:...]]`

**Stage 1 전략**: MPSGraph softmax 직접 사용.

---

### 3. Activation Functions (SiLU, GELU, Softplus 등)

**소스**: `Activation.mm`

모든 activation은 **MPSGraph** 기반:

| 함수 | MPSGraph 구현 |
|------|--------------|
| **SiLU** | `x / (1 + exp(-x))` — `divisionWithPrimaryTensor(x, additionWithPrimaryTensor(one, exponentWithTensor(negativeWithTensor(x))))` |
| **GELU (normcdf)** | `x * 0.5 * (1 + erf(x / sqrt(2)))` |
| **GELU (tanh)** | `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))` |
| **Softplus** | `log(1 + exp(beta * x)) / beta`, threshold 초과 시 `x`로 fallback |
| **Mish** | `x * tanh(softplus(x))` |
| **PReLU** | `x > 0 ? x : weight * x` |

**Stage 1 전략**: MPSGraph로 SiLU 구현 (`Qwen3.5의 주요 activation`).
**Stage 2 고려**: SiLU를 fused Metal shader로 (matmul + silu gate를 한 커널에).

---

### 4. LayerNorm / RMSNorm

**소스**: `Normalization.mm`

| 연산 | 방식 | 상세 |
|------|------|------|
| **LayerNorm forward** | Custom Metal Shader | `layer_norm_single_row_` (N ≤ 1024×4), `layer_norm_looped_` (N > 1024×4) |
| **LayerNorm backward** | MPSGraph | `normalizationWithTensor:...` |
| **BatchNorm** | MPSGraph | `normalizationWithTensor:meanTensor:varianceTensor:gammaTensor:betaTensor:epsilon:` |

**LayerNorm Forward 디테일**:
- `lib.getPipelineStateForFunc("layer_norm_single_row_" + typeStr)` — single threadgroup per row
- `lib.getPipelineStateForFunc("layer_norm_looped_" + typeStr)` — 큰 N일 때 루프 기반
- weight/bias 적용도 shader 내부에서 처리

**RMSNorm 관련**:
- PyTorch에는 native `RMSNorm` MPS 구현 없음
- Qwen3.5에서는 `RMSNorm = rsqrt(mean(x²) + eps) * x * weight`
- `rsqrt` → `Unary`, `mean` → `ReduceOps`, `mul` → `BinaryOps` 으로 분해됨
- 또는 LayerNorm shader를 참고하여 fused RMSNorm kernel 작성 가능

**Stage 1 전략**: elementwise 분해 (rsqrt, mean, mul)로 MPSGraph 구현.
**Stage 2 전략**: LayerNorm custom shader 패턴 참고하여 fused RMSNorm Metal shader 작성.

---

### 5. RoPE (Rotary Positional Embedding)

**소스**: 전용 MPS 구현 없음

- PyTorch MPS에 RoPE 전용 커널 없음
- `cos`, `sin` → `Activation.mm` (MPSGraph `cosWithTensor:`, `sinWithTensor:`)
- `outer` → `LinearAlgebra.mm` 또는 elementwise broadcast
- apply_rotary: `(x * cos) + (rotate_half(x) * sin)` → elementwise BinaryOps

**Stage 1 전략**: cos/sin은 MPSGraph, rotate_half + apply는 elementwise ops 조합.
**Stage 2 전략**: fused RoPE Metal shader (cos/sin LUT + apply를 한 커널에).

---

### 6. Conv1d

**소스**: `Convolution.mm`

| 경로 | 방식 | 조건 |
|------|------|------|
| Standard Conv | MPSGraph `MPSGraphConvolution2DOpDescriptor` | `groups == 1` 또는 `weight.size(1) != 1` |
| Depthwise Conv | MPSGraph `MPSGraphDepthwiseConvolution3DOpDescriptor` | `groups > 1 && weight.size(1) == 1` |
| Conv3d | MPSGraph `MPSGraphConvolution3DOpDescriptor` | 3D 입력 |

**주의**: Conv1d는 내부적으로 Conv2d로 변환 (unsqueeze → conv2d → squeeze).
- macOS 15+에서 `ChannelsLast` 메모리 포맷 지원
- `MPSGraphConvolution2DOpDescriptor`에 padding/stride/dilation 설정

**Qwen3.5 관련**: `causal_conv1d` 사용 — short kernel (d_conv=4), 1D depthwise conv.

**Stage 1 전략**: MPSGraph Conv2D 경유 (PyTorch 동일 경로).
**Stage 2 전략**: short-kernel causal conv1d 전용 Metal shader (d_conv=4 hardcode).

---

### 7. Elementwise Binary Ops (add, mul, sub, div)

**소스**: `BinaryOps.mm`

**모두 MPSGraph 기반**:

| 연산 | MPSGraph API |
|------|-------------|
| `add` | `additionWithPrimaryTensor:secondaryTensor:` (alpha 지원 시 `binary_op_kernel` 경유) |
| `sub` | `subtractionWithPrimaryTensor:secondaryTensor:` |
| `mul` | `multiplicationWithPrimaryTensor:secondaryTensor:` |
| `div` | `divisionWithPrimaryTensor:secondaryTensor:` |
| `pow` | `powerWithPrimaryTensor:secondaryTensor:` |

**패턴**: `binaryOpTensor()` 함수가 공통 래퍼:
1. dtype promotion (`c10::promoteTypes`)
2. `LookUpOrCreateCachedGraph` 로 MPSGraph 캐시
3. scalar input은 `getMPSGraphTensorFromScalar`로 변환
4. `runMPSGraph(stream, graph, feeds, output)`

**비교 연산**: `equal`, `notEqual`, `lessThan`, `greaterThan` 등 — 모두 MPSGraph.

**`add_sub_lerp_template`**: alpha ≠ 1.0 일 때 `binary_op_kernel("add_alpha", ...)` → `BinaryKernel.h`에서 Metal shader 디스패치 (fused alpha*other + self).

**Stage 1 전략**: MPSGraph elementwise 직접 사용.

---

### 8. Embedding Lookup

**소스**: `Indexing.mm`

| 연산 | 방식 | API |
|------|------|-----|
| **Forward** (`nn.Embedding`) | MPSGraph | `[mpsGraph gatherWithUpdatesTensor:indicesTensor:axis:batchDimensions:]` |
| **Backward** | MPSGraph | `[mpsGraph scatterNDWithUpdatesTensor:indicesTensor:shape:batchDimensions:mode:MPSGraphScatterModeAdd]` |

- `nn.Embedding.forward` → 내부적으로 `index_select(weight, 0, indices)` 호출
- `index_select_out_mps` → MPSGraph `gatherWithUpdatesTensor`
- Float16 backward 시 float32로 캐스팅 후 scatterND → 다시 fp16 (Apple MPSGraph 버그 회피)

**Stage 1 전략**: MPSGraph gather 사용.

---

### 9. Reduce Ops (Argmax, Sum, Mean 등)

**소스**: `ReduceOps.mm`

| 연산 | 방식 | API |
|------|------|-----|
| `sum` | MPSGraph | `reductionSumWithTensor:axes:` |
| `mean` | MPSGraph | `reductionMeanWithTensor:axes:` (nanmean: NaN→0 치환 후 sum/count) |
| `prod` | MPSGraph | `reductionProductWithTensor:axes:` |
| `argmax` | MPSGraph | `reductionArgMaximumWithTensor:axis:` |
| `argmin` | MPSGraph | `reductionArgMinimumWithTensor:axis:` |
| `all` | MPSGraph | `reductionAndWithTensor:axes:` |
| `any` | MPSGraph | `reductionOrWithTensor:axes:` |
| `norm` | Custom Metal | `norm_` + type suffix, `lib.getPipelineStateForFunc(...)` |

**Stage 1 전략**: argmax는 MPSGraph `reductionArgMaximumWithTensor` 사용.

---

### 10. Indexing Operations (기타)

**소스**: `Indexing.mm`

| 연산 | 방식 |
|------|------|
| `index` (advanced indexing) | Custom Metal (`index_select_Nbit`) |
| `index_put` | Custom Metal (`index_put_Nbit` / `index_put_accumulate_`) |
| `index_copy` | Custom Metal (`index_copy_dense_` / `index_copy_strided_`) |
| `index_fill` | Custom Metal (2-pass mask 또는 scatter 방식) |
| `index_add` | MPSGraph `scatterWithDataTensor:...mode:MPSGraphScatterModeAdd` |
| `masked_fill` | Custom Metal (`masked_fill_scalar_`) |
| `nonzero` | Custom Metal (3-step: prefix sum → block offsets → scatter) |
| `flip` | MPSGraph `reverseTensor:axes:` |

**Qwen3.5 관련**: 대부분 불필요. `index_select` (embedding)만 사용.

---

### 11. BPE Tokenize / Detokenize

CPU에서 처리. GPU 디스패치 불필요.

---

## 요약 테이블

| 연산 | 구현 방식 | Stage 1 전략 | Stage 2 기회 |
|------|-----------|-------------|-------------|
| MatMul (Linear) | **Dual**: Custom Metal + MPSGraph | MPSGraph | simdgroup_matmul 커스텀 |
| Softmax | MPSGraph | MPSGraph | online softmax shader |
| SiLU | MPSGraph (분해) | MPSGraph | fused gate shader |
| GELU | MPSGraph (분해) | MPSGraph | — |
| Softplus | MPSGraph (분해) | MPSGraph | — |
| LayerNorm | **Custom Metal** (forward) | elementwise 분해 | fused RMSNorm shader |
| RoPE | 전용 커널 없음 (분해) | elementwise 분해 | fused RoPE shader |
| Conv1d | MPSGraph (Conv2D 경유) | MPSGraph | short-kernel shader |
| add/mul/sub/div | MPSGraph | MPSGraph | — |
| Embedding | MPSGraph (gather) | MPSGraph | — |
| Argmax | MPSGraph | MPSGraph | — |
| Sum/Mean | MPSGraph | MPSGraph | — |
| BPE tokenize | CPU | CPU | CPU |

## Stage 1 핵심 결론

**대부분의 연산이 MPSGraph 기반** → Stage 1에서는 MPSGraph API를 주로 사용하면 PyTorch MPS와 동일한 수준 달성 가능.

Custom Metal shader가 사용되는 곳:
1. **MatMul** — edge case (small, integral, complex)에서만
2. **LayerNorm** — forward pass 전체
3. **Norm** (vector norm) — 전체
4. **Indexing** — advanced indexing, index_put, masked_fill, nonzero

→ Qwen3.5 추론에서 실제 필요한 custom shader: **없음** (Stage 1 기준).
→ Stage 2에서 fused kernel 작성 대상: RMSNorm, RoPE, SiLU gate, attention.
