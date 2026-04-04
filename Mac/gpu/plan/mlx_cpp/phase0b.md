# MLX→C++ Port — Phase 0b: MLX Framework C++ Core 조사

## 목표
ml-explore/mlx의 C++ 코어를 분석하여 MLX Python 코드가 내부적으로 어떤 C++ 함수를 호출하는지 파악한다.
이 정보가 있어야 MLX Python → C++ 변환 시 어떤 부분을 직접 구현하고 어떤 부분은 MLX C++ API를 그대로 쓸 수 있는지 판단할 수 있다.

**소스 위치**: `.repo_cache/mlx/` (shallow clone of ml-explore/mlx)

---

## 1. MLX 아키텍처 개요

| 속성 | 내용 |
|------|------|
| 언어 | C++20 |
| 빌드 | CMake (`add_library(mlx)`, static/shared) |
| 프레임워크 의존성 | Metal.framework, Foundation.framework, Accelerate.framework |
| 외부 라이브러리 | fmt (header-only) |
| Metal 커널 | ~150+ .metal 파일, pre-compiled metallib 또는 JIT |
| 배열 시스템 | lazy evaluation, unified memory, 자체 array 타입 |
| Quantization | 2/3/4/5/6/8-bit group-wise (2596줄 Metal shader) |

---

## 2. C++ API Surface (mlx/fast.h)

```cpp
namespace mlx::core::fast {
  // RMSNorm — GPU: rms_norm.metal
  array rms_norm(array x, optional<array> weight, float eps, StreamOrDevice s = {});

  // LayerNorm — GPU: layer_norm.metal
  array layer_norm(array x, optional<array> weight, optional<array> bias, float eps, StreamOrDevice s = {});

  // RoPE — GPU: rope.metal
  array rope(array x, int dims, bool traditional, optional<array> freqs,
             float base, float scale, int offset, optional<array> offsets,
             bool forward, StreamOrDevice s = {});

  // Scaled Dot Product Attention
  //   Decode: sdpa_vector.h (simdgroup per-key parallelism)
  //   Prefill: steel_attention.h (tiled GEMM-based)
  array scaled_dot_product_attention(
      array queries, array keys, array values,
      float scale, string mask_mode, optional<array> mask,
      optional<array> sinks, StreamOrDevice s = {});

  // Custom Metal Kernel JIT — custom_kernel.cpp
  CustomKernelFunction metal_kernel(
      string name, vector<string> input_names, vector<string> output_names,
      string source, string header = "", bool ensure_row_contiguous = true,
      bool atomic_outputs = false);
  //   CustomKernelFunction = function<vector<array>(inputs, shapes, dtypes, grid, threadgroup, template_args, ...)>
}
```

---

## 3. Metal Kernel 상세 분석

### 3.1 RMSNorm (rms_norm.metal, ~160줄)

| 변형 | 조건 | 패턴 |
|------|------|------|
| `rms_single_row` | axis_size ≤ looped_limit | 1 threadgroup per row |
| `rms_looped` | axis_size > looped_limit | loop over dimension |

**알고리즘**:
1. 각 스레드: N_READS개 원소의 x² 누적
2. simd_sum → threadgroup shared memory → simd_sum (2-level reduction)
3. rsqrt(mean + eps)
4. out[i] = w[i] * x[i] * inv_mean

**Dispatch** (normalization.cpp):
- threadgroup_size = simd_size × ceil(axis_size / (N_READS × simd_size))
- grid = n_rows × threadgroup_size
- 버퍼: x(0), w(1), out(2), eps(3), axis_size(4), w_stride(5)

### 3.2 RoPE (rope.metal, ~150줄)

| 변형 | 용도 |
|------|------|
| `rope_single` | 단일 시퀀스, base에서 inv_freq 계산 |
| `rope_single_freqs` | 사전 계산된 frequency 배열 사용 |
| `rope` | 배치+멀티헤드, offset 배열 지원 |

**알고리즘**:
- theta = scale × offset × inv_freq
- cos/sin 회전: rx1 = x1·cos - x2·sin, rx2 = x1·sin + x2·cos
- traditional mode: 인접 쌍 (2i, 2i+1)
- non-traditional: split-half (i, i+half_dim)

### 3.3 SDPA Vector — Decode Path (sdpa_vector.h, ~200줄)

**아키텍처**: 단일 쿼리 vs 전체 KV 시퀀스 (decode 시 1 token씩)

| 파라미터 | 값 |
|----------|----|
| BN | 32 (keys per simdgroup) |
| BD | 32 (dim per thread) |
| qk_per_thread | D / BD |
| v_per_thread | V / BD |

**알고리즘** (Online Softmax):
1. 쿼리 로드 (pre-scaled by √d)
2. 각 key block(32개): dot(q, k) → score
3. Online max/exp tracking: `new_max = max(old_max, score)`, `factor = exp(old - new)`
4. `o[j] = o[j] * factor + exp_score * v[j]`
5. Threadgroup reduction: simd_max → 최종 max, factor 재적용 → simd_sum
6. Output = o / sum_exp_score

**지원 헤드 크기**: 64, 96, 128, 256 (template instantiation)
**2-pass 변형**: 매우 긴 시퀀스용 — pass 1: partial results, pass 2: aggregate

### 3.4 Steel Attention — Prefill Path (steel_attention.h)

- Tiled GEMM 기반 attention (BQ×BK×BD block sizes)
- WM×WN warp 구성
- Causal mask, float/bool mask, attention sinks 지원
- 32 threads × simdgroup

### 3.5 Quantized MatMul (quantized.h, 2596줄)

| 커널 | 용도 |
|------|------|
| `affine_qmv` / `affine_qmv_fast` / `affine_qmv_quad` | quantized mat-vec (decode) |
| `affine_qvm` / `affine_qvm_split_k` | vec-mat (alternative layout) |
| `affine_qmm_t` / `affine_qmm_n` | quantized mat-mat (prefill) |
| `affine_gather_*` | gathered/indexed variants |

- Group-wise dequant: `x_float = (x_packed - zero) * scale`
- 지원 비트: 2, 3, 4, 5, 6, 8
- Float16/BFloat16/Float32 지원

### 3.6 Custom Kernel JIT (custom_kernel.cpp, 430줄)

**gated_delta_kernel 실행 흐름**:
1. Python: `mx.fast.metal_kernel(name, inputs, outputs, source)` 호출
2. C++: `metal_kernel()` → `CustomKernelFunction` (lambda) 반환
3. Lambda 호출 시:
   - `write_signature()`: 입력/출력 dtype에서 Metal 함수 시그니처 자동 생성
   - Template args → 소스 코드에 인라인
   - 커널 이름으로 캐시 확인 → 없으면 JIT 컴파일
4. `CustomKernel::eval_gpu()`:
   - `d.get_library(name, source)` → Metal 라이브러리 컴파일
   - `d.get_kernel(name, lib)` → MTLComputePipelineState 생성
   - 입력 버퍼 바인딩 (shape, strides 자동 추가)
   - `compute_encoder.dispatch_threads(grid, group)` 실행

### 3.7 Conv1d (conv.metal)

- General N-d convolution (unfold + GEMM)
- Steel Conv: 최적화된 tiled conv 커널 (steel/conv/)
- Depthwise conv (groups = channels) 지원

---

## 4. 포팅 전략 결정

### 세 가지 옵션 비교

| | A) MLX 라이브러리 링크 | B) Metal 커널 추출 | C) 하이브리드 (추천) |
|---|---|---|---|
| 방식 | libmlx.a 빌드/링크 | .metal 파일 복사 + 자체 dispatch | MLX 링크 + gated_delta 직접 |
| RMSNorm | `mlx::core::fast::rms_norm()` | rms_norm.metal 수동 dispatch | MLX API |
| SDPA | `mlx::core::fast::sdpa()` | sdpa_vector.h 수동 dispatch | MLX API |
| RoPE | `mlx::core::fast::rope()` | rope.metal 수동 dispatch | MLX API |
| Quantized | 자동 포함 (2596줄) | 수동 추출 필요 😱 | MLX API |
| Conv1d | MLX nn.Conv1d | conv.metal 수동 dispatch | MLX API |
| gated_delta | `metal_kernel()` JIT | 직접 .metal 파일 | `metal_kernel()` 또는 직접 |
| 장점 | 모든 최적화 무료 | 의존성 없음, 완전한 제어 | 최적화 + 유연성 |
| 단점 | C++20 필요, MLX array 체계 | 엄청난 작업량 | MLX 의존성 |

### 결정: **Option C — 하이브리드**

**근거**:
1. MLX는 Apple Silicon (M1-M4) 전용 설계, Apple 연구팀 유지보수
2. Quantized 커널 (2596줄)을 직접 작성하는 것은 비현실적
3. Steel GEMM/Attention은 고도로 최적화된 tiled 구현
4. gated_delta Metal 소스는 Python 문자열에 이미 존재 → 추출 용이
5. `metal_kernel()` API로 gated_delta shader JIT 컴파일 가능
6. SDPA vector(decode) + Steel attention(prefill) 양쪽 경로 모두 커버
7. MLX가 메모리 관리, command buffer encoding, 커널 캐싱 처리

### 핵심 과제

**과제 1: MLX 빌드 및 링크**
```bash
cd .repo_cache/mlx
cmake -B build \
  -DMLX_BUILD_METAL=ON \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DBUILD_SHARED_LIBS=OFF
cmake --build build -j8
# → build/lib/libmlx.a
```

**과제 2: MLX array 체계 적응**
- MLX array = lazy evaluation + unified memory
- Option A: MLX array를 전면 채택 (가장 단순)
- Option B: zero-copy wrapper (MTLBuffer ↔ MLX array)
- **추천**: MLX array 전면 채택 → mlx-lm Python 코드와 1:1 매핑 가능

**과제 3: gated_delta 커널**
- Metal 소스 (Python 문자열, ~50줄)를 .metal 파일로 추출
- `mlx::core::fast::metal_kernel()` 또는 직접 MTLComputePipelineState
- **추천**: `metal_kernel()` 사용 → 시그니처 자동 생성, 캐싱 무료

---

## 5. Qwen3.5 구성 요소 → MLX C++ API 매핑

| Qwen3.5 구성 요소 | MLX Python | MLX C++ API |
|---|---|---|
| RotaryEmbedding | `mx.fast.rope()` | `mlx::core::fast::rope()` |
| RMSNormGated | `mx.fast.rms_norm()` + gate | `mlx::core::fast::rms_norm()` + gate |
| Attention (MHA) | `mx.fast.scaled_dot_product_attention()` | `mlx::core::fast::scaled_dot_product_attention()` |
| GDN kernel | `mx.fast.metal_kernel(source=...)` | `mlx::core::fast::metal_kernel(source=...)` |
| Linear | `nn.Linear` → `mx.matmul` | `mlx::core::matmul()` |
| SwiGLU | `silu(gate) * x` | `mlx::core::sigmoid(gate) * gate * x` |
| Conv1d | `nn.Conv1d(groups=)` | `mlx::core::conv1d()` |
| Softplus | `mx.log(1 + mx.exp(x))` | `mlx::core::softplus()` |
| KV Cache | ArraysCache, KVCache | 직접 구현 (MLX array로) |

---

## 상태
- [x] mlx C++ core 구조 분석
- [x] Metal kernel 소스 위치 정리 (rms_norm, rope, sdpa_vector, steel_attn, quantized, conv, custom_kernel)
- [x] 직접 링크 vs Metal 코드 추출 결정 → **하이브리드 (Option C)**
- [x] Qwen3.5 → MLX C++ API 매핑 완료
- [x] libmlx.a 빌드 완료 (34MB at .repo_cache/mlx/build/libmlx.a)
- [x] mlx.metallib 빌드 완료 (.repo_cache/mlx/build/mlx/backend/metal/kernels/mlx.metallib)
