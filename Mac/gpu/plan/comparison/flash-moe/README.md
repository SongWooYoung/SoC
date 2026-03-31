# flash-moe Comparison

## Repo Snapshot

- Repository: `danveloper/flash-moe`
- Local snapshot: `/tmp/soc_compare/flash-moe`
- Inspected commit: `3601d41`

## 왜 이 레포를 보는가

이 레포는 dense Transformer 전체 엔진은 아니지만, Apple Silicon에서 Metal compute, dequant matvec, fused expert path, SSD/I/O, unified memory contention까지 실제로 부딪힌 기록이 매우 풍부하다. 우리 쪽에서 반복 실수 방지 문서를 쓸 때 참고 가치가 크다.

## 직접 읽은 파일

- `metal_infer/main.m`
- `metal_infer/shaders.metal`
- `docs/io-and-gpu-exploration.md`
- `docs/optimization-experiments-q4.md`

## 구조 요약

`flash-moe`는 본질적으로 "SSD에서 expert weight를 읽어 와 Metal kernel로 dequant matvec를 돌리는 엔진"이다. dense Qwen decoder 전체를 다루는 우리 코드와는 다르지만, 아래 항목은 직접 비교할 수 있다.

- command queue / command buffer를 만들고 shader를 runtime compile 또는 metallib에서 로드한다.
- expert matvec는 4-bit packed weight + per-group scale/bias를 kernel 안에서 dequant하면서 바로 accumulate한다.
- kernel 종류를 여러 개 두고 실측으로 채택/폐기한다.
- CPU, SSD I/O, page cache, unified memory contention을 compute와 같은 수준으로 다룬다.

## op 비교

### 1. MatMul / projection

`flash-moe`

- [`shaders.metal`](/tmp/soc_compare/flash-moe/metal_infer/shaders.metal)에서 `dequant_matvec_4bit`, `dequant_matvec_4bit_fast`, `dequant_matvec_4bit_v3` 등 여러 variant가 있다.
- 핵심 아이디어는:
  - packed `uint32`에서 nibble을 꺼낸다.
  - `bf16` scale/bias를 `float32`로 올린다.
  - dequant와 matvec를 한 kernel에서 fused 한다.
  - `threadgroup` shared memory에 input vector를 cache한다.
  - SIMD reduction과 vector load를 적극 쓴다.

우리 코드

- [`matmul_op.mm`](/Volumes/990pro/Documents/SoC/Mac/gpu/src/op/matmul_op.mm)는 현재 `float32 lhs`와 `float32/float16 rhs`의 dense matmul 중심이다.
- dequant fused path는 없다.
- decode tile 정책은 shape 기반 heuristic이다.

차이:

- `flash-moe`는 weight bandwidth 자체를 줄인 quantized fused kernel이다.
- 우리는 아직 dense float 계열이라 memory bandwidth pressure가 훨씬 크다.
- 지금 `MatMul`이 전체 GPU 시간 대부분을 차지하는 이유가 여기서도 설명된다.

### 2. activation / MLP

`flash-moe`

- `fused_gate_up_swiglu`를 둬서 gate와 up projection 결과를 계산하고 바로 `silu(gate) * up`를 만든다.
- `weighted_sum`, `rms_norm` 같은 작은 op도 별도 최적화 kernel이 있다.

우리 코드

- `GateProj`, `UpProj`, `SwiGLU`, `DownProj`가 분리되어 있다.
- decode MLP fusion을 한 번 시도했지만 긴 decode에서 회귀가 나와 폐기했다.

차이:

- `flash-moe`의 fusion은 quantized expert path에 맞는 구조다.
- 우리 dense path는 gate/up weight를 둘 다 읽는 비용이 너무 커서, 단순 fusion만으로는 이익이 작았다.

### 3. batching / command encoding

`flash-moe`

- 실험 기록상 "모든 걸 한 encoder에 몰아넣는 방식"은 득이 거의 없거나 손해였다.
- cluster affinity를 기대하고 한 expert의 gate/up/down을 더 강하게 묶었지만 오히려 느려졌다.
- 문서 결론은 "GPU scheduler를 억지로 통제하지 마라"에 가깝다.

우리 코드

- full/layer `CommandStream` batching은 실제로 fault 또는 regression을 냈다.

차이:

- 둘 다 "너무 큰 수동 batching"이 잘 안 맞는다.
- `flash-moe`는 성능 회귀였고, 우리는 더 나쁘게는 GPU fault까지 갔다.

### 4. sampling / non-matmul op

`flash-moe`

- expert weighted sum, rms norm 등은 GPU에 두되, 시스템 레벨 병목은 I/O와 unified memory contention으로 본다.

우리 코드

- GPU top-k sampler는 실제로 CPU fallback보다 느렸다.
- 그래서 [`sampler.cpp`](/Volumes/990pro/Documents/SoC/Mac/gpu/src/runtime/sampler.cpp)에서 기본 CPU fallback으로 돌린다.

차이:

- 둘 다 작은 GPU op 하나를 억지로 GPU에 올리는 것보다, end-to-end 파이프라인에서 실제 병목이 뭔지 보는 쪽이 중요하다는 결론을 준다.

## 이 레포에서 배울 것

### 1. microbenchmark 승자를 바로 채택하지 않는다

`flash-moe` 문서는 LUT dequant, vector load, compression, prefetch, routing prediction 같은 실험을 실제 full pipeline에서 다시 검증하고 폐기했다. 우리도 이미 같은 실수를 반복했고, [`02_performance_regressions.md`](/Volumes/990pro/Documents/SoC/Mac/gpu/test/errors/02_performance_regressions.md)에 남기고 있다. 이 원칙은 계속 유지해야 한다.

### 2. unified memory contention을 설계 제약으로 본다

문서상 `F_RDADVISE` prefetch가 I/O를 줄여도 GPU를 느리게 만든 이유는 SSD DMA와 GPU compute가 같은 memory controller를 쓰기 때문이다. 우리 코드도 Apple Silicon에서 `shared/private` residency와 blit 전략을 조심해야 한다.

### 3. harmful optimization을 명시적으로 금지한다

이 레포는 실패한 아이디어를 꽤 선명하게 문서화했다. 우리도 같은 방식으로 "방법론 자체가 틀린 게 아니라, 어떤 사용 방식이 문제인지"를 기록해야 한다.

## 우리 코드에 대한 직접 아이디어

### 채택 후보 1. decode dominant projection만 별도 특화

`DownProjDecode`, `OProjDecode`, `LMHeadDecode`가 제일 크다. 모든 matmul을 한 번에 바꾸기보다, 이 세 shape에 특화된 kernel variant를 만든 뒤 full benchmark로 채택 여부를 결정한다.

### 채택 후보 2. vectorized load는 isolated kernel이 아니라 full graph에서만 채택

`flash-moe`도 vector load variant가 isolated idea로는 그럴듯했지만 실제 pipeline에서는 회귀가 있었다. 우리도 `float4` load 실험을 하더라도 quick run과 full benchmark 둘 다 통과해야 채택해야 한다.

### 채택 후보 3. optimization blacklist 문서 유지

다음 항목은 계속 금지 또는 opt-in 실험만 허용:

- giant `CommandStream`
- stale `gpu.metallib` 의존
- microbenchmark만 보고 kernel variant 채택
- sampling 같은 작은 op를 GPU에 올려 end-to-end를 악화시키는 변경

## 우리 코드에 대한 결론

`flash-moe`가 주는 가장 큰 교훈은 "GPU kernel만 잘 짜면 된다"가 아니라, Apple Silicon에서는 메모리 시스템과 스케줄링을 같이 봐야 한다는 점이다. 우리 쪽에서는 I/O 스트리밍 대신 dense float weight bandwidth가 병목이지만, end-to-end 기준으로 micro-optimization을 계속 검증해야 한다는 점은 완전히 같다.
