# mlx-lm Comparison

## Repo Snapshot

- Repository: `ml-explore/mlx-lm`
- Local snapshot: `/tmp/soc_compare/mlx-lm`
- Inspected commit: `6ddfdda`

## 왜 이 레포를 보는가

`mlx-lm`은 Metal kernel을 직접 쓰는 엔진은 아니지만, Apple이 설계한 MLX execution model 위에서 generation, KV cache, prompt cache, sampling, memory budgeting을 어떻게 분리하는지 보여준다. 우리 코드의 runtime 계층을 정리할 때 참고할 가치가 크다.

## 직접 읽은 파일

- `mlx_lm/generate.py`
- `mlx_lm/models/cache.py`

## 구조 요약

`mlx-lm`의 generation 경로는 대략 이렇다.

- 모델과 tokenizer를 로드한다.
- generation 전용 stream을 따로 둔다.
- wired memory limit를 모델 크기에 맞춰 조정한다.
- prompt cache와 KV cache를 모델 특성에 맞게 만든다.
- sampling은 logits processor와 sampler object로 분리한다.
- KV cache는 기본형, rotating, quantized, batch형을 상황에 따라 바꾼다.

즉 Metal kernel보다 runtime policy와 memory policy가 더 잘 정리되어 있다.

## op / runtime 비교

### 1. generation stream

`mlx-lm`

- `generation_stream = mx.new_stream(mx.default_device())`
- generation path를 위한 stream을 명시적으로 분리한다.

우리 코드

- [`metal_context.mm`](/Volumes/990pro/Documents/SoC/Mac/gpu/src/metal/metal_context.mm)는 queue 하나를 들고, 대부분 op가 즉시 command buffer를 만들고 wait한다.
- stream 개념은 experimental `CommandStream`에만 부분적으로 있었다.

차이:

- `mlx-lm`은 "generation용 실행 컨텍스트"를 분명히 나눈다.
- 우리는 execution context와 scheduler policy가 거의 섞여 있다.

### 2. wired limit / memory budgeting

`mlx-lm`

- `wired_limit()`에서 model bytes와 `max_recommended_working_set_size`를 기준으로 경고와 limit 조정을 한다.
- Apple Silicon working set을 runtime 수준에서 의식한다.

우리 코드

- device info는 [`metal_context.mm`](/Volumes/990pro/Documents/SoC/Mac/gpu/src/metal/metal_context.mm)에서 읽지만, runtime policy 결정에 적극 활용하지 않는다.
- `Private` weight/KV cache는 만들지만 전체 working set budget 정책은 약하다.

차이:

- 우리는 residency는 만지지만 budget policy가 없다.

### 3. KV cache 계층

`mlx-lm`

- `KVCache`
- `RotatingKVCache`
- `QuantizedKVCache`
- `BatchKVCache`
- `BatchRotatingKVCache`
- prompt cache save/load/trim

즉 캐시는 "하나의 구현"이 아니라 policy 집합이다.

우리 코드

- [`kv_cache.mm`](/Volumes/990pro/Documents/SoC/Mac/gpu/src/runtime/kv_cache.mm)는 append/view 중심의 단일 `Private` KV cache 구현이다.
- prompt reuse, rotating window, quantized KV는 아직 없다.

차이:

- 우리 쪽은 cache policy 선택지가 거의 없다.
- long context 대응이나 memory/latency tradeoff 실험이 어렵다.

### 4. sampling

`mlx-lm`

- sampling은 `make_sampler`로 분리된 정책 객체다.
- generation core와 sampling implementation이 느슨하게 결합돼 있다.

우리 코드

- [`sampler.cpp`](/Volumes/990pro/Documents/SoC/Mac/gpu/src/runtime/sampler.cpp)에서 GPU/CPU top-k fallback을 직접 관리한다.

차이:

- 우리는 sampling kernel 실험이 runtime 구조와 더 강하게 묶여 있다.
- `mlx-lm`은 sampling policy 교체가 더 쉽다.

## 우리 코드에 대한 직접 아이디어

### 채택 후보 1. KV cache policy 분리

`KVCache`를 단일 구현으로 두지 말고 다음을 인터페이스 수준에서 나누는 게 좋다.

- contiguous cache
- rotating cache
- future quantized cache

당장 quantized KV를 구현하지 않더라도, 구조를 먼저 열어 두면 long-context 실험이 쉬워진다.

### 채택 후보 2. prompt cache 저장/복원

`mlx-lm`처럼 prompt cache를 저장하고 재사용하면 benchmark나 반복 프롬프트 비교에서 prefill 비용을 분리해 볼 수 있다. 지금 우리 벤치는 prefill/decode를 분리해 보긴 하지만, prompt cache artifact가 없어서 반복 실험이 다소 무겁다.

### 채택 후보 3. memory budget를 runtime policy로 연결

`recommended_max_working_set_size`를 읽고만 끝내지 말고:

- f16/f32 weight policy
- KV cache max length
- temporary arena size
- aggressive experimental path 허용 여부

같은 정책을 자동으로 조절하는 방향이 맞다.

## 우리 코드와 다른 철학

`mlx-lm`은 "모든 걸 하나의 최적 kernel로 해결"하려 하기보다, runtime policy 계층을 잘게 나눠서 memory와 generation behavior를 제어한다. 지금 우리 코드는 kernel 튜닝 비중이 크고 runtime policy 계층은 상대적으로 얇다.

즉 이 레포에서 가져와야 할 핵심은 새로운 kernel보다:

- cache policy abstraction
- prompt cache lifecycle
- memory budget aware runtime

이다.
