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

## 2026-04-01 GitHub 재검토: prefill / decode 최적화

GitHub 메인라인에서 다시 확인한 핵심 파일은 다음과 같다.

- `mlx_lm/generate.py`
- `mlx_lm/models/cache.py`
- `mlx_lm/cache_prompt.py`
- `mlx_lm/server.py`

### prefill에서 실제로 하는 일

1. `generate_step()`는 `prefill_step_size`를 기본 옵션으로 받고, prompt를 이 step 크기만큼 잘라서 순차 처리한다.
2. 각 prefill chunk마다 model 호출 후 `mx.eval([c.state for c in prompt_cache])`로 KV state를 고정하고 `mx.clear_cache()`로 working set을 비운다.
3. prompt cache는 `make_prompt_cache()`, `save_prompt_cache()`, `load_prompt_cache()`로 별도 lifecycle을 가지며, prefill 결과를 artifact로 저장해 반복 prompt를 재사용할 수 있다.
4. batch prefill도 따로 있다. `BatchGenerator._process_prompts()`는 left padding, right padding, merged cache, checkpoint token을 조합해 여러 prompt를 chunked prefill로 처리한다.
5. long context 대응은 cache 종류로 푼다. `KVCache`, `RotatingKVCache`, `ChunkedKVCache`, `BatchKVCache`, `BatchRotatingKVCache`가 모두 prefill 형태와 메모리 budget에 맞춰 선택된다.

### decode에서 실제로 하는 일

1. decode는 별도 `generation_stream = mx.new_stream(...)` 위에서 돌아가고, `_step()`은 이 stream 안에서 logits 계산과 sampling 직전 처리까지 수행한다.
2. `mx.async_eval()`를 이용해 다음 token 계산을 미리 걸어 두고, host는 detokenize와 bookkeeping만 한다. 즉 decode는 작은 비동기 겹침을 기본 동작으로 사용한다.
3. KV quantization도 decode 중간에 들어간다. `maybe_quantize_kv_cache()`가 offset이 `quantized_kv_start`를 넘으면 cache를 quantized form으로 바꾼다.
4. speculative decode도 `_prefill()`과 `_step()`을 따로 갖고 있어, prefill과 decode의 비용 모델이 구조적으로 분리돼 있다.
5. server path는 `prompt-concurrency`, `completion-batch-size`, `prefill-step-size`, prompt cache LRU를 모두 runtime option으로 드러낸다.

### 지금 우리에게 바로 적용할 점

1. `prefill_step_size`를 `Mac/gpu`에 넣고, 각 chunk 뒤에 temp arena / Metal cache를 명시적으로 정리하는 흐름을 추가한다.
2. prompt cache artifact를 도입해서 benchmark 시 prefill과 decode를 실제 artifact 기준으로 분리한다.
3. `KVCache`를 단일 구현으로 두지 말고, 최소한 `contiguous`, `rotating`, `future quantized` 인터페이스를 먼저 연다.
4. `recommended_max_working_set_size`를 그냥 로깅하는 대신, prefill step 크기와 future kv max length에 연결한다.

### 이번 재검토의 결론

`mlx-lm`의 prefill 최적화는 `chunked prefill + prompt cache + memory budget`, decode 최적화는 `generation stream + async_eval + cache policy switching`에 있다. 우리 쪽에서 가장 빨리 옮길 수 있는 건 새 kernel이 아니라 `prefill_step_size`, `prompt cache`, `KV policy abstraction`이다.
