# mlx-vlm Comparison

## Repo Snapshot

- Repository: `Blaizzy/mlx-vlm`
- Local snapshot: `/tmp/soc_compare/mlx-vlm`
- Inspected commit: `1f0c0ff`

## 왜 이 레포를 보는가

`mlx-vlm`은 multimodal inference 레포지만, 실제 generation과 cache 정책 상당수를 `mlx-lm`에서 재사용한다. 즉 "모델별 입력 준비와 공통 generation core를 분리하는 법"을 보여준다. 우리 코드도 앞으로 text-only baseline을 유지한 채 기능을 늘릴 때 참고할 수 있다.

## 직접 읽은 파일

- `mlx_vlm/generate.py`
- `mlx_vlm/utils.py`

## 구조 요약

`mlx-vlm`은 크게 두 층으로 나뉜다.

- 입력 준비층:
  - image/audio loading
  - resize / processor kwargs
  - multimodal prompt construction
  - `prepare_inputs`, `group_images_by_shape`
- generation 실행층:
  - MLX stream
  - wired limit
  - sampler / logits processor
  - KV quantization, prefill step size

즉 실행 엔진은 공통화하고, 모델별 입력 조립만 분리한다.

## 우리 코드와 비교

### 1. prefill batching

`mlx-vlm`

- `--prefill-step-size`를 별도 옵션으로 둔다.
- 긴 prefill에서 peak memory와 속도 tradeoff를 명시적으로 조절한다.

우리 코드

- prefill은 [`qwen_causal_lm.cpp`](/Volumes/990pro/Documents/SoC/Mac/gpu/src/model/qwen_causal_lm.cpp)에서 full token batch를 그대로 블록에 흘린다.
- step-size control은 없다.

차이:

- 우리는 긴 prompt에서 prefill memory/latency 제어 수단이 부족하다.

### 2. cache / generation core 재사용

`mlx-vlm`

- `maybe_quantize_kv_cache`, `make_sampler`, `wired_limit` 등 공통 generation policy를 재사용한다.
- multimodal repo인데도 generation core를 별도 계층으로 둔다.

우리 코드

- text model에 맞춘 runtime이 비교적 직접 결합돼 있다.

차이:

- 코드 분리가 약해 feature 실험이 core runtime 변경으로 번지기 쉽다.

### 3. input preparation

`mlx-vlm`

- `load_model`, `load_tokenizer`, `prepare_inputs`, `AutoProcessor` 등을 통해 입력 준비를 실행 경로와 분리한다.
- shape grouping, resize normalization 같은 단계가 명확하다.

우리 코드

- 현재는 text tokenizer + manifest 기반 bundle loader 중심이다.
- 입력 준비는 상대적으로 단순하지만, 그만큼 runtime benchmark와 input preprocessing이 섞일 여지가 있다.

### 4. quantization / model load policy

`mlx-vlm`

- config와 weight metadata를 읽고 quantization mode, activation quantization, 모듈 skip 정책을 모델 로드 단계에서 결정한다.
- model conversion dtype도 명시적으로 관리한다.

우리 코드

- loader는 bundle manifest 기반으로 비교적 단순하고, experimental f16 weight path는 제한적이다.

차이:

- 우리는 load-time policy branching이 아직 작다.

## 우리 코드에 대한 직접 아이디어

### 채택 후보 1. prefill step-size 도입

decode 병목과 별개로, 긴 prompt에서 prefill을 작은 step으로 나누는 옵션을 도입하면:

- peak temporary memory 감소
- command buffer 길이 제한
- profiling 분해 향상

을 동시에 얻을 수 있다.

### 채택 후보 2. benchmark input path와 runtime path 분리

현재도 benchmark script가 분리돼 있지만, prompt preparation과 model execution 사이 계층을 더 나누면 "실제 kernel/runtime 성능"과 "입력 가공/토크나이즈"를 더 명확히 구분할 수 있다.

### 채택 후보 3. loader policy를 명시적 옵션 집합으로 정리

`mlx-vlm`처럼 load-time config를 구조화하면:

- f16/f32 weights
- future quantized weights
- activation quantization
- cache policy

를 manifest와 runtime 옵션으로 더 깔끔하게 제어할 수 있다.

## 우리 코드에 대한 결론

`mlx-vlm`은 우리와 같은 low-level Metal kernel 레포는 아니지만, 공통 generation core와 입력/모델 준비 계층을 분리한 방식은 배울 가치가 크다. 특히 prefill step-size와 cache policy를 runtime 옵션으로 드러내는 방식은 바로 참고할 수 있다.

## 2026-04-01 GitHub 재검토: prefill / decode 최적화

GitHub 메인라인에서 다시 확인한 핵심 파일은 다음과 같다.

- `mlx_vlm/generate.py`
- `mlx_vlm/server.py`
- `mlx_vlm/models/qwen2_vl/language.py`
- `mlx_vlm/models/qwen2_5_vl/language.py`
- `mlx_vlm/models/molmo_point/molmo_point.py`

### prefill에서 실제로 하는 일

1. `generate_step()`는 `prefill_step_size`를 직접 인자로 받고, 문서와 CLI 둘 다에서 "peak memory를 줄이기 위한 chunked prefill"로 설명한다.
2. multimodal 입력은 먼저 `prepare_inputs()`와 `get_input_embeddings()`로 정리되고, 그 뒤에만 language model prefill로 넘어간다. 즉 입력 준비층과 prefill 실행층이 분리돼 있다.
3. `inputs_embeds.shape[1] > prefill_step_size`이면 chunked prefill을 수행하고, 각 chunk 후 `mx.eval([c.state for c in prompt_cache])`와 `mx.clear_cache()`를 호출한다.
4. 일부 모델은 `no_chunked_prefill` 플래그로 이를 금지한다. 즉 chunked prefill도 모델 적합성 기준으로 gated 된다.
5. Qwen2-VL 계열 language model은 RoPE index 계산을 prefill stage에서 한 번만 수행하고, chunked prefill 호환을 위해 mask shape mismatch를 따로 처리한다.

### decode에서 실제로 하는 일

1. decode는 `generation_stream` 위에서 `_step()`을 반복하며, multimodal 공통 출력인 `cross_attention_states` 또는 `encoder_outputs`를 다음 step kwargs로 넘긴다.
2. `BatchGenerator`는 `prefill_batch_size`와 `completion_batch_size`를 따로 들고 있어, prompt 처리와 completion 처리의 병렬도를 분리한다.
3. KV cache quantization은 `mlx_lm.generate.maybe_quantize_kv_cache`를 재사용해서 decode 중간에 켠다.
4. server는 `PREFILL_STEP_SIZE`, `KV_BITS`, `KV_GROUP_SIZE`, `MAX_KV_SIZE`, `QUANTIZED_KV_START`를 environment/runtime option으로 바로 묶어 둔다.
5. 일부 모델은 prefill과 generation forward를 아예 분리한다. 예를 들어 `molmo_point`는 `_prefill_forward()`와 `_generate_forward()`를 별도 함수로 갖는다.

### 지금 우리에게 바로 적용할 점

1. text-only 엔진이어도 `prefill path`와 `decode path` entry를 분리하면 profiling과 future feature 추가가 쉬워진다.
2. `prefill_batch_size`와 `completion_batch_size`를 다른 knob로 두면 long prompt benchmark와 steady-state decode benchmark를 분리해서 볼 수 있다.
3. 입력 준비 경로와 runtime 경로를 분리해 benchmark가 tokenizer / preprocessing 오버헤드에 덜 흔들리게 해야 한다.
4. multimodal은 당장 필요 없지만, `prepare_inputs -> embeddings -> language runtime` 3층 분리는 우리 text runtime에도 그대로 유효하다.

### 이번 재검토의 결론

`mlx-vlm`의 prefill 최적화는 `chunked prefill + embeddings 기반 prefill + model별 opt-out`, decode 최적화는 `generation stream + batch generator + shared cache policy`에 있다. 우리 쪽에서 바로 적용할 것은 `prefill/decode entry 분리`, `prefill batch knob`, `input preparation 분리`다.
