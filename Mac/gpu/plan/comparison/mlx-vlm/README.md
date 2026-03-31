# mlx-vlm Comparison

## Repo Snapshot

- Repository: `mlx-vlm`
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
