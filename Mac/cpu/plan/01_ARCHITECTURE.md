# Architecture Plan

## 1. End State

프로젝트의 end state는 다음 아키텍처다.

```text
HF/Qwen checkpoint
  -> LLM_interpreter export
  -> manifest + raw weight bins + tokenizer assets
  -> C++ IO layer
  -> Storage/Tensor layer
  -> Kernel/Op layer
  -> NN Module layer
  -> Qwen3 model graph
  -> Generation runtime (prefill + decode + KV cache)
  -> Future autograd/train layer
```

이 구조는 `backend 없는 작은 PyTorch`를 C++로 만드는 방향과 정확히 일치해야 한다.

## 2. Layered Runtime Model

### 2.1 Asset Layer

책임:

1. `manifest.json` parsing
2. tensor metadata index 생성
3. raw `.bin` weight file 읽기
4. tokenizer asset 읽기

절대 금지:

1. 파일명 추측
2. state_dict 키 재구성 추측
3. config 필드 미검증 상태 사용

### 2.2 Storage Layer

책임:

1. raw memory ownership
2. mmap/file-backed loading 지원
3. aligned allocation
4. arena/pool allocator 추후 연결

필수 이유:

1. LLM inference는 weight 대부분이 read-only다.
2. activation, KV cache, scratch buffer는 수명이 짧다.
3. ownership과 lifetime을 분리해야 메모리 최적화가 가능하다.

### 2.3 Tensor Layer

책임:

1. dtype
2. shape
3. stride
4. storage pointer + offset
5. contiguous 여부
6. view/reshape

Phase 1 단순화:

1. CPU only
2. contiguous row-major only
3. slice/view 최소 지원

이 단순화는 구현량을 줄이지만, stride 필드는 미리 설계에 포함해야 이후 training/autograd와 transpose-view를 얹기 쉽다.

### 2.4 Kernel Layer

책임:

1. 가장 낮은 수준의 CPU loop/SIMD kernel
2. matmul, rmsnorm, rope, silu, softmax, embedding lookup
3. fp16/bf16/fp32 accumulation 정책
4. thread parallelism

중요:

1. Kernel layer는 model semantics를 몰라야 한다.
2. kernel은 입력 pointer, shape, stride, dtype를 받아 계산만 한다.
3. tensor object 조합은 op layer에서 처리한다.

### 2.5 Op Layer

책임:

1. shape inference
2. dtype rule
3. kernel dispatch
4. temporary tensor allocation

예시:

1. `matmul(Tensor a, Tensor b)`
2. `softmax_last_dim(Tensor x)`
3. `rmsnorm(Tensor x, Tensor weight, epsilon)`
4. `rope_apply(Tensor q, Tensor k, position_offset, rope_theta)`

### 2.6 Module Layer

책임:

1. parameter ownership/binding
2. `forward()` 정의
3. submodule 구성

필수 module:

1. `Embedding`
2. `Linear`
3. `RMSNorm`
4. `QwenAttention`
5. `QwenMLP`
6. `QwenBlock`
7. `QwenCausalLM`

### 2.7 Runtime Layer

책임:

1. prompt tokenize
2. prefill
3. decode loop
4. KV cache update
5. sampling
6. detokenize

### 2.8 Future Train Layer

책임:

1. autograd graph
2. backward kernels
3. optimizer
4. checkpoint save
5. mixed precision training policy

Phase 1에서는 구현하지 않더라도 아래 설계는 미리 유지한다.

1. Tensor가 non-owning view를 표현할 수 있어야 한다.
2. Module parameter registry가 있어야 한다.
3. Op dispatch 구조가 backward hook을 달 수 있는 형태여야 한다.

## 3. Current Gaps vs Required Architecture

현재 `Mac/cpu`는 다음 상태다.

1. matrix 중심의 header-only 연산 모음
2. attention, KV cache, packed/quantized matmul 일부 프로토타입 존재
3. 실제 manifest loader 없음
4. tokenizer 없음
5. Tensor abstraction 없음
6. Module graph 없음
7. generation runtime 없음

즉, 현재 구현은 `Kernel/Math prototype` 수준이고, 목표로 가려면 `Storage/Tensor/IO/Module/Runtime` 계층을 새로 세워야 한다.

## 4. Hard Constraints

반드시 지켜야 할 제약:

1. backend dependency 없음
2. CPU memory layout을 직접 제어
3. Qwen3 config 기반으로 model graph 생성
4. export format과 runtime format의 계약을 문서화
5. inference path가 training path를 막지 않도록 계층을 분리

## 5. Phase 1 Architectural Freeze

Phase 1에서 얼려야 하는 인터페이스:

1. `DType`
2. `Shape`
3. `Storage`
4. `Tensor`
5. `ManifestLoader`
6. `WeightLoader`
7. `Tokenizer`
8. `Module`
9. `GenerationContext`
10. `KVCache`
11. `Qwen3TemplateBuilder`
12. `TokenizerRuntimeExport`

이 인터페이스들은 조사 결과를 반영해 다음 규칙을 따라야 한다.

1. `Tensor`는 contiguous-first지만 `stride`와 `offset`을 구조에 포함한다.
2. `KVCache` public layout은 canonical shape만 노출하고 packed layout은 private optimization으로 둔다.
3. `Module`은 tied parameter를 표현할 수 있어야 한다.
4. `GenerationContext`는 thinking / non-thinking template 선택과 special token table에 접근할 수 있어야 한다.

이 12개 인터페이스가 Phase 1의 구조적 골격이다.
