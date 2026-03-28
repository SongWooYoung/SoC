# Architecture Plan

## 1. End State

프로젝트의 end state는 다음 아키텍처다.

```text
HF/Qwen checkpoint
  -> LLM_interpreter export
  -> manifest + raw weight bins + tokenizer assets
  -> Host asset layer
  -> Metal device/bootstrap layer
  -> Buffer/allocator layer
  -> Tensor/view layer
  -> Kernel registry + pipeline cache
  -> Op dispatch + command scheduler
  -> Qwen3 module graph
  -> Generation runtime (prefill + decode + KV cache)
  -> Future graph/backward layer
```

이 구조는 `Metal backend 없는 MPSGraph wrapper`가 아니라, Metal을 직접 제어하는 작은 LLM runtime이어야 한다.

## 2. Layered Runtime Model

### 2.1 Asset Layer

책임:

1. `manifest.json` parsing
2. tensor metadata index 생성
3. raw `.bin` weight file 읽기
4. tokenizer asset 읽기
5. GPU prepack 여부를 판단할 metadata 제공

절대 금지:

1. 파일명 추측
2. weight name 재구성 추측
3. config 미검증 상태 사용

### 2.2 Metal Device Layer

책임:

1. `MTLDevice` 생성
2. `MTLCommandQueue` 소유
3. feature set / threadgroup / simdgroup capability 조회
4. pipeline library / function loading
5. debug label, validation, capture hook 관리

필수 이유:

1. Apple GPU feature 차이를 여기서 흡수해야 한다.
2. kernel code는 device capability를 직접 모르면 안 된다.
3. runtime이 startup 시점에 specialization policy를 고정해야 한다.

### 2.3 Buffer / Allocator Layer

책임:

1. `MTLBuffer` ownership
2. storage mode 관리 (`shared`, `private`, 필요 시 `managed`)
3. aligned suballocation
4. scratch / activation / KV cache arena 제공
5. upload / download staging policy 관리

중요:

1. weight는 read-mostly다.
2. activation은 짧은 lifetime을 가진다.
3. KV cache는 길고 커지는 lifetime을 가진다.
4. 이 셋은 allocator 정책이 다르므로 분리해야 한다.

### 2.4 Tensor Layer

책임:

1. dtype
2. shape
3. stride
4. buffer + offset
5. device location
6. logical view와 physical layout 분리

Phase 1 단순화:

1. Metal device resident tensor 우선
2. contiguous row-major 우선
3. non-owning view 최소 지원

그러나 stride와 logical shape는 구조에 미리 포함해야 한다.

### 2.5 Kernel Layer

책임:

1. `.metal` compute kernels
2. threadgroup/shared memory policy
3. specialization constants 또는 function constants 관리
4. matmul, RMSNorm, RoPE, softmax, sampling, KV cache update 구현

중요:

1. kernel은 model semantics를 몰라야 한다.
2. kernel은 buffer, offset, shape, specialization info만 알아야 한다.
3. Qwen attention semantics는 op/module layer에서 조합한다.

### 2.6 Pipeline Cache Layer

책임:

1. `MTLComputePipelineState` 캐시
2. dtype / tile size / head dim / group size별 specialization 캐시
3. startup prewarm 정책
4. shader versioning과 invalidation 정책

GPU runtime에서 pipeline cache는 optional이 아니다. pipeline state 생성 비용을 runtime hot path에 두면 안 된다.

### 2.7 Op Layer

책임:

1. shape inference
2. dtype rule
3. kernel dispatch
4. temporary buffer 요청
5. command encoding 순서 정의

예시:

1. `matmul`
2. `rmsnorm`
3. `rope_apply`
4. `attention_prefill`
5. `attention_decode`
6. `fused_add_rmsnorm`

### 2.8 Module Layer

책임:

1. parameter binding
2. `forward()` 정의
3. submodule 구성
4. weight layout contract 유지

필수 module:

1. `Embedding`
2. `Linear`
3. `RMSNorm`
4. `QwenAttention`
5. `QwenMLP`
6. `QwenBlock`
7. `QwenCausalLM`

### 2.9 Scheduler Layer

책임:

1. command buffer 경계 결정
2. prefill / decode path 분리
3. kernel fusion scheduling
4. CPU-GPU synchronization 최소화
5. double-buffered scratch 또는 pipelined execution 전략

이 레이어는 CPU runtime에는 거의 없던 GPU 전용 계층이다.

### 2.10 Runtime Layer

책임:

1. prompt tokenize
2. prefill
3. decode loop
4. KV cache update
5. sampling
6. detokenize
7. CPU reference와 비교 가능한 debug hooks 제공

### 2.11 Future Graph / Train Layer

책임:

1. graph capture
2. backward op registration
3. optimizer state buffer
4. checkpoint write-back

Phase 1에서 구현하지 않아도 다음 규칙은 지켜야 한다.

1. tensor view가 non-owning descriptor여야 한다.
2. op dispatch가 explicit temporary allocation을 사용해야 한다.
3. module parameter registry가 있어야 한다.

## 3. Current Gaps vs Required Architecture

현재 `Mac/gpu`는 다음 상태다.

1. 사실상 runtime 계층이 없음
2. shader asset 없음
3. Metal bootstrap 없음
4. tokenizer / manifest loader 없음
5. buffer / tensor / module abstraction 없음

즉, 현재는 `placeholder` 수준이고, `device -> buffer -> kernel -> module -> runtime` 전부를 새로 세워야 한다.

## 4. Hard Constraints

반드시 지켜야 할 제약:

1. Metal API를 직접 제어한다.
2. CPU fallback과 GPU optimized path를 분리한다.
3. host-device transfer를 hot path에서 최소화한다.
4. export format과 GPU prepack format의 계약을 문서화한다.
5. fused optimization이 기본 구조를 깨지 않도록 계층을 분리한다.

## 5. Phase 1 Architectural Freeze

Phase 1에서 얼려야 하는 인터페이스:

1. `MetalContext`
2. `MetalBuffer`
3. `BufferArena`
4. `TensorDesc`
5. `DeviceTensor`
6. `KernelRegistry`
7. `PipelineCache`
8. `CommandScheduler`
9. `KVCache`
10. `WeightUploader`
11. `Qwen3TemplateBuilder`
12. `GenerationContext`

이 인터페이스는 다음 규칙을 따라야 한다.

1. `DeviceTensor`는 layout metadata와 physical buffer를 분리한다.
2. `KVCache` public layout은 canonical shape를 노출하고 packed implementation은 private optimization으로 둔다.
3. `KernelRegistry`는 specialization key를 명시적으로 가진다.
4. `CommandScheduler`는 prefill과 decode를 다른 execution policy로 다룬다.