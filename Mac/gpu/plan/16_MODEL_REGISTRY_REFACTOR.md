# Model Registry Refactor Plan

## Goal

`Mac/gpu`를 단일 `Qwen3` 구현체에서 `shared engine + model-specific implementations` 구조로 재편한다.

최종 목표는 다음과 같다.

1. 공용 low-level runtime은 유지한다.
2. 모델별 실행 순서와 loader, planner, module graph는 `gpu/models/<model>/` 아래로 내린다.
3. `gpu_infer` 바이너리는 유지하되, 내부적으로 `model registry`를 통해 모델 구현체를 선택한다.
4. profiling, sampling, prompt/decode runtime option은 공용 옵션 계층으로 올린다.
5. 이후 `qwen3.5`처럼 다른 구조를 가진 모델을 같은 엔진 위에 추가할 수 있어야 한다.

## Current State

현재 기준으로 공용층과 모델 전용층은 아래처럼 섞여 있다.

### Shared enough today

1. `metal/`
2. `buffer/`
3. `tensor/`
4. `kernel/`
5. `op/`
6. `runtime/sampler.*`
7. `runtime/kv_cache.*`
8. `runtime/runtime_policy.*`

### Qwen3-coupled today

1. `model/qwen_causal_lm.*`
2. `model/qwen_model_loader.*`
3. `module/qwen_attention.*`
4. `module/qwen_block.*`
5. `module/qwen_mlp.*`
6. `runtime/command_scheduler.*`
7. `runtime/generation_context.*`
8. `infer.mm`

핵심 문제는 이름만 generic인 계층이 실제론 `Qwen3 decode graph`를 직접 품고 있다는 점이다.

## Target Layout

```text
Mac/gpu/
  include/
    metal/
    buffer/
    tensor/
    kernel/
    op/
    runtime/
  models/
    model_registry.h
    model_runner.h
    qwen3/
      qwen3_registry.h
      qwen3_runner.h
      qwen3_loader_adapter.h
  src/
    metal/
    buffer/
    tensor/
    kernel/
    op/
    runtime/
  models/
    model_registry.cpp
    qwen3/
      qwen3_registry.cpp
      qwen3_runner.cpp
      qwen3_loader_adapter.cpp
```

초기 단계에서는 기존 `include/model`, `src/model`, `include/module`, `src/module`를 바로 지우지 않는다.
대신 `models/qwen3/`에서 현재 구현을 감싸는 adapter를 두고, 점진적으로 실제 구현을 이동한다.

## Shared Interfaces

### 1. `ModelArchitecture`

역할:

1. manifest 또는 CLI에서 어떤 모델 구현체를 선택할지 결정한다.
2. `gpu_infer`는 concrete class 이름 대신 이 enum과 registry를 사용한다.

### 2. `ModelRegistry`

역할:

1. `ManifestData`와 `--model-type`를 바탕으로 모델을 결정한다.
2. 추후 `qwen3`, `qwen3_5`, `llama`, `phi` 등을 등록 가능한 구조로 만든다.

초기 단계에서는 다음만 제공한다.

1. `ResolveModelSelection()`
2. `ModelArchitectureName()`
3. `ModelArchitectureDisplayName()`

### 3. `ModelRunner`

최종 역할:

1. prefill
2. decode
3. logits projection
4. scheduler/planner access
5. KV cache contract

하지만 1차 리팩터링에서는 full generic interface까지 한 번에 넣지 않는다.
우선 registry와 qwen3 adapter를 세우고, 이후 `GenerationContext`와 `CommandScheduler`를 generic runner interface로 바꾼다.

## Migration Strategy

### Phase A. Entry and Registry

1. `models/model_registry.*` 추가
2. `models/qwen3/qwen3_registry.*` 추가
3. `infer.mm`에 `--model-type` 추가
4. `infer.mm`가 직접 `QwenModelLoader`를 고르지 않고 registry를 통해 `qwen3`를 선택하게 변경

이 단계의 목적은 바이너리 진입점에서 concrete model selection을 분리하는 것이다.

### Phase B. Qwen3 Adapters

1. `models/qwen3/qwen3_loader_adapter.*`
2. `models/qwen3/qwen3_runner.*`

초기에는 내부적으로 기존 `QwenCausalLM`, `QwenModelLoader`를 그대로 호출한다.
즉 behavior는 같고 dependency 방향만 바뀐다.

### Phase C. Runtime De-coupling

1. `GenerationContext`가 `QwenCausalLM` 대신 generic runner를 받도록 변경
2. `CommandScheduler`에서 Qwen-specific decode plan을 `models/qwen3/planner/...`로 이동
3. 공용 scheduler는 command-buffer orchestration만 남긴다

### Phase D. Module Move

1. `qwen_attention`, `qwen_block`, `qwen_mlp`를 `models/qwen3/modules/`로 이동
2. `qwen_causal_lm`, `qwen_model_loader`를 `models/qwen3/`로 이동
3. include path와 test path를 새 위치로 전환

### Phase E. Options and Profiling

공용 runtime option 계층으로 끌어올릴 것:

1. profiling
2. temperature
3. top-k
4. top-p
5. batch/prefill-step-size
6. seed
7. context/max-seq-len
8. sampler backend

권장 CLI:

1. `--model-type`
2. `--profiling off|summary|trace`
3. `--profile-output <path>`
4. `--temperature`
5. `--top-k`
6. `--top-p`
7. `--batch-size`
8. `--prefill-step-size`
9. `--ctx-size`
10. `--seed`

## Design Rules

### Shared

shared로 둘 것:

1. `MetalContext`
2. `MetalBuffer`
3. `BufferArena`
4. `TensorDesc`
5. `DeviceTensor`
6. `PipelineCache`
7. 모든 low-level `op`
8. `Sampler`
9. `KVCache` storage primitive
10. profiling collection infrastructure

### Model-specific

model-specific로 둘 것:

1. weight schema
2. module graph
3. prefill/decode execution order
4. planner / decode plan
5. loader / parameter binding
6. architecture-specific quantized path policy

## Immediate Implementation Scope

이번 리팩터링 시작 단계에서 실제로 할 일:

1. `gpu/models/` 디렉터리 추가
2. registry 코드 추가
3. `qwen3` registry/adapter scaffold 추가
4. `gpu_infer`를 registry 기반으로 바꾸기
5. build system이 `models/`를 컴파일하도록 바꾸기

이번 단계에서 아직 하지 않는 일:

1. `GenerationContext` generic runner 완전 전환
2. `CommandScheduler` generic planner 완전 전환
3. 기존 Qwen source file의 대규모 이동

그건 다음 단계 작업이다.
