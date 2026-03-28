# Directory Layout Plan

## 1. Target Layout

권장 디렉터리 구조는 아래와 같다.

```text
Mac/gpu/
  plan/
  include/
    asset/
    metal/
    buffer/
    tensor/
    kernel/
    op/
    module/
    model/
    runtime/
    debug/
  src/
    asset/
    metal/
    buffer/
    tensor/
    kernel/
    op/
    module/
    model/
    runtime/
    debug/
  shaders/
    common/
    matmul/
    norm/
    attention/
    elementwise/
    sampling/
  test/
    asset/
    metal/
    kernel/
    module/
    runtime/
  tools/
  build/
```

## 2. Folder Responsibilities

### 2.1 `include/asset`, `src/asset`

책임:

1. manifest parsing
2. tokenizer/runtime asset loading
3. model metadata 검증

### 2.2 `include/metal`, `src/metal`

책임:

1. `MetalContext`
2. device capability query
3. pipeline library loading
4. command queue / command buffer helper
5. capture / validation hook

### 2.3 `include/buffer`, `src/buffer`

책임:

1. `MetalBuffer`
2. upload/download staging
3. scratch arena
4. activation arena
5. KV cache arena

### 2.4 `include/tensor`, `src/tensor`

책임:

1. tensor descriptor
2. shape/stride helper
3. device tensor view
4. layout conversion helper

### 2.5 `include/kernel`, `src/kernel`

책임:

1. kernel registry
2. specialization key
3. pipeline cache
4. encoder helper

이 폴더는 shader source를 직접 담지 않는다. shader source는 `shaders/`에 둔다.

### 2.6 `include/op`, `src/op`

책임:

1. op-level shape validation
2. kernel dispatch
3. temporary buffer planning
4. fused op boundary 관리

### 2.7 `include/module`, `src/module`

책임:

1. embedding
2. linear
3. RMSNorm
4. attention
5. MLP

### 2.8 `include/model`, `src/model`

책임:

1. Qwen block
2. Qwen causal LM
3. model loader / parameter binding

### 2.9 `include/runtime`, `src/runtime`

책임:

1. tokenizer/runtime glue
2. prefill/decode scheduler
3. sampling
4. end-to-end generation path

### 2.10 `include/debug`, `src/debug`

책임:

1. CPU reference compare
2. tensor dump
3. timing/profiling marker
4. Metal capture support

### 2.11 `shaders/`

책임:

1. `.metal` compute kernels
2. kernel family별 파일 분리
3. reusable threadgroup helper macro 관리

예시:

1. `shaders/matmul/matmul_fp16_fp32acc.metal`
2. `shaders/norm/rmsnorm.metal`
3. `shaders/attention/attention_prefill.metal`
4. `shaders/attention/attention_decode.metal`
5. `shaders/attention/rope.metal`

### 2.12 `test/`

책임:

1. asset parse test
2. device bootstrap test
3. kernel correctness test
4. module smoke test
5. runtime integration test

## 3. Naming Rules

1. API header는 noun 중심 이름을 쓴다.
2. runtime entry는 `*_runtime`, `*_scheduler`, `*_context`처럼 책임이 드러나야 한다.
3. shader는 dtype, tile, 역할이 이름에 드러나야 한다.
4. fused path는 `fused_` prefix를 붙인다.

## 4. What Must Not Happen

1. Metal API 호출이 model/module 곳곳에 흩어지면 안 된다.
2. shader source와 host dispatch logic가 같은 파일에 섞이면 안 된다.
3. `private` / `shared` storage mode policy가 call-site마다 제각각이면 안 된다.
4. debug utility가 hot path dependency가 되면 안 된다.