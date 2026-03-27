# Optimization And Modularization Plan

## 1. Modularization Is Mandatory

이 프로젝트에서는 모듈화와 최적화를 분리해서 생각하면 안 된다. 순수 C++ LLM runtime은 성능 문제가 곧 구조 문제로 이어지기 때문이다.

### 1.1 분리해야 하는 경계

반드시 분리할 경계:

1. IO vs Tensor core
2. Tensor core vs Kernel
3. Kernel vs Op
4. Op vs Module
5. Module vs Runtime session
6. Inference vs Training

이 분리가 필요한 이유:

1. kernel 교체와 최적화를 독립적으로 수행할 수 있다.
2. 특정 module만 unit test 가능하다.
3. training support 추가 시 inference path를 오염시키지 않는다.

## 2. Memory Strategy

backendless runtime에서 가장 중요한 최적화는 메모리다.

### 2.1 Weight Memory

전략:

1. weight는 가능한 mmap
2. immutable
3. module 초기화 시 prepack 가능
4. HF export 원본 tensor는 보존하고, packed/quantized variant는 secondary cache로 둔다

반드시 구현할 것:

1. `MappedStorage`
2. optional `PackedWeightStorage`
3. dtype-aware view 생성

### 2.2 Activation Memory

전략:

1. arena allocator
2. prefill와 decode scratch를 분리
3. layer-by-layer 재사용

필수 이유:

1. 작은 텐서를 heap new/delete로 계속 만들면 CPU inference가 급격히 느려진다.

### 2.3 KV Cache Memory

전략:

1. 연속 buffer 한 번 크게 할당
2. layer-major 또는 head-major layout 고정
3. decode write path branch 최소화

필수 최적화:

1. cache append 시 memcpy-friendly layout
2. attention read 시 contiguous segment 우선
3. public canonical layout과 optimized internal layout을 구분

## 3. Matmul Optimization Plan

### 3.1 Phase 1

1. contiguous row-major only
2. blocked matmul
3. rhs pretranspose/prepack
4. fp32 path
5. fp16 weight path with fp32 accumulate
6. current `header/matrix_ops.h`의 `PackMatMulRhs` / `MatMulPacked`를 새 runtime reference로 재사용 검토

### 3.2 Phase 2

1. thread parallelism
2. architecture-specific SIMD specialization
3. packed layout reuse across repeated linear layers

### 3.3 Phase 3

1. int8 or grouped quantized weight path
2. dequant fusion
3. attention projection fusion

## 4. Numerical Stability Rules

반드시 문서화하고 지켜야 하는 규칙:

1. softmax는 fp32 accumulate
2. RMSNorm variance/rms accumulate는 fp32
3. fp16 weight matmul도 accumulator는 최소 fp32
4. logits sampler 전 softmax or logit transform overflow guard 필요

## 5. Threading Plan

권장 순서:

1. matmul만 parallelize
2. prefill path parallelize
3. decode는 작은 batch라 과도한 thread spawn 금지

실행 원칙:

1. thread pool 또는 고정 worker 권장
2. layer 단위 병렬화보다 matmul 내부 병렬화 우선
3. false sharing 회피

## 6. Model-Specific Optimization

Qwen3-0.6B 기준으로 반드시 고려할 것:

1. GQA
2. RoPE
3. RMSNorm pre-norm 구조
4. gate/up/down MLP path
5. long context with KV cache

특히 GQA는 `n_heads != n_kv_heads`이므로 단순 multi-head attention 구현으로 끝나지 않는다.

## 7. Export-Side Optimization Requirements

`LLM_interpreter`도 최적화 전략의 일부다.

추가 필요 항목:

1. tokenizer custom export
2. weight layout hint export
3. packed weight cache optional pre-export
4. tensor grouping by module/layer
5. tied parameter metadata export

권장 manifest 확장:

1. `layout`
2. `module_group`
3. `layer_index`
4. `is_weight`
5. `preferred_runtime_dtype`
6. `tied_to`

## 8. Training-Ready Modularization

training으로 확장하기 위해 지금부터 분리해야 할 모듈:

1. forward-only kernel namespace
2. graph-aware op namespace
3. parameter storage and gradient storage 분리
4. inference session과 training session 분리

중요:

1. inference에 필요 없는 graph bookkeeping은 기본 경로에 넣지 않는다.
2. 대신 Tensor/Module 인터페이스는 확장 가능하게 둔다.

## 9. Optimization Checklist

반드시 plan에 포함해야 하는 최적화 항목 체크리스트:

1. weight mmap
2. rhs prepack
3. fp16/bf16 support
4. fp32 accumulation
5. KV cache contiguous layout
6. arena allocator for activation
7. thread parallelism for matmul
8. minimal tensor copy policy
9. model load-time validation
10. tokenizer fast-path for special tokens
11. batch=1 decode fast path
12. long-context feature flag kept off by default
