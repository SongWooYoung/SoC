# Module Specs

## 1. `MetalContext`

책임:

1. `MTLDevice`, `MTLCommandQueue` 소유
2. feature query
3. default library / pipeline cache 초기화
4. debug capture control

필수 API:

1. `CreateDefault()`
2. `SupportsSIMDGroupMatrix()` 또는 동등 capability query
3. `CreateCommandBuffer()`
4. `GetPipelineCache()`

## 2. `MetalBuffer`

책임:

1. `MTLBuffer` 래핑
2. size, offset, storage mode metadata
3. upload / map / readback helper

규칙:

1. host-visible 여부를 명시적으로 보관한다.
2. private buffer readback은 staging path를 거쳐야 한다.

## 3. `BufferArena`

종류:

1. weight arena
2. activation arena
3. KV cache arena
4. staging arena

책임:

1. suballocation
2. alignment 보장
3. resettable scratch lifetime 관리

## 4. `TensorDesc`

필드:

1. dtype
2. shape
3. stride
4. contiguous flag
5. logical layout tag

중요:

1. tensor descriptor는 buffer ownership을 가지지 않는다.
2. layout conversion과 fused dispatch 판단은 이 descriptor 기반으로 이루어진다.

## 5. `DeviceTensor`

필드:

1. `MetalBuffer*`
2. byte offset
3. `TensorDesc`
4. residency/device tag

필수 이유:

1. 동일 buffer 위의 view를 cheap 하게 표현해야 한다.
2. packed weight와 logical tensor를 분리해야 한다.

## 6. `KernelKey`

필드 예시:

1. op kind
2. input dtype
3. output dtype
4. tile size
5. head dim
6. causal flag
7. decode/prefill mode

역할:

1. pipeline specialization lookup
2. prewarm cache key

## 7. `PipelineCache`

책임:

1. `KernelKey -> MTLComputePipelineState`
2. lazy build
3. optional startup prewarm
4. shader compile error surface

## 8. `CommandScheduler`

책임:

1. prefill command sequence encode
2. decode command sequence encode
3. barrier / synchronization 최소화
4. temporary buffer recycling 시점 결정

중요:

1. prefill은 throughput 우선
2. decode는 latency 우선
3. 같은 op라도 mode에 따라 다른 kernel policy를 허용해야 한다.

## 9. `WeightUploader`

책임:

1. exported `.bin` 읽기
2. Metal buffer upload
3. optional prepack 수행
4. tied weight aliasing 처리

중요:

1. CPU export format과 GPU packed format은 분리한다.
2. packed copy는 runtime startup 한 번만 수행해야 한다.

## 10. `KVCache`

필드:

1. layer count
2. kv head count
3. head dim
4. max sequence length
5. packed/unpacked layout metadata

필수 API:

1. `Reserve()`
2. `AppendPrefill()`
3. `AppendDecodeToken()`
4. `ViewForLayer()`

## 11. Core Ops

필수 op:

1. embedding lookup
2. matmul
3. fused bias/add
4. RMSNorm
5. RoPE apply
6. attention score + mask + softmax + value reduce
7. SiLU + gate
8. logits projection
9. sampling/top-k

우선순위:

1. baseline correctness op
2. fused fast path
3. specialized decode path

## 12. Core Modules

### 12.1 `Embedding`

1. CPU upload된 token ids를 GPU lookup input으로 전달
2. tied embedding/lm head를 지원

### 12.2 `Linear`

1. weight packing optional
2. fp16 input / fp32 accumulate 기본 정책
3. small batch decode path specialization 필요

### 12.3 `RMSNorm`

1. standalone kernel
2. residual-add fused variant 필요

### 12.4 `QwenAttention`

1. prefill path와 decode path 분리
2. GQA head sharing 지원
3. RoPE, q/k norm, KV cache append를 포함

### 12.5 `QwenMLP`

1. gate/up matmul
2. SiLU-gate fused path
3. down projection

### 12.6 `QwenBlock`

1. attention residual branch
2. MLP residual branch
3. norm placement 정확성 유지

### 12.7 `QwenCausalLM`

1. block stack
2. final norm
3. lm head projection

## 13. Debug Modules

필수 debug 모듈:

1. tensor readback compare
2. per-op max error report
3. command timing instrumentation
4. optional Metal capture trigger