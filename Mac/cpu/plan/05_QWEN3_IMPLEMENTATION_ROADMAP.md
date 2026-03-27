# Qwen3-0.6B Implementation Roadmap

## 1. Target Fixation

첫 타겟 모델은 반드시 `Qwen/Qwen3-0.6B`로 고정한다.

이 모델에서 지금 확정된 핵심 스펙은 다음이다.

1. 28 layers
2. hidden size 1024
3. intermediate size 3072
4. 16 Q heads / 8 KV heads
5. head dim 128
6. RMSNorm epsilon 1e-6
7. rope theta 1000000
8. tied embeddings
9. checkpoint dtype bf16
10. baseline context 32768

이 결정으로 얻는 이점:

1. config 필드와 module shape를 먼저 고정할 수 있다.
2. tokenizer와 weight naming을 모델별 범용화하기 전에 실제 성공 경로를 만든다.
3. GQA, RMSNorm, RoPE, SwiGLU 계열 attention stack을 정확히 맞출 수 있다.

## 2. Definition of Done For Phase 1

다음이 모두 만족되면 Phase 1 완료다.

1. `LLM_interpreter` export로 생성된 Qwen3-0.6B 자산을 C++ 런타임이 읽는다.
2. tokenizer가 prompt를 token id로 변환한다.
3. prefill forward가 동작한다.
4. decode loop with KV cache가 동작한다.
5. sampling 후 detokenize가 동작한다.
6. 짧은 프롬프트에 대해 의미 있는 텍스트가 생성된다.

## 3. Phase Breakdown

### Phase 0. Export Contract Freeze

작업:

1. manifest 스키마 문서화
2. tensor record validation rule 고정
3. tokenizer export 전략 결정
4. Qwen3 config 필수 필드 목록 확정
5. thinking / non-thinking template metadata export 결정
6. tied embedding contract 명시

완료 기준:

1. C++ runtime이 참조할 export contract가 고정됨

### Phase 1. Asset Loading

작업:

1. JSON parser 도입 또는 lightweight parser 작성
2. `ManifestLoader`
3. `WeightLoader`
4. `TokenizerLoader`
5. startup validation
6. `Qwen3TemplateBuilder`

완료 기준:

1. 임의의 tensor name으로 weight 로드 가능
2. config 추출 가능
3. tokenizer encode/decode smoke test 통과
4. thinking / non-thinking prompt serialization 가능

### Phase 2. Tensor Core

작업:

1. DType
2. Shape/Stride
3. Storage
4. Tensor
5. allocator

완료 기준:

1. 2D matmul/rmsnorm/softmax가 tensor 기반으로 동작

### Phase 3. Kernel And Op Layer

작업:

1. matmul family
2. embedding lookup
3. rmsnorm
4. silu
5. softmax
6. rope

완료 기준:

1. unit test 통과
2. fp32 path와 fp16 mixed path 동작

### Phase 4. Qwen3 Module Graph

작업:

1. `Embedding`
2. `Linear`
3. `RMSNorm`
4. `QwenAttention`
5. `QwenMLP`
6. `QwenBlock`
7. `QwenCausalLM`
8. weight mapping table

완료 기준:

1. single layer forward 가능
2. full model logits 계산 가능

### Phase 5. Prefill And Decode Runtime

작업:

1. `GenerationContext`
2. `KVCache`
3. prefill path
4. decode path
5. sampler
6. detokenize

완료 기준:

1. prompt -> generated text end-to-end

### Phase 6. Optimization Pass

작업:

1. rhs prepack
2. fp16 weight default
3. matmul threading
4. KV cache layout tuning
5. arena scratch allocator
6. optional fused QKV load-time cache

완료 기준:

1. “쓸만한 속도” 수준 확보

주의:

1. YaRN long context는 이 단계의 기본 완료 기준이 아니다.
2. packed key cache public ABI 변경도 이 단계 기본 완료 기준이 아니다.

## 4. Tokenizer Decision

질문 항목에 대한 권장 답:

1. 첫 타겟 모델은 `Qwen/Qwen3-0.6B`로 고정한다.
2. tokenizer는 초기에 C++가 직접 `tokenizer.json` 전체를 해석하기보다, exporter가 단순화한 tokenizer runtime format을 추가로 만든다.
3. 성능 목표는 fp16 weight + fp32 accumulate + rhs prepack + matmul 병렬화로 잡는다.

이 선택이 현실적인 이유:

1. tokenizer 전체 스펙 구현보다 custom export가 빠르다.
2. 단일 타겟 모델에서 빠르게 inference 성공을 얻을 수 있다.
3. 이후 모델 확장은 exporter를 늘리는 방식으로 처리할 수 있다.

## 5. Detailed Build Checklist

### 5.1 Export Side

1. `manifest.json` validation tool
2. tokenizer runtime export 추가
3. Qwen3 config sanity check 추가

### 5.2 CPU Runtime Side

1. dtype enum
2. storage abstraction
3. tensor abstraction
4. JSON manifest loader
5. weight loader with mmap
6. tokenizer loader
7. embedding kernel
8. matmul kernel
9. rmsnorm kernel
10. silu kernel
11. rope kernel
12. softmax kernel
13. linear module
14. attention module with GQA
15. MLP module
16. transformer block
17. causal LM wrapper
18. KV cache
19. sampler
20. generation loop

### 5.3 Validation

1. tensor load test
2. tokenizer roundtrip test
3. single op parity test
4. single block parity test
5. full forward logits sanity test
6. decode cache correctness test
7. short generation smoke test

## 6. Training Extension Plan

training은 inference가 먼저 완성된 다음 확장한다.

순서:

1. `Tensor`에 optional grad slot 추가
2. op trace framework 추가
3. backward kernel 추가
4. loss and optimizer 추가
5. checkpoint writer 추가

그러나 이 training 계획은 inference runtime의 구조를 흔들지 않아야 한다. 따라서 training은 반드시 `train/` 계층에 격리한다.

## 7. Final Recommendation

가장 먼저 착수해야 할 실제 구현 순서는 다음이다.

1. `LLM_interpreter` exporter에 tokenizer runtime export와 template metadata export 추가
2. `ManifestLoader` + `WeightLoader` + `TokenizerLoader` + `Qwen3TemplateBuilder`
3. 새 `Tensor`/`Storage` 계층 생성
4. `matmul`, `embedding`, `rmsnorm`, `rope`, `softmax`, `silu`
5. tied embedding/lm_head 포함 `QwenAttention`, `QwenMLP`, `QwenBlock`, `QwenCausalLM`
6. canonical `KVCache` + prefill/decode runtime
7. packed/fp16/threading optimization
