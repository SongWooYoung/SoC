# Qwen3 Implementation Roadmap

## Phase 0. Contract Freeze

목표:

1. export contract 재사용 가능 여부 확인
2. Metal runtime에서 필요한 추가 metadata 정의
3. baseline numerical target과 debug policy 정의

체크리스트:

1. `LLM_interpreter` export contract 검토
2. Qwen3 config 필수 필드 고정
3. tokenizer/chat template reuse 경로 결정
4. GPU prepack metadata를 runtime private로 둘지 확정

## Phase 1. Metal Bootstrap

목표:

1. device 생성
2. command queue 생성
3. shader library load
4. pipeline cache skeleton 작성

체크리스트:

1. `MetalContext` 구현
2. `PipelineCache` 구현
3. debug capture hook 추가
4. capability logging 추가

## Phase 2. Buffer And Tensor Baseline

목표:

1. `MetalBuffer`
2. `BufferArena`
3. `TensorDesc`
4. `DeviceTensor`

체크리스트:

1. staging upload path 작성
2. private/shared buffer policy 확정
3. scratch arena reset policy 작성
4. basic tensor view smoke 작성

## Phase 3. Core Kernel Baseline

목표:

1. embedding
2. matmul
3. RMSNorm
4. RoPE
5. softmax

체크리스트:

1. CPU compare test 확보
2. fp16 input / fp32 accumulate 정책 구현
3. threadgroup sizing baseline 확보

## Phase 4. Module Baseline

목표:

1. `Linear`
2. `RMSNorm`
3. `QwenAttention`
4. `QwenMLP`
5. `QwenBlock`
6. `QwenCausalLM`

체크리스트:

1. loader와 parameter binding 연결
2. tied embedding/lm head 지원
3. q/k norm, GQA, RoPE 정확성 검증

## Phase 5. Prefill And Decode Runtime

목표:

1. prompt tokenize
2. prefill
3. single-step decode
4. KV cache append/reuse

체크리스트:

1. `GenerationContext`
2. `CommandScheduler`
3. `KVCache`
4. tokenizer/runtime glue
5. sampling readback path

## Phase 6. Real Bundle Bring-Up

목표:

1. real Qwen3 export bundle load
2. one-token generation
3. CPU reference와 token/logit smoke compare

체크리스트:

1. startup validation 추가
2. prompt tokenization match 확인
3. first-token generation smoke 확보

## Phase 7. Optimization Pass 1

목표:

1. decode latency 개선
2. KV cache path 개선
3. weight prepack 도입

체크리스트:

1. decode specialized matmul
2. residual + RMSNorm fusion
3. MLP gate fusion
4. pipeline prewarm

## Phase 8. Optimization Pass 2

목표:

1. prefill throughput 개선
2. attention kernel 개선
3. command scheduling 다듬기

체크리스트:

1. prefill/decode 분기 최적화
2. attention softmax/value reduce path 측정
3. sync point 제거

## Phase 9. Regression And Packaging

목표:

1. deterministic smoke
2. debug path와 fast path 분리
3. build/test target 구조 정리

체크리스트:

1. unit test
2. module test
3. integration test
4. real bundle golden smoke