# SoC Pure C++ LLM Plan

## Mission

이 프로젝트의 최종 목적은 다음 두 가지를 동시에 만족하는 것이다.

1. 기존 `ggml`, `llama.cpp`, `oneDNN`, `Eigen`, `xnnpack` 같은 외부 텐서 백엔드 없이, 순수 C++와 직접적인 메모리 조작으로 CPU에서 LLM inference를 수행한다.
2. 추후 training까지 확장 가능한 구조를 처음부터 설계한다. 즉, 지금은 inference-first이지만 나중에 autograd, optimizer, checkpoint write-back, dataset pipeline을 얹을 수 있어야 한다.

이 계획 문서는 현재 `Mac/cpu` 코드와 `LLM_interpreter`의 Hugging Face export 포맷을 기준으로, 반드시 필요한 런타임 계층, 모듈화 전략, 최적화 방향, 디렉토리 구조, 단계별 구현 순서를 정리한다.

이번 개정본부터는 추측성 설계를 줄이기 위해, 문서 안의 중요한 결정마다 공개 구현과 공식 문서를 근거로 둔다. 특히 다음 자료를 설계 기준선으로 삼는다.

1. Qwen3 공식 Hugging Face model card / `config.json` / `tokenizer_config.json`
2. Qwen 공식 문서의 Transformers, llama.cpp, long-context, thinking mode 설명
3. PyTorch internals와 ATen/C10의 `Storage`-`Tensor`-`view` 분리 구조
4. `llm.c`의 pure C reference가 보여주는 contiguous activation/parameter 메모리 운영 방식
5. `ggml`과 `FasterTransformer`의 allocator, KV cache, packed layout 패턴

## Core Principles

1. Backendless runtime
   순수 C++ 표준 라이브러리와 직접 작성한 커널만 사용한다.

2. Inference first, training-ready architecture
   1차 목표는 Qwen3-0.6B CPU inference end-to-end 성공이다. 그러나 Tensor/Op/Module/Storage 설계는 gradient와 optimizer가 들어갈 수 있도록 확장 가능해야 한다.

3. Hard modularization
   `storage`, `tensor`, `kernel`, `op`, `nn module`, `model`, `io`, `tokenizer`, `runtime`, `train`을 분리한다.

4. Optimization is not optional
   모듈화와 동시에 메모리 레이아웃, weight prepack, fp16/fp32 mixed accumulate, KV cache layout, threading, allocator 전략을 설계에 포함한다.

5. Runtime consumes exported assets, never guesses
   `LLM_interpreter`가 생성한 `manifest.json`, `weights/*.bin`, `tokenizer/*`를 기준으로 동작하고, 파일명이나 weight naming을 추측하지 않는다.

## Planning Documents

1. [01_ARCHITECTURE.md](01_ARCHITECTURE.md)
   전체 런타임 계층 구조와 핵심 설계 원칙

2. [02_DIRECTORY_LAYOUT.md](02_DIRECTORY_LAYOUT.md)
   실제 권장 폴더 구조와 파일별 책임

3. [03_MODULE_SPECS.md](03_MODULE_SPECS.md)
   Tensor, Storage, Op, Tokenizer, WeightLoader, Attention, KV cache 등 세부 모듈 사양

4. [04_OPTIMIZATION_AND_MODULARIZATION.md](04_OPTIMIZATION_AND_MODULARIZATION.md)
   모듈화 전략과 반드시 포함해야 하는 최적화 계획

5. [05_QWEN3_IMPLEMENTATION_ROADMAP.md](05_QWEN3_IMPLEMENTATION_ROADMAP.md)
   Qwen3-0.6B를 실제 타겟으로 잡은 단계별 구현 로드맵과 체크리스트

6. [06_RESEARCH_QUESTIONS.md](06_RESEARCH_QUESTIONS.md)
   plan을 읽으며 생긴 핵심 의문, 조사 출처, 답변, 설계 반영점

7. [07_IMPLEMENTATION_DECISIONS.md](07_IMPLEMENTATION_DECISIONS.md)
   현재 시점에서 확정한 구현 결정과 defer한 결정

## Immediate Direction

첫 타겟은 `Qwen/Qwen3-0.6B`다. 구현은 아래 순서를 따른다.

1. `LLM_interpreter` export 포맷 고정
2. C++ manifest/weight/tokenizer loader 작성
3. backendless Tensor/Storage/Kernel/Op 계층 작성
4. `Linear`, `RMSNorm`, `RoPE`, `Attention`, `MLP`, `TransformerBlock`, `CausalLM` 작성
5. KV cache + token-by-token generation 경로 완성
6. fp16 weight + fp32 accumulate + packed matmul + threading 적용
7. 이후 training-ready graph/autograd 계층 확장

단, 이제부터는 위 순서를 그대로 구현하지 않는다. 먼저 질문을 정리하고 답을 고정한 뒤 구현 순서를 따른다.

1. Qwen3-0.6B에서 실제로 필요한 config/tokenizer/chat-template 사실 확인
2. baseline runtime이 감당할 범위와 최적화 전용 layout을 분리
3. export contract를 확정
4. 그 다음에 loader, tensor, kernels, modules, runtime 순으로 구현

## Non-Goals For Phase 1

1. CUDA / Metal / Vulkan
2. 분산 학습
3. 범용 NumPy 수준 broadcasting 전체 구현
4. 모든 Hugging Face 모델 자동 지원
5. optimizer까지 즉시 구현

Phase 1의 성공 기준은 다음이다.

1. Qwen3-0.6B export를 읽는다.
2. prompt를 tokenize한다.
3. prefill + decode with KV cache가 동작한다.
4. CPU 상에서 반복 생성이 가능하다.
5. output projection, GQA, RMSNorm, RoPE가 config 기반으로 맞게 적용된다.
6. thinking / non-thinking template 선택이 C++ runtime에서 재현된다.
7. tied embedding/lm head 계약이 export/runtime 양쪽에서 일관된다.
