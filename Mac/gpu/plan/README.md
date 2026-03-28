# SoC Metal LLM Plan

빠른 실행 경로와 현재 검증된 명령은 workspace 루트의 [README.md](/Users/song-ganghui/Documents/SoC/README.md)에서 먼저 보는 편이 낫다. 이 문서는 `Mac/gpu`를 Metal 기반 LLM runtime으로 키우기 위한 설계 계획과 구현 방향을 다룬다.

현재 `Mac/gpu`는 더 이상 placeholder 단계는 아니다. 이미 Metal bootstrap runtime, fallback shader loading, build/test scaffold, `MetalContext`, `MetalBuffer`, `BufferArena`, `TensorDesc`, `DeviceTensor`, `PipelineCache`, GPU asset loader, baseline `EmbeddingOp`, `RmsNormOp`, `RoPEOp`, `SoftmaxOp`, `MatMulOp`, `LinearOp`, `QwenAttention`, `QwenMLP`, `QwenBlock`, baseline `KVCache`, prefill/decode attention split, `QwenCausalLM`, `GenerationContext`, command scheduler, sampling, 그리고 real-bundle regression/benchmark path까지 올라와 있다. `test_metal_bootstrap`, `test_rms_norm`, `test_linear`, `test_asset_loader`, `test_phase3_kernels`, `test_qwen_modules`, `test_generation_context`, `test_real_bundle_regression`이 동작한다.

즉, 이 문서 세트는 더 이상 빈 구조를 세우는 계획만이 아니라, 이미 올라온 baseline 위에 module, runtime, model bring-up과 performance validation을 정리하는 계획이다.

## Mission

이 프로젝트의 최종 목적은 다음 두 가지를 동시에 만족하는 것이다.

1. Apple Silicon의 GPU를 사용해 Metal 기반으로 LLM inference를 수행한다.
2. 추후 training 또는 최소한 backward-capable graph로 확장 가능한 구조를 처음부터 설계한다.

이 계획 문서는 `Mac/cpu/plan`의 구조를 참고하되, GPU라는 환경 차이를 반영해 다음 문제를 먼저 해결 대상으로 둔다.

1. host-device 메모리 이동 최소화
2. Metal command scheduling과 kernel specialization
3. KV cache, attention, matmul, normalization의 fused path 설계
4. weight layout prepack과 pipeline state cache 전략
5. CPU fallback과 Metal 전용 최적화의 경계 분리

이번 개정본부터는 추측성 설계를 줄이기 위해, 중요한 결정마다 공개 구현과 공식 문서를 설계 기준선으로 둔다. 특히 다음 자료를 기준선으로 삼는다.

1. Apple Metal Shading Language spec, Metal Feature Set Tables, Apple GPU optimization guidance
2. Qwen3 공식 Hugging Face model card / `config.json` / `tokenizer_config.json`
3. llama.cpp Metal backend, MLX, MPSGraph, FasterTransformer류의 GPU runtime 패턴
4. PyTorch ATen/C10의 tensor-storage-view 분리와 device abstraction 패턴
5. FlashAttention, paged KV cache, fused RMSNorm/RoPE 계열의 공개 구현 아이디어

## Core Principles

1. Metal-first runtime
   `Mac/gpu`는 Metal을 기본 실행 백엔드로 삼는다. CPU는 reference path일 뿐이며, GPU plan의 중심이 아니다.

2. Inference first, graph-ready architecture
   1차 목표는 Qwen3-0.6B Metal inference end-to-end 성공이다. 그러나 tensor, buffer, command graph, op dispatch 구조는 이후 backward 또는 training extension을 막지 않아야 한다.

3. Hard modularization
   `asset`, `host_runtime`, `metal_device`, `buffer`, `tensor`, `kernel`, `op`, `module`, `model`, `scheduler`, `runtime`, `debug`, `train_future`를 분리한다.

4. Optimization is mandatory
   GPU runtime는 correctness만으로 가치가 없다. weight packing, command batching, kernel fusion, shared memory/threadgroup 활용, persistent cache, async overlap까지 구조에 반영해야 한다.

5. Runtime consumes exported assets, never guesses
   `LLM_interpreter`가 생성한 `manifest.json`, `weights/*.bin`, `tokenizer/*`를 기준으로 동작하고, weight naming이나 config를 추측하지 않는다.

## Planning Documents

1. [01_ARCHITECTURE.md](01_ARCHITECTURE.md)
   Metal runtime의 전체 계층 구조와 핵심 설계 원칙

2. [02_DIRECTORY_LAYOUT.md](02_DIRECTORY_LAYOUT.md)
   실제 권장 폴더 구조와 파일별 책임

3. [03_MODULE_SPECS.md](03_MODULE_SPECS.md)
   device, buffer, tensor, command encoder, kernel registry, attention, KV cache 등 세부 모듈 사양

4. [04_OPTIMIZATION_AND_MODULARIZATION.md](04_OPTIMIZATION_AND_MODULARIZATION.md)
   Metal에서 반드시 필요한 최적화 축과 기능별 모듈화 전략

5. [05_QWEN3_IMPLEMENTATION_ROADMAP.md](05_QWEN3_IMPLEMENTATION_ROADMAP.md)
   Qwen3-0.6B를 실제 타겟으로 잡은 단계별 구현 로드맵과 체크리스트

6. [06_RESEARCH_QUESTIONS.md](06_RESEARCH_QUESTIONS.md)
   plan을 읽으며 생긴 핵심 의문, 조사 포인트, 설계 반영점

7. [07_IMPLEMENTATION_DECISIONS.md](07_IMPLEMENTATION_DECISIONS.md)
   현재 시점에서 확정한 구현 결정과 defer한 결정

8. [08_REAL_BUNDLE_BASELINE.md](08_REAL_BUNDLE_BASELINE.md)
   실측 CPU:GPU multi-token regression baseline과 해석

9. [09_TEST_RESULTS.md](09_TEST_RESULTS.md)
   현재 저장된 test result 위치와 후속 작업 정리

## Current Test Result Paths

현재 real-bundle regression 결과는 두 군데에 남긴다.

1. build artifact report: `Mac/gpu/build/reports/test_real_bundle_regression_report.md`
2. repo-stable summary: `Mac/gpu/plan/09_TEST_RESULTS.md`

첫 번째는 test를 다시 돌릴 때 갱신되는 실행 결과이고, 두 번째는 현재 확인된 baseline과 남은 작업을 repo 안에 고정해 두는 문서다.

## Immediate Direction

첫 타겟은 `Qwen/Qwen3-0.6B`다. 구현은 아래 순서를 따른다.

1. `LLM_interpreter` export contract 재사용
2. Metal device/bootstrap layer 작성
3. GPU buffer/tensor/view abstraction 작성
4. matmul, RMSNorm, RoPE, embedding, sampling의 baseline Metal kernel 작성
5. Qwen attention / MLP / block / causal LM graph 작성
6. prefill + decode + KV cache path 작성
7. fused kernels, weight packing, async scheduling으로 성능 개선

단, CPU와 마찬가지로 이 순서를 바로 구현하지는 않는다. 먼저 질문을 정리하고 답을 고정한 뒤 구현 순서를 따른다.

1. Apple GPU에서 실질적으로 병목이 어디인지 가설을 세운다.
2. baseline runtime과 optimized runtime의 경계를 분리한다.
3. export contract와 GPU 전용 prepack contract를 분리한다.
4. 그 다음에 이미 올라온 device, buffer, kernel baseline 위에 module, scheduler, runtime 순으로 구현한다.

## Non-Goals For Phase 1

1. CUDA / Vulkan / DirectML 동시 지원
2. MPSGraph 기반 고수준 graph runtime 의존
3. 다중 GPU 분산 추상화
4. 범용 training 즉시 구현
5. 모든 Hugging Face 모델 자동 지원

Phase 1의 성공 기준은 다음이다.

1. Qwen3-0.6B export를 읽는다.
2. prompt를 tokenize한다.
3. prefill + decode with KV cache가 Metal 상에서 동작한다.
4. CPU reference 대비 numerically sane 한 logits를 낸다.
5. output projection, GQA, RMSNorm, RoPE가 config 기반으로 맞게 적용된다.
6. command buffer 구성과 weight/buffer lifetime이 안정적으로 관리된다.
7. fused optimization을 얹을 수 있는 모듈 경계가 유지된다.