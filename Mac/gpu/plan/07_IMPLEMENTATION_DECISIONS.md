# Implementation Decisions

## Confirmed Now

1. `Mac/gpu`는 Metal direct compute runtime으로 설계한다.
   이유: user requirement가 명시적으로 Metal 기반이며, GPU plan의 핵심 가치도 low-level control에 있다.

2. CPU plan 문서 구조를 그대로 따라간다.
   이유: 프로젝트 전체에서 architecture, layout, specs, optimization, roadmap, questions, decisions의 구성이 일관돼야 한다.

3. MPSGraph는 Phase 1의 기본 경로가 아니다.
   이유: 구조를 직접 제어하기 어렵고 backendless/low-level runtime 목표와 맞지 않는다.

4. optimization과 modularization을 동시에 설계한다.
   이유: GPU에서는 fused path, pipeline cache, KV layout이 구조와 분리되지 않는다.

5. `LLM_interpreter` export contract를 우선 재사용한다.
   이유: tokenizer/runtime asset, manifest, weight naming 계약을 이미 CPU path에서 정리했기 때문이다.

6. prefill path와 decode path는 초기에 분리한다.
   이유: throughput/latency 목표가 다르고 kernel policy도 다르다.

7. test와 debug는 설계 초기부터 포함한다.
   이유: Metal path는 correctness regression을 늦게 잡으면 복구 비용이 커진다.

## Deferred Decisions

1. weight prepack의 정확한 physical layout
   이유: 실제 profiling 없이 고정하면 잘못된 방향일 수 있다.

2. paged KV cache 도입 여부
   이유: Phase 1 baseline에서는 과할 수 있다.

3. FlashAttention류 attention kernel 도입 시점
   이유: baseline attention과 decode latency 측정 후 판단하는 편이 낫다.

4. GPU sampling 전면 도입 여부
   이유: first-token latency와 readback 비용 측정이 먼저다.

5. multi-model / multi-device abstraction
   이유: 현재 범위를 넘는다.

## Immediate Build Direction

1. `Mac/gpu/plan` 문서 세트를 먼저 고정
2. Metal bootstrap skeleton 작성
3. buffer/tensor abstraction 작성
4. baseline kernel correctness path 작성

## Phase 0 Decisions

1. Phase 0 host 언어는 C++17 + Objective-C++ bridge 조합으로 간다.
   이유: public runtime surface는 C++로 유지하고, Metal API boundary만 `.mm` 파일로 제한하기 위해서다.

2. shader는 runtime compile이 아니라 build-time metallib로 묶는다.
   이유: regression 재현성과 compile failure surfacing이 더 명확하다.

3. Phase 0 bootstrap kernel은 single-dispatch correctness probe만 둔다.
   이유: device/library/pipeline/command-buffer 경로가 먼저 살아야 그 위에 matmul과 attention을 얹을 수 있다.

4. 첫 smoke test는 numerical sophistication보다 bootstrap contract를 검증한다.
   이유: 현재 필요한 것은 "Metal runtime이 실제로 올라오는가"에 대한 빠른 yes/no signal이다.

5. 초기 capability snapshot은 최소한 device name, unified memory, recommended working set, thread execution width를 포함한다.
   이유: 이후 kernel policy와 KV cache policy를 연결할 수 있는 최소 측정값이기 때문이다.