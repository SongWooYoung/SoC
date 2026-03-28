# Research Questions

## 1. Weight Buffer는 `private`가 기본이어야 하는가?

왜 중요한가:

1. read bandwidth와 upload simplicity가 trade-off다.
2. Apple GPU에서 `shared`가 충분한지, `private`가 실질적 이득이 있는지 확인해야 한다.

설계 반영점:

1. `WeightUploader`가 staging -> private copy path를 가질 수 있어야 한다.

## 2. Qwen3-0.6B decode에서 실제 병목은 matmul인가, KV cache read인가?

왜 중요한가:

1. 최적화 우선순위를 결정한다.
2. decode latency 개선 target을 정해야 한다.

설계 반영점:

1. profiling marker와 per-op timing이 early phase부터 필요하다.

## 3. Apple GPU에서 FlashAttention 스타일 구현이 Phase 1에 과한가?

왜 중요한가:

1. 구현 난이도와 유지보수 비용이 높다.
2. baseline attention과 fused attention의 경계를 정해야 한다.

설계 반영점:

1. attention op를 baseline/fused 두 계층으로 분리한다.

## 4. KV cache는 paged layout이 필요한가?

왜 중요한가:

1. 긴 context와 memory fragmentation 문제에 직접 연결된다.
2. 하지만 Phase 1에서는 complexity가 너무 커질 수 있다.

설계 반영점:

1. public API는 logical view만 노출하고 backend layout은 숨긴다.

## 5. sampling은 CPU readback으로 충분한가?

왜 중요한가:

1. small logits readback은 단순하지만 latency bottleneck일 수 있다.
2. top-k나 argmax를 GPU에서 미리 줄이는 편이 더 나을 수 있다.

설계 반영점:

1. sampling kernel과 CPU fallback 둘 다 열어 둔다.

## 6. Metal function constant specialization을 어느 수준까지 쓸 것인가?

왜 중요한가:

1. head dim, tile size, causal mode를 specialization하면 빠를 수 있다.
2. 하지만 pipeline state 수가 폭증할 수 있다.

설계 반영점:

1. `KernelKey`와 `PipelineCache`를 early phase에 먼저 고정한다.

## 7. CPU reference compare는 어느 granularity까지 필요할까?

왜 중요한가:

1. GPU debugging은 오차가 생겼을 때 원인 위치를 찾기 어렵다.
2. op-level compare 없이는 fused path 디버깅이 힘들다.

설계 반영점:

1. debug layer에 per-op tensor compare를 넣는다.

## 8. Apple GPU에서 threadgroup memory pressure가 attention kernel 선택을 얼마나 바꾸는가?

왜 중요한가:

1. Apple GPU는 tile memory와 occupancy trade-off가 강하게 나타날 수 있다.
2. same algorithm이라도 threadgroup memory 사용량에 따라 decode latency가 크게 달라질 수 있다.

설계 반영점:

1. attention baseline/fused path 모두 threadgroup memory 사용량을 기록해야 한다.

## 9. SIMD-group matrix 경로를 어떤 GPU family부터 열어야 하는가?

왜 중요한가:

1. family gate를 잘못 잡으면 pipeline count만 늘고 실제 이득이 없을 수 있다.
2. Phase 1에서 지원 범위를 너무 넓히면 디버깅 비용이 급격히 커진다.

설계 반영점:

1. `MetalContext`가 family/capability snapshot을 명시적으로 노출해야 한다.

## 10. `recommendedMaxWorkingSetSize`를 기준으로 KV cache upper bound를 동적으로 잡아야 하는가?

왜 중요한가:

1. Apple Silicon은 physical VRAM 대신 unified memory pressure 관리가 중요하다.
2. 모델 로드 직후와 긴 context decode 중의 메모리 한계가 다르게 느껴질 수 있다.

설계 반영점:

1. bootstrap 단계부터 device working-set 정보를 수집한다.

## 11. shader compile 전략은 startup compile인가, build-time metallib 고정인가?

왜 중요한가:

1. startup compile은 iteration은 편하지만 reproducibility와 launch latency가 불리하다.
2. build-time metallib는 배포와 regression에 유리하지만 shader variant 관리가 더 중요해진다.

설계 반영점:

1. Phase 0은 build-time metallib를 기본으로 잡고, runtime은 path만 주입받도록 한다.

## 12. prefill과 decode의 command buffer 경계를 어느 시점에 나눠야 하는가?

왜 중요한가:

1. command buffer를 너무 잘게 쪼개면 latency가 올라간다.
2. 너무 크게 묶으면 profiling과 error isolation이 어려워진다.

설계 반영점:

1. early runtime에서 per-stage timing과 command count를 바로 측정해야 한다.