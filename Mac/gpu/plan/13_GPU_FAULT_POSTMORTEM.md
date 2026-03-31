# 13 GPU Fault Postmortem

## Summary

Mac mini M4 32GB에서 `WindowServer` 오류창과 화면 깨짐을 동반한 GPU fault가 발생했다. 현재까지의 비교에서 가장 강하게 상관된 변경은 `8197a2d3` 이후 도입된 `CommandStream` 기반의 `forward-pass-wide` command-buffer batching이다.

핵심 결론은 다음이다.

1. command batching 자체가 잘못된 것은 아니다.
2. 하지만 이 머신/드라이버 조합에서는 `한 decode step 전체` 또는 `여러 레이어 범위 전체`를 하나의 giant command buffer로 묶는 방식이 unsafe하다.
3. 안정적으로 복구된 baseline은 `per-op command buffer` 경계다.
4. 다음 최적화는 `full-range batching`이 아니라 `bounded batching`으로만 진행해야 한다.

## Known-Good vs Fault-Prone Range

- known-good benchmark baseline commit: `8197a2d3`
- fault-prone working tree base observed by user: `85e798ee`
- stable recovery after fix: `SOC_GPU_ENABLE_EXPERIMENTAL_COMMAND_STREAM` 기본값 `off`

`8197a2d3`의 full GPU baseline은 [12_LAYER_SPLIT_BENCHMARK.md](12_LAYER_SPLIT_BENCHMARK.md)에 기록된 `1.794 tok/s`다. 느리지만 `WindowServer` 수준의 fault는 재현되지 않았다.

## Reproduction Conditions

다음 조건 조합이 fault-prone으로 분류된다.

1. `CommandStream`으로 여러 op를 하나의 `MTLCommandBuffer`에 누적
2. 그 범위가 `single op`나 `single block`이 아니라 `block range` 또는 `full decode step`까지 커짐
3. attention, KV blit, matmul, normalization이 같은 giant buffer 안에 함께 인코드됨

이 패턴은 Mac mini M4 32GB에서 다음 현상과 함께 관찰됐다.

1. 화면 전체 그래픽 뒤틀림
2. `WindowServer` 관련 오류 팝업
3. inference 프로세스 정상 실패가 아니라 UI compositor까지 영향을 주는 GPU-level fault

## What This Does Not Mean

다음 해석은 틀렸다.

1. "Metal command batching은 원천적으로 잘못됐다"
2. "float16 자체가 항상 위험하다"
3. "optimization을 하면 안 된다"

현재까지 맞는 해석은 이것이다.

1. `full-range batching`은 unsafe했다.
2. `bounded batching`은 아직 금지 대상이 아니라, 측정과 실기기 검증을 거쳐 다시 시도할 수 있다.
3. mixed precision도 `real hardware` 실행 검증 없이 기본 경로에 넣으면 안 된다.

## Current Safety Policy

기본 경로는 다음 정책을 따른다.

1. `SOC_GPU_ENABLE_EXPERIMENTAL_COMMAND_STREAM` unset
   결과: per-op command buffer
2. `SOC_GPU_ENABLE_EXPERIMENTAL_COMMAND_STREAM=layer`
   결과: 한 레이어당 한 command buffer로 제한된 batching
3. `SOC_GPU_ENABLE_EXPERIMENTAL_COMMAND_STREAM=1|full|range`
   결과: full-range batching, 현재 M4에서 unsafe로 간주
4. `SOC_GPU_ENABLE_EXPERIMENTAL_F16_WEIGHTS` unset
   결과: fp16 export bundle이 있어도 기본 실행은 안전하게 fp32 storage로 승격

`full-range batching`과 `experimental f16 weights`는 둘 다 명시적 opt-in이어야 한다.

## Measured Impact

안정성 복구 후 실기기 비교:

- 초기 안정성 복구 상태: C++ full GPU 약 `3.0 tok/s`, PyTorch MPS fp16 약 `47.3 tok/s`
- GPU sampler를 CPU fallback으로 되돌린 뒤: C++ full GPU 약 `4.23 tok/s`, PyTorch MPS fp16 약 `43.6 tok/s`
- RMSNorm SIMD kernel 적용 뒤: C++ full GPU 약 `5.13 tok/s`, PyTorch MPS fp16 약 `54.9 tok/s`

즉 현재 엔진은 안정성은 복구했지만, PyTorch 대비 throughput이 크게 부족하다. 따라서 다음 작업의 우선순위는 다음과 같다.

1. per-op/per-scope GPU timing
2. decode-dominant MatMul weight prepack
3. Softmax reduction 개선
4. gated mixed precision 재검증

추가로 `SOC_GPU_ENABLE_EXPERIMENTAL_COMMAND_STREAM=layer` 실험은 `WindowServer` fault까지는 재현되지 않았지만, 8-token quick run에서도 baseline보다 현저히 오래 걸리며 종료되지 않아 현재는 `accepted optimization`이 아니라 `regression/hang candidate`로 분류한다.

2026-04-01 기준 추가 관찰:

- `q4 decode projections`로 decode matmul GPU 시간은 크게 줄였지만, full benchmark에서도 `command_buffer_count`는 여전히 `17120`이다.
- 즉 다음 scheduler 실험의 목표는 "더 큰 giant buffer"가 아니라 "엄격히 제한된 scope에서 submit/encode/wait overhead를 줄이는 것"이어야 한다.
- 앞으로 scheduler 실험은 `encoder cap`, `flush reason`, `scope boundary`를 같이 기록한다.

같은 날 추가로 확인된 사실:

- `real buffer-range hazard tracker`와 stage-local scratch arena를 도입한 것만으로는 fault가 재현되지 않았다.
- 하지만 그 위에 `SOC_GPU_ENABLE_EXPERIMENTAL_PREBUILT_DECODE_LAYER_STREAM=1`을 얹어 `QwenCausalLM` decode stage를 외부 `CommandStream`으로 replay하면, `REAL_BUNDLE_MAX_NEW_TOKENS=1` smoke는 통과해도 `gpu_infer --max-new-tokens 8`에서 timeout/hang이 재현됐다.
- 사용자가 같은 시점에 다시 `WindowServer` 문제를 관찰했으므로, 이 경로는 `unsafe pattern`으로 재분류한다.
- 따라서 "prebuilt graph/plan" 자체를 포기하는 것이 아니라, `layer-scope replay`를 금지하고 `safe sublayer batching only`를 다음 조건으로 삼는다.

## Method Rules Going Forward

앞으로는 다음 규칙을 지킨다.

1. `full forward` 또는 `full decode step` giant command buffer는 기본 경로에 넣지 않는다.
2. batching 실험은 `layer` 또는 그보다 더 작은 scope에서만 시작한다.
3. `fault`를 일으킨 실험 플래그는 문서와 코드에서 `experimental`로 표시한다.
4. 새 최적화는 한 번에 하나씩만 올린다.
5. `build/test` 통과만으로 GPU 경로를 기본값으로 승격하지 않는다. 실기기 benchmark와 fault check가 필요하다.
6. 새 scheduler는 `full-range`가 아니라 `bounded`여야 하며, attention + KV blit + 다수 layer를 한 command buffer에 무제한으로 섞지 않는다.
7. `prebuilt decode plan` 실험은 `layer replay`를 기본값으로 사용하지 않는다. `real buffer-range` tracking은 유지 가능하지만, 실행 scope는 이미 실기기에서 안전성이 확인된 sublayer batch 경계로 제한한다.

## References

- Apple Metal Best Practices Guide: <https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/index.html>
- Apple Metal Programming Guide, resource storage modes: <https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/WhatsNewiniOS9andOSX1011/WhatsNewiniOS9andOSX1011.html>
- full GPU vs PyTorch benchmark report: `Mac/gpu/build/reports/full_gpu_vs_pytorch.md`
