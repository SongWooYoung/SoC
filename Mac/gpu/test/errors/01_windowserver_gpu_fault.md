# WindowServer GPU Fault Modes

## Why This File Exists

Mac mini M4 32GB에서 실제로 `WindowServer` 레벨 fault가 발생했다. 이 문서는 "최적화가 위험하다"가 아니라 "어떤 방식으로 쓰면 위험했는가"를 남기기 위한 기록이다.

## Confirmed Unsafe Pattern

다음 패턴은 기본 경로에서 금지한다.

1. `SOC_GPU_ENABLE_EXPERIMENTAL_COMMAND_STREAM=1`
2. 여러 transformer layer를 한 `CommandStream`에 누적
3. attention score/value, KV cache blit, matmul, norm dispatch를 한 giant command buffer에 함께 넣음

관찰된 결과:

1. 화면 깨짐
2. `WindowServer` 관련 경고창
3. inference 실패를 넘어 시스템 UI까지 오염

## Allowed Pattern

다음은 허용 대상이다.

1. 기본값: per-op command buffer
2. 제한적 실험: `SOC_GPU_ENABLE_EXPERIMENTAL_COMMAND_STREAM=layer`
3. 혼합 정밀도 실험: `SOC_GPU_ENABLE_EXPERIMENTAL_F16_WEIGHTS=1`을 따로 켜고 독립 검증

즉 batching 자체를 금지하는 것이 아니라, `scope`가 커지는 순간 위험해질 수 있음을 명시한다.

단, `layer` mode는 현재 기준으로도 채택되지 않았다. `WindowServer` fault 대신 종료 지연/정지 성향이 있어, 성능 최적화로 받아들이지 않고 실험 상태로 유지한다.

## Regression Rule

새로운 batching 변경은 아래를 만족해야 한다.

1. env flag 없이 기본 활성화하지 않는다.
2. 실기기에서 `integration-real-bundle`과 full-GPU benchmark를 통과해야 한다.
3. `WindowServer`/화면 깨짐이 한 번이라도 발생하면 해당 mode는 다시 `experimental`로 격하한다.

## Related Files

- `Mac/gpu/src/model/qwen_causal_lm.cpp`
- `Mac/gpu/include/metal/command_stream.h`
- `Mac/gpu/src/metal/command_stream.mm`
- `Mac/gpu/plan/13_GPU_FAULT_POSTMORTEM.md`
