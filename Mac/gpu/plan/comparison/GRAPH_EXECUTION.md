# Graph Execution Comparison

## 현재 우리 코드 상태

- 실제 prebuilt graph executor는 없다.
- [`CommandScheduler`](/Volumes/990pro/Documents/SoC/Mac/gpu/include/runtime/command_scheduler.h)는 현재 prefill/decode를 `QwenCausalLM`에 그대로 위임하는 thin wrapper다.
- 즉 runtime은 graph-aware executor가 아니라 imperative call chain이다.
- [`plan/README.md`](/Volumes/990pro/Documents/SoC/Mac/gpu/plan/README.md)에 `graph-ready architecture`라는 설계 의도는 있지만, 현재 구현체는 아직 거기까지 가지 않았다.

## 다른 엔진은 어떻게 하는가

### `llama.cpp`

- graph를 먼저 만든다: `ggml_cgraph`
- graph allocator가 메모리 배치를 예약한다.
- backend scheduler가 node range와 memory hazard를 보고 command buffer를 분할한다.
- 즉 "실행 전에 dependency와 memory range를 안다".

### `mlx-lm`, `mlx-vlm`

- custom Metal command scheduler를 직접 쓰는 게 아니라 MLX의 lazy execution / `mx.compile`을 사용한다.
- hot function 단위로 compile된 graph 조각을 재사용한다.
- 즉 low-level command buffer graph가 아니라 higher-level compiled function graph에 가깝다.

### `transformers`

- model graph는 PyTorch가 실행한다.
- Metal fast path는 custom graph executor가 아니라 `MetalLinear + affine_qmm_t` 같은 low-bit op path에서 나온다.
- 즉 graph 자체보다 operator replacement가 핵심이다.

## 우리 코드에 필요한 다음 단계

1. full graph executor를 곧바로 만들지 않는다.
2. 먼저 decode 전용 `prebuilt execution plan`을 만든다.
3. 이 plan에는 다음이 들어가야 한다.
   - layer별 stage 목록
   - 각 stage의 입력/출력 tensor range
   - flush boundary
   - 재사용 가능한 pipeline key
   - temporary buffer offset 계획
4. 그 다음에야 bounded scheduler가 "매 토큰마다 조건문으로 결정"하는 구조에서 벗어날 수 있다.

## 권장 구현 순서

1. `hazard tracker`
   - `DeviceTensor buffer + offset + size` 기준 read/write range 추적
2. `decode stage enum`
   - 예: `AttnPrep`, `KvAppend`, `AttnContext`, `PostNormMlp`, `LmHead`
3. `decode plan builder`
   - model shape와 env flag를 받아 고정 stage plan 생성
4. `bounded executor`
   - stage별 encoder cap / flush reason 기록
5. `plan cache`
   - shape와 mode가 같으면 재사용

## 이번 턴 결론

- 지금 넣은 `SOC_GPU_ENABLE_EXPERIMENTAL_SAFE_DECODE_BATCH=1`은 graph executor는 아니지만, 그 이전 단계인 `bounded stage execution`이다.
- 실측상 command buffer 수를 줄이고 tok/s를 올렸기 때문에, 다음은 이를 일반화한 `decode plan`으로 가는 것이 맞다.

## 구현된 첫 단계

- 새 env: `SOC_GPU_ENABLE_EXPERIMENTAL_PREBUILT_DECODE_PLAN=1`
- 구현 내용:
  - scheduler가 layer별 decode stage 목록을 캐시한다.
  - hidden state는 persistent ping-pong buffer 2개를 재사용한다.
  - abstract slot read/write 기반 hazard tracker로 batch id를 계산한다.
- 구현 파일:
  - [`include/runtime/command_scheduler.h`](/Volumes/990pro/Documents/SoC/Mac/gpu/include/runtime/command_scheduler.h)
  - [`src/runtime/command_scheduler.cpp`](/Volumes/990pro/Documents/SoC/Mac/gpu/src/runtime/command_scheduler.cpp)

## 현재 평가

- correctness smoke와 `integration-real-bundle`은 통과했다.
- full benchmark는 `16.53 -> 16.86 tok/s`로 소폭 개선됐다.
- 하지만 quick run에서는 약간의 회귀가 있고, hazard tracker가 아직 `buffer id + byte range`가 아니라 `abstract slot` 기준이라 기본값으로 올리기엔 이르다.

즉 현재 상태는:

- `prebuilt decode plan`: `experimental`
- `safe stage batching`: 유효
- 다음 필수 작업: real buffer-range hazard tracker
