# Agent Synthesis Execution Plan

## 목적

여러 agent 비교 분석(`llama.cpp`, `PyTorch MPS`, `flash-moe`, `mlx-lm`, `mlx-vlm`, `transformers`)을 현재 `Mac/gpu` 실측 결과와 합쳐서, 다음 수정 순서를 고정한다.

이 문서는 새 아이디어 목록이 아니라 다음 세 가지를 명확히 하기 위한 것이다.

1. 무엇을 유지할지
2. 무엇을 당장 수정할지
3. 무엇을 하지 말아야 하는지

## 이미 확인된 사실

### 1. 현재 repo의 방향 자체는 맞다

- `Mac/gpu`는 `ggml` backend처럼 범용 추상화로 돌아가는 구조가 아니라, 직접 제어하는 Metal runtime이다.
- 따라서 `llama.cpp`처럼 구조를 그대로 베끼는 것이 아니라, 그쪽의 교훈을 `runtime policy`, `hazard tracking`, `bounded batching`에 옮기는 것이 맞다.
- `PyTorch MPS`처럼 framework-managed execution으로 갈 필요도 없다. 지금 repo의 강점은 low-level scheduling control이다.

### 2. decode와 prefill은 분리해서 최적화해야 한다

- `q4 + safe decode batch`는 decode에서 의미 있는 개선을 냈다.
- 반대로 dense/q4 모두에서 prefill은 여전히 PyTorch 대비 크게 뒤처진다.
- 최근 실험에서도 prefill attention bounded batching은 real-bundle `GPU context` throughput을 개선했고, decode hidden-slot hazard relaxation은 실측 개선 없이 폐기됐다.

결론:

- decode 병목과 prefill 병목을 같은 정책으로 다루면 안 된다.
- 다음 수정도 `decode track`과 `prefill track`을 분리해서 진행한다.

### 3. giant batching은 다시 시도하지 않는다

- `llama.cpp`의 교훈은 giant batch가 아니라 hazard-aware graph partition이다.
- `flash-moe`도 수동 giant command buffer 결합이 항상 정답은 아니었다.
- 우리 repo도 실제로 large scope batching에서 fault와 회귀를 이미 경험했다.

결론:

- batching은 `작은 안전 경계`만 허용한다.
- 모든 batching 실험은 env gate + real-bundle 검증이 필수다.

## 외부 repo 분석을 합친 핵심 결론

### `llama.cpp`에서 가져와야 하는 것

- `real range hazard tracking`
- `bounded graph partition`
- host-visible 경로를 무조건 배제하지 않는 storage policy

### `PyTorch MPS`에서 가져와야 하는 것

- 메모리/실행 policy를 runtime 계층에서 관리하는 방식
- command buffer reuse와 커널 coalescing 관점
- fp16/low-bit 중심의 bandwidth 절감 우선순위

### `flash-moe`에서 가져와야 하는 것

- 큰 weight blob을 offset/view로 다루는 로딩 전략
- decode hot path에 대한 구조적 specialization
- 불필요한 read/alloc/copy 수를 줄이는 I/O 및 buffer policy

### `mlx-lm`, `mlx-vlm`에서 가져와야 하는 것

- `prefill_step_size`, prompt cache, KV policy 같은 runtime policy 분리
- prefill/completion batch를 분리해 다루는 실행 knobs
- working-set budget 기반 정책화

### `transformers`에서 가져와야 하는 것

- Apple Silicon fast path는 결국 low-bit fused linear가 핵심이라는 점
- dense kernel 미세 튜닝만으로는 MPS 격차를 다 메우기 어렵다는 점

## 수정 원칙

### 유지할 것

- direct Metal runtime
- `MetalContext` / `MetalBuffer` / `DeviceTensor` / `CommandStream` 구조
- env-gated 실험 경로
- real-bundle과 benchmark 기반 keep-or-revert 정책

### 수정할 것

- runtime policy 계층 강화
- decode/pre-fill 경로 분리
- storage mode 및 residency policy 명문화
- command-buffer batching budget 계측
- low-bit/fused decode path의 기본 승격 기준 정리

### 하지 말 것

- giant full-range batching 재시도
- 실측 없는 micro-optimization 유지
- `ggml` 또는 MPSGraph 스타일 대규모 구조 전환
- unified memory라는 이유만으로 shared-only 정책으로 회귀

## 실행 계획

## Phase 1. 계측 먼저 보강

목표:

- 다음 decode metadata 실험이 실제로 command buffer 수를 줄였는지 즉시 알 수 있게 만든다.
- prefill 실험이 어느 attention sub-stage에서 개선됐는지 분리해서 본다.

수정 항목:

1. `CommandScheduler` / decode plan에 아래 계측을 추가한다.
   - merged layer range count
   - stage group size histogram
   - token당 command buffer count
   - token당 encoder count
2. prefill attention batching 경로에 아래 label breakdown을 추가한다.
   - qkv projection
   - rope/norm
   - attention score/value
   - output projection
3. real-bundle report에 prefill-specific ratio를 더한다.
   - baseline context vs experimental prefill batching
   - prompt length별 throughput ratio

채택 조건:

- 계측 추가 후 기존 benchmark 출력이 깨지지 않아야 한다.
- `test_generation_context`와 `integration-real-bundle` 통과.

## Phase 2. prefill track

목표:

- 현재 확인된 `prefill attention batching` 이득이 우연이 아닌지 확인하고, bounded scope를 더 일반화한다.

수정 항목:

1. `SOC_GPU_ENABLE_EXPERIMENTAL_PREFILL_ATTENTION_BATCH=1` 경로를 prompt 길이별로 재측정한다.
2. 개선이 유지되면 다음 bounded scope만 추가 실험한다.
   - attention prep only
   - attention score/value only
   - single block prefill only
3. `prefill_step_size`와 상호작용을 같이 본다.
   - `step_size=1`
   - `step_size=4`
   - longer prompt

채택 기준:

- raw/chat 둘 다 `GPU context` throughput 개선
- longer prompt에서도 회귀 없음
- command buffer/encoder count 감소가 확인됨

폐기 기준:

- short prompt만 좋아지고 longer prompt에서 회귀
- real-bundle 기준 tok/s 개선 없이 gpu_ms만 줄어드는 경우

## Phase 3. decode track

목표:

- 다음 decode metadata 변경은 실제 merged range가 실행 경로까지 반영될 때만 유지한다.

수정 항목:

1. `prebuilt decode plan`의 `batch_id`만 바꾸는 실험은 중단한다.
2. 대신 `merged range -> actual command stream reduction`이 연결되는 지점부터 수정한다.
3. 다음 실험은 아래 조건을 모두 만족할 때만 시도한다.
   - merged range count가 증가함
   - full-run command buffer count가 감소함
   - steady-state tok/s가 증가함

우선 실험 후보:

1. `safe decode batch`와 `prebuilt decode plan`의 경계를 하나의 공용 policy로 통합
2. decode stage whitelist를 metadata 기반으로만 확장하지 말고, 실제 stream flush budget과 함께 결정
3. `encoder_count cap` 또는 `gpu_ms per flush` 상한을 둔 safe stream policy 추가

## Phase 4. storage / residency policy

목표:

- 현재 있는 `Shared` / `Private` / `PrivateInitialized`를 실제 policy 계층으로 끌어올린다.

수정 항목:

1. tensor class를 다음으로 분리한다.
   - static weight
   - kv cache
   - temporary workspace
   - token / metadata / staging
2. 각 class에 기본 storage mode를 명시한다.
   - static weight: `Private` 우선
   - kv cache: `Private` 우선, copy/update 패턴 측정
   - token / metadata: `Shared`
   - staging: `Shared`
3. model loader에서 개별 tensor 업로드 대신 packed blob + offset view 후보를 정리한다.

이 Phase의 목적은 즉시 tok/s를 올리는 것이 아니라, 이후 low-bit/fused path의 기반을 정리하는 것이다.

## Phase 5. low-bit / fused path

목표:

- dense micro-tuning보다 bandwidth reduction을 우선한다.

수정 항목:

1. `q4 decode` 경로를 dense와 별개 실험이 아니라 공용 decode projection 설계로 재정리한다.
2. 다음 우선순위는 유지한다.
   - `Q/K/V/O`
   - `Gate/Up/Down`
   - `LMHead`
3. dense kernel 추가 실험은 low-bit fused projection 방향을 방해하지 않는 범위에서만 유지한다.

판단 기준:

- quick labeled breakdown에서 projection label이 줄어드는지
- full benchmark에서 tok/s가 오르는지
- command buffer count가 그대로면 scheduler 문제가 남아 있음을 함께 기록할 것

## Phase 6. runtime policy layer

목표:

- kernel과 runtime policy를 분리해 long-prompt / prompt cache / KV evolution을 수용한다.

수정 항목:

1. `prefill_step_size`
2. prompt cache artifact
3. KV policy abstraction
4. recommended working set size 기반 budget knob

이 Phase는 즉시 1순위 성능 과제는 아니지만, prefill 최적화가 반복 실패하지 않게 만드는 구조 과제다.

## 바로 다음 순서

1. Phase 1의 decode/pre-fill 계측 추가
2. prefill attention batching의 longer-prompt benchmark 확장
3. decode plan은 metadata 완화 재시도 대신 merged-range 계측부터 추가

## keep / revert 규칙

모든 수정은 아래 순서를 지킨다.

1. 작은 env-gated 실험으로 넣는다.
2. `test_generation_context` 또는 관련 단위 테스트를 통과시킨다.
3. `integration-real-bundle`을 돌린다.
4. quick benchmark와 full benchmark를 모두 본다.
5. tok/s 개선이 없으면 revert하고 `test/errors/02_performance_regressions.md`에 남긴다.

## 현재 기준 최종 우선순위

1. prefill/decode 계측 보강
2. prefill bounded batching 일반화
3. decode merged-range 계측과 safe stream policy
4. storage/residency policy 정리
5. low-bit fused projection 재정리
6. runtime policy layer 확장