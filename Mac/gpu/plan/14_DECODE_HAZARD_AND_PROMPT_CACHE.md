# Decode Hazard And Prompt Cache Plan

## 목적

이번 단계의 목표는 두 가지다.

1. decode plan이 `buffer kind + byte offset + byte size` 기준 access 메타데이터를 갖게 한다.
2. prefill 결과를 `prompt cache artifact`로 저장/복원해서 prefill 비용과 decode 비용을 분리 검증할 수 있게 한다.

## Decode Hazard Tracker 설계

기존 `CommandScheduler`의 prebuilt decode plan은 hidden ping-pong slot만 보고 `batch_id`를 계산했다. 이 방식은 빠르게 실험하기엔 충분했지만, 실제 buffer layout을 반영하지 못했다.

이번 단계에서 plan metadata를 아래 구조로 확장한다.

- `DecodePlanBufferKind`
  - `HiddenSlot0`
  - `HiddenSlot1`
  - `Logits`
  - `KvKeys`
  - `KvValues`
- `DecodePlanAccessRange`
  - `buffer_kind`
  - `byte_offset`
  - `byte_size`
  - `write`

현재 구현은 decode token별 actual append offset을 사용해 layer별 KV write range를 계산한다. 즉 `layer base + current sequence_length * row_bytes`를 write 시작점으로 쓴다. 이 구조의 이점은 다음과 같다.

1. abstract slot보다 실제 Metal buffer layout에 더 가깝다.
2. layer 간 KV 접근이 non-overlap range로 표현된다.
3. hidden/logits/KV를 같은 tracker에서 볼 수 있다.
4. 이후 stage-local scheduler가 같은 metadata를 그대로 사용할 수 있다.

## Prompt Cache Artifact 설계

artifact는 benchmark와 regression에서 prefill 비용을 분리하기 위한 local binary file이다. 외부 입력 포맷으로 쓰지 않는다.

저장 내용은 다음과 같다.

1. fixed header
2. prompt token ids
3. layer별 `sequence_lengths`
4. KV key buffer bytes
5. KV value buffer bytes
6. prefill logits bytes

초기 구현은 prefill logits의 마지막 row만이 아니라 full prefill logits buffer 전체를 저장한다. 이유는 다음과 같다.

1. artifact load 후에도 기존 sampling 경로를 거의 그대로 재사용할 수 있다.
2. correctness 비교가 단순하다.
3. 구현 복잡도를 늘리지 않고 prefill/decode 분리 검증을 바로 시작할 수 있다.

artifact 크기가 부담되면 이후 단계에서 마지막 prompt row logits만 저장하는 경량 포맷으로 줄일 수 있다.

## 안정성 가드

1. artifact magic/version 검증
2. model config mismatch 검증
   - `vocab_size`
   - `layer_count`
   - `num_key_value_heads`
   - `head_dim`
3. `max_sequence_length` capacity 검증
4. payload byte count 검증
5. prompt cache 저장은 `running_token_ids == prompt_token_ids`일 때만 허용

## 테스트 기준

1. stepped prefill regression은 기존 runtime test에서 유지한다.
2. prompt cache artifact round-trip 후 generated token sequence와 running ids가 baseline과 같아야 한다.
3. prebuilt decode plan을 켠 2-layer identity model에서 stage access metadata가 layer별 KV byte offset을 올바르게 가져야 한다.
4. hazard 때문에 `batch_id`가 stage 진행에 따라 증가해야 한다.

## 현재 상태

1. decode plan은 actual KV append offset 기준 access metadata를 갖는다.
2. prompt cache artifact는 runtime test와 real-bundle regression report에 연결돼 있다.
3. report의 `GPU cached` column은 artifact load/restore 비용과 cached decode 비용을 분리해서 보여준다.

## 다음 단계

1. stage-local bounded scheduler가 이 metadata를 직접 사용해 flush 여부를 결정
2. prompt cache artifact를 benchmark_full_gpu_vs_pytorch 같은 장기 benchmark에도 연결
3. artifact 포맷을 경량화할지, full logits 저장을 유지할지 측정 기반으로 결정