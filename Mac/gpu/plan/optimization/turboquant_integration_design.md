# TurboQuant 방법론의 qwen3_5-mlx 적용 설계

## 개요

이 문서는 `0xSero/TurboQuant`의 방법론을 `qwen3_5-mlx`에 적용하기 위한 설계안이다.

참고 분석 문서:

- `Mac/gpu/docs/turboquant_analysis.md`

이 문서의 목적은 “TurboQuant를 그대로 포팅한다”가 아니라, `qwen3_5-mlx`의 구조와 Apple Silicon/MLX 제약에 맞게 **선별 적용 가능한 설계**를 고정하는 것이다.

---

## 1. 설계 목표

### 1.1 1차 목표

1. `full-attention KV cache`의 메모리 사용량을 줄인다.
2. long-context에서 더 많은 token capacity를 확보한다.
3. 기존 `qwen3_5-mlx` 구조를 크게 깨지 않고 실험 가능한 reference path를 만든다.
4. 정확도와 품질을 baseline과 비교할 수 있게 한다.

### 1.2 2차 목표

1. hybrid cache가 실제 long-context 실험에서 유의미한 memory saving을 주는지 확인한다.
2. decode 속도 손실을 정량화한다.
3. 이후 Metal kernel 최적화가 필요한지 판단한다.

### 1.3 비목표

1. linear-attention state(`conv_state`, `recurrent_state`)를 TurboQuant 방식으로 압축하지 않는다.
2. 첫 단계에서 decode 속도 개선을 보장하려고 하지 않는다.
3. vLLM monkey patch 구조를 재현하지 않는다.
4. CUDA/Triton 코드를 직접 이식하지 않는다.

---

## 2. 적용 범위

TurboQuant 방법론은 `qwen3_5-mlx`의 모든 layer에 적용되지 않는다.

### 2.1 적용 대상

- full-attention layer의 `KVCache`

### 2.2 적용 제외

- linear-attention layer의 `ArraysCache`
- `conv_state`
- `recurrent_state`
- `GatedDeltaNet` 내부 상태

이 결정의 이유는 단순하다.

1. TurboQuant는 dot-product attention의 key/value cache를 전제로 한다.
2. `qwen3_5-mlx`의 linear-attention layer는 KV cache가 아니라 recurrence state를 사용한다.
3. 따라서 TurboQuant를 전체 cache 전략으로 보는 것은 맞지 않고, **full-attention 전용 압축 계층**으로 보는 것이 맞다.

---

## 3. qwen3_5-mlx에 맞는 해석

TurboQuant repo에서 실제로 가져올 것은 코드 자체가 아니라 아래 네 가지 개념이다.

1. `exact recent buffer + compressed history`
2. key용 `rotation + Lloyd-Max + residual sign correction`
3. value용 group quantization
4. chunk append + lazy materialization 형태의 compressed store

반대로 그대로 가져오지 않는 것은 아래다.

1. vLLM integration layer
2. Triton kernels
3. CUDA-specific memory path
4. monkey patch 방식

즉 `qwen3_5-mlx`에 대한 실제 적용은 다음 한 줄로 요약된다.

> full-attention KV cache를 exact recent segment와 compressed historical segment로 나누는 hybrid cache를 새로 설계한다.

---

## 4. 제안 아키텍처

### 4.1 새 runtime mode

기존 full-attention cache mode는 다음을 가진다.

- `legacy`
- `step_buffer`

여기에 아래 mode를 추가한다.

- `turboquant_ref`

선택적으로 이후에 추가할 수 있다.

- `turboquant_hybrid`
- `turboquant_fused`

첫 구현은 `turboquant_ref` 하나로 충분하다.

### 4.2 새 cache abstraction

`models/qwen3_5_mlx/language.h` 기준으로 기존 `KVCache`와 별도로 아래 구조를 둔다.

```cpp
struct TurboQuantKVCache {
    ExactRecentCache recent;
    CompressedKeyStore compressed_keys;
    CompressedValueStore compressed_values;

    int recent_capacity;
    int total_tokens;
    int num_kv_heads;
    int head_dim;

    TurboQuantKeyParams key_params;
    TurboQuantValueParams value_params;
};
```

### 4.3 내부 세그먼트 구조

#### ExactRecentCache

- 최근 `L` token의 exact key/value 유지
- decode hot path에서 append cheap하게 유지
- small context에서는 이것만 써도 되도록 함

#### CompressedKeyStore

- rotated key quantization 결과 저장
- MSE index packed buffer
- residual sign packed buffer
- norm / residual norm buffer

#### CompressedValueStore

- group-wise packed value
- scale / zero buffer

---

## 5. 데이터 구조 상세

### 5.1 Key 압축 파라미터

```cpp
struct TurboQuantKeyParams {
    int bits;
    int head_dim;
    mx::array rotation_matrix;     // [D, D]
    mx::array qjl_matrix;          // [D, D]
    mx::array centroids;           // [2^b]
    mx::array boundaries;          // [2^b + 1]
    float qjl_scale;
};
```

설계 원칙:

1. layer별로 고정된 rotation/QJL matrix 사용
2. codebook은 head_dim/bit-width별로 공유 가능
3. 초기화 시점에 모두 준비

### 5.2 Key 저장 구조

```cpp
struct CompressedKeyStore {
    mx::array mse_indices;        // packed uint8
    mx::array qjl_signs;          // packed uint8
    mx::array norms;              // fp16/fp32
    mx::array residual_norms;     // fp16/fp32
    int token_count;
};
```

첫 단계에서는 append를 단순화하기 위해 step-buffer 방식 또는 chunk append 방식을 사용한다.

### 5.3 Value 저장 구조

```cpp
struct TurboQuantValueParams {
    int bits;
    int group_size;
};

struct CompressedValueStore {
    mx::array packed_values;      // uint8
    mx::array scales;             // fp16/fp32
    mx::array zeros;              // fp16/fp32
    int token_count;
};
```

첫 구현은 품질 보존을 위해 `value_bits=4`를 기본값으로 시작한다. `2-bit`는 나중 단계에서만 검토한다.

---

## 6. 실행 시퀀스

### 6.1 Prefill

prefill에서는 full-attention layer에서 생성된 key/value를 아래처럼 처리한다.

1. 새 KV가 들어오면 token 길이를 본다.
2. 전체가 `recent_capacity` 이하이면 exact recent로만 저장한다.
3. capacity를 초과하면 prefix는 compressed history로 이동시키고 tail만 recent로 유지한다.

prefill 단계 목표:

- correctness 확보
- 전체 long prompt에 대해 hybrid cache state 구축

prefill 단계에서 속도보다 중요한 건 “정확히 같은 attention output이 나오는지”다.

### 6.2 Decode

decode 한 step에서 query가 들어오면:

1. recent exact segment에 대해 exact attention score 계산
2. compressed history에 대해 reference score 계산
3. 둘을 합쳐 최종 attention output 계산
4. 새 token의 K/V를 recent에 append
5. recent가 넘치면 가장 오래된 chunk를 compressed history로 flush

### 6.3 Reference score path

첫 단계의 compressed history score는 다음 두 방식 중 하나다.

#### Option A: dequantize-then-attend

- 구현이 쉽다.
- 속도는 느릴 수 있다.
- correctness 기준선으로 적합하다.

#### Option B: score-only reconstruction

- key를 full dequantize하지 않고 inner-product만 계산
- 구현 난도는 높지만 memory overhead가 작다.

첫 단계는 `Option A`로 가는 것이 현실적이다.

---

## 7. 구현 단계

### Phase 1: Reference path

목표:

- long-context memory saving 여부 확인
- output/logit correctness 비교
- 구현 복잡도 최소화

할 일:

1. `TurboQuantKVCache` 구조 추가
2. full-attention layer에서 mode switch 추가
3. key rotation/codebook load path 추가
4. value group quantization 추가
5. `dequantize-then-attend` reference path 구현
6. short/long prompt correctness 및 memory 측정

완료 조건:

- exact path와 비교해 major divergence 없음
- full-attention KV memory 감소 수치 확보

### Phase 2: Long-context hybrid mode

목표:

- 짧은 문맥에서는 기존 exact path 유지
- 긴 문맥에서만 TQ path 활성화

할 일:

1. threshold activation 추가
2. recent capacity sweep (`64/128/256`)
3. `key_bits 3/4`, `value_bits 4` 비교
4. needle / long-context retrieval 평가

완료 조건:

- long-context memory extension 유의미
- quality drop 허용 범위 내

### Phase 3: Metal optimization

목표:

- decode cost를 실제로 줄일 수 있는지 확인

할 일:

1. MSE score fused kernel 후보 정의
2. QJL score fused kernel 후보 정의
3. recent + compressed merge path 최적화
4. 가능한 경우 online softmax 형태로 확장

완료 조건:

- reference path 대비 decode overhead 유의미하게 감소

---

## 8. 품질 및 성능 검증 계획

### 8.1 정확도 검증

필수 비교 항목:

1. full-attention layer output cosine similarity
2. full-model logits max abs diff
3. greedy decode token match rate
4. long-context retrieval/needle test

### 8.2 성능 검증

필수 비교 항목:

1. prefill ms
2. decode ms/tok
3. peak memory
4. max supported context length

### 8.3 판정 기준

`turboquant_ref` 단계에서는 아래가 핵심이다.

1. decode 속도 개선 여부는 필수 아님
2. memory saving과 correctness가 핵심
3. speed regression이 과도하면 fused path 필요성 확인

---

## 9. 기존 qwen3_5-mlx 구조와의 접점

### 9.1 수정 대상

- `models/qwen3_5_mlx/language.h`
- full-attention cache struct/class
- runtime option parser
- output eval / benchmark harness

### 9.2 직접 수정하지 않는 대상

- `GatedDeltaNet` 수학
- linear-attention `ArraysCache`
- recurrent kernel path

즉 이 설계는 현재 decode 병목 조사와도 충돌하지 않는다. TurboQuant path는 **memory extension 기능 추가**로 보고, 기존 base-vs-custom decode root-cause 분석과 분리해서 진행할 수 있다.

---

## 10. 주요 리스크

### 10.1 속도 악화 가능성

Metal fused kernel 없이 compressed history를 복원해 attention을 하면 decode가 느려질 수 있다.

대응:

- 처음부터 speed feature로 포장하지 않는다.
- memory feature로 시작한다.

### 10.2 품질 저하 가능성

특히 low-bit value quantization이 품질을 흔들 수 있다.

대응:

- 초기 기본값은 `key_bits=3`, `value_bits=4`
- `2-bit value`는 추후 실험

### 10.3 라이선스 이슈

원본 repo는 GPL-3.0이다.

대응:

- 직접 코드 혼합은 피한다.
- 수학/방법론을 바탕으로 독자 구현한다.

### 10.4 구조적 절감 한계

모델이 hybrid attention 구조라 linear-attention 상태는 압축되지 않는다.

대응:

- 전체 cache 절감이 아니라 full-attention KV 절감으로 KPI를 정의한다.

---

## 11. 구현 우선순위

우선순위는 아래 순서가 맞다.

1. `turboquant_ref` cache mode 추가
2. codebook/rotation 초기화 path 추가
3. exact recent + compressed history data structure 구현
4. reference decode attention path 구현
5. correctness + memory benchmark
6. long-context benchmark
7. 이후 필요한 경우에만 Metal fused kernel 설계

---

## 12. 최종 판단

`qwen3_5-mlx`에 대한 TurboQuant 적용은 다음처럼 정의하는 것이 가장 정확하다.

> TurboQuant는 decode 성능 최적화의 즉시 해답이 아니라, full-attention KV cache를 long-context 친화적으로 재설계하는 방법론 후보이다.

따라서 진행 방식도 아래가 맞다.

1. 먼저 reference hybrid cache로 correctness와 memory 효과를 본다.
2. 효과가 있으면 long-context 전용 feature로 고정한다.
3. 그 다음에만 Metal kernel 최적화를 검토한다.

이 순서를 지키면 현재 진행 중인 decode root-cause 분석과도 충돌하지 않고, 기능 추가와 성능 분석을 분리해서 진행할 수 있다.