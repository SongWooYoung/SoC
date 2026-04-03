# masking_utils.py 분석

## 원본 위치
`transformers/masking_utils.py`

## import 사용처
```python
from ...masking_utils import create_causal_mask
```

## `create_causal_mask` 호출 경로

`Qwen3_5TextModel.forward()`에서 호출:
```python
causal_mask = create_causal_mask(
    config=self.config,
    inputs_embeds=inputs_embeds,
    attention_mask=attention_mask,
    past_key_values=past_key_values,
    position_ids=position_ids,
)
```

## 마스크 생성 로직 요약

1. **전처리**: `_preprocess_mask_arguments()`가 q_length, kv_length, kv_offset 계산
   - cache가 있으면: `q_offset = cache.get_seq_length()`, `kv_length, kv_offset = cache.get_mask_sizes(q_length)`
   - 없으면: `q_offset = 0`, `kv_length = q_length`, `kv_offset = 0`
   
2. **마스크 함수 선택**: `config._attn_implementation`에 따라
   - `"sdpa"` → `sdpa_mask()` → boolean 4D 텐서 [batch, 1, q_len, kv_len]
   - `"eager"` → `eager_mask()` → float 4D 텐서 (0 / -inf)
   - `"flash_attention_2"` → `flash_attention_mask()` → None 또는 2D padding mask
   - `"flex_attention"` → `flex_attention_mask()` → BlockMask

3. **Causal mask 함수**: `kv_idx <= q_idx` (하삼각행렬)

4. **생략 가능 조건** (sdpa):
   - `query_length == 1` (decode): causal mask 불필요 → `is_causal=True` 대신 사용
   - `kv_length == query_length` (prefill, no padding): `is_causal=True` 대신 사용

## 하이브리드 캐시에서의 마스크 처리

Qwen3.5는 하이브리드 (full_attention + linear_attention):
- `create_causal_mask`는 **full_attention 레이어용** 마스크만 생성
- `is_sliding`에서 `False`인 레이어 (non-sliding = full attention)의 `layer_idx` 사용
- linear_attention 레이어는 별도 마스크 불필요 (GatedDeltaNet은 자체 gating)

## C++ Metal 구현 전략

### Prefill 단계
마스크를 명시적으로 생성할 필요 없음. 대신:
- Metal attention kernel 내부에서 `q_idx >= kv_idx` 조건으로 처리
- `attn_weights[q_idx][kv_idx] = (kv_idx <= q_idx + kv_offset) ? weight : -inf`
- 또는 triu mask를 kernel launch 시 전달

### Decode 단계
- 쿼리가 1개 토큰 → 모든 이전 KV에 attend → **마스크 불필요**
- 이것이 가장 빈번한 경우

### Padding mask
- 배치 크기 1이면 불필요 (배치마다 길이가 같으므로)
- 배치 > 1: padding token 위치를 boolean mask로 전달

### 구현하지 않을 것
- `sliding_window_causal_mask_function` (Qwen3.5에 sliding window 없음)
- `chunked_causal_mask_function` (chunked attention 없음)
- `flex_attention_mask` (torch.compile 전용)
- `vmap` 기반 마스크 확장
- `packed_sequence_mask` (training 전용)
- `bidirectional_mask` (encoder 전용)
- `AttentionMaskInterface` 레지스트리 시스템
