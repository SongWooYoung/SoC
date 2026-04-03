# Transformers 의존성 조사 결과

## 개요
`modeling_qwen3_5.py`가 import하는 transformers 내부 유틸을 조사하고,
C++ Metal 구현에서 각각 어떻게 대체할지 판단한다.

## 파일 목록

| 문서 | 원본 모듈 | 핵심 내용 |
|------|----------|----------|
| [cache_utils.md](cache_utils.md) | `transformers.cache_utils` | KV 캐시 + LinearAttention 캐시 |
| [masking_utils.md](masking_utils.md) | `transformers.masking_utils` | Causal / Sliding / Chunked attention mask |
| [rope_utils.md](rope_utils.md) | `transformers.modeling_rope_utils` | RoPE 주파수 초기화 & dynamic scaling |
| [activations.md](activations.md) | `transformers.activations` | ACT2FN 활성화 함수 매핑 |
| [gated_delta_net.md](gated_delta_net.md) | modeling_qwen3_5.py 인라인 | GatedDeltaNet fallback 구현 |
| [misc_utils.md](misc_utils.md) | 여러 transformers 유틸 | 기타 유틸 (training-only, 미구현 대상) |

## C++ 구현 판단 요약

| 카테고리 | 판단 | 이유 |
|----------|------|------|
| Cache (DynamicCache) | **직접 구현** | KV 텐서 append/concat만 필요 |
| LinearAttentionCache | **직접 구현** | conv_state + recurrent_state 관리 |
| Causal Mask | **간접** | Metal attention kernel 안에서 처리 |
| RoPE | **직접 구현** | inv_freq 계산 + cos/sin 적용 |
| ACT2FN | **직접 구현** | silu만 사용 (Qwen3.5 default) |
| GatedDeltaNet | **직접 구현** | torch fallback 코드를 Metal로 변환 |
| GenerationMixin | **불필요** | Python generation loop로 대체 |
| PreTrainedModel | **불필요** | weight loading만 필요 |
| FlashAttention | **불필요** | Metal attention kernel로 대체 |
| Tokenizer | **별도** | tokenizer.json 직접 파싱 or cpp library |
