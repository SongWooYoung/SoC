# Phase 4: 검증 (baseline 확정)

## 목표
C++ 구현의 출력이 Python(transformers) 구현과 일치하는지 확인하고, baseline 성능 수치를 기록한다.

## 검증 설정

### 생성 모드
- **Non-thinking (Instruct) mode**: `enable_thinking=False`
- Thinking 모드는 출력이 길고 비결정적이므로 검증에는 non-thinking 사용
- Non-thinking 시 chat template이 빈 think 블록을 주입하여 모델이 바로 응답

### Chat Template (Non-thinking, text-only)
```
<|im_start|>system\n{system_message}<|im_end|>\n
<|im_start|>user\n{user_message}<|im_end|>\n
<|im_start|>assistant\n<think>\n\n</think>\n\n
```
- system message 생략 가능 (생략 시 system 줄 없음)
- `<|im_start|>` = 248045, `<|im_end|>` = 248046

### Sampling Parameters (Non-thinking general)
- `temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5`
- 검증은 **greedy (argmax)** 로 진행 (결정론적 비교 위해)

### Generation Limits
- `max_new_tokens=64` (검증용, 짧게)
- EOS: `<|im_end|>` (248046) 또는 `<|endoftext|>` (248044)

## 검증 방법

### Token-level 검증
1. Python에서 chat template 적용 + greedy + non-thinking → output token sequence 기록
2. C++에서 동일 prompt token sequence → output token sequence 비교
3. **완전 일치** 확인 (greedy decoding 기준)

### Logits-level 검증
1. Python에서 single forward pass → 마지막 token의 logits 전체 저장 (vocab_size 차원)
2. C++에서 동일 입력 → logits 비교
3. 허용 오차: `max |diff| < 1e-3` (bf16 기준)

### Layer-by-layer 검증 (디버깅용)
필요 시 각 서브모듈의 중간 출력을 비교:
- Embedding output
- RMSNorm output
- Attention output (per-layer)
- GatedDeltaNet output (per-layer)
- MLP output (per-layer)

## 성능 계측 (baseline 수치)

### 계측 항목
| 지표 | 설명 |
|------|------|
| `prefill_ms` | 전체 prompt를 처리하는 시간 |
| `decode_ms` | 토큰 1개 생성 평균 시간 |
| `wall_ms` | 전체 생성 완료까지 걸린 시간 |
| `throughput` | tok/s (생성된 토큰 수 / wall_ms * 1000) |

### 테스트 조건
- 모델: Qwen3.5-4B (raw safetensors, bf16)
- prompt: chat template 적용된 non-thinking text-only
- max_new_tokens: 64
- sampling: greedy (argmax)
- device: Apple Silicon (M4, Mac-mini 32GB)

## 결과물
- `test/gen_reference.py`: Python reference output 생성 (token IDs + logits 저장)
- `test/test_validation.cpp`: C++ output 비교 + 성능 계측
- baseline 성능 수치 기록 (표 형태)

## 상태
- [ ] Python reference output 생성 스크립트 (gen_reference.py)
- [ ] C++ 검증 도구 (test_validation.cpp)
- [ ] token-level 일치 확인
- [ ] logits-level 비교
- [ ] baseline 성능 수치 기록
