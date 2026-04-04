# MLX→C++ Port — Phase 4: Validation & Benchmark

## 목표
MLX C++ 구현의 정확성과 성능을 MLX Python reference와 비교 검증한다.

---

## 4.1 정확성 검증

### Reference 생성 (MLX Python)

```bash
# mlx-vlm으로 reference 토큰 생성
pip install mlx-vlm
python -c "
from mlx_vlm import load, generate
model, processor = load('Qwen/Qwen3.5-4B')
# greedy decoding, non-thinking mode
for prompt in prompt_suite:
    result = generate(model, processor, prompt,
                      max_tokens=20, temperature=0.0)
    print(result)
"
```

### 비교 방법

| 비교 | 대상A | 대상B | 목표 |
|------|-------|-------|------|
| 1차 | mlx_cpp | mlx-vlm Python | Exact match (동일 구현) |
| 2차 | mlx_cpp | py_cpp | 허용 오차 범위 확인 |

### 테스트 프롬프트 (test/prompt_suite.json)

py_cpp Phase 4와 동일:
- "Hello" (단순)
- "What is 2+2?" (추론)
- "Translate to Korean: Good morning" (번역)
- 등 다양한 프롬프트

### 검증 기준

1. **Token-level exact match**: mlx_cpp vs mlx-vlm → 20/20 목표
2. **Logit 비교**: top-5 logit 값 비교 (numerical drift 측정)
3. **Layer-by-layer 디버깅**: 불일치 시 중간 hidden state 비교

---

## 4.2 성능 벤치마크

### 측정 항목

| 항목 | 설명 |
|------|------|
| Prefill tok/s | 프롬프트 처리 속도 |
| Decode tok/s | 토큰 생성 속도 |
| Memory (MB) | 피크 메모리 사용량 |
| TTFT (ms) | Time to first token |

### 비교 대상

| 구현 | 백엔드 |
|------|--------|
| mlx_cpp | libmlx.a (Metal GPU) |
| mlx-vlm Python | MLX Python (Metal GPU) |
| py_cpp | Accelerate (CPU) |

### 벤치마크 방법

```cpp
// C++ timing
auto t0 = std::chrono::high_resolution_clock::now();
// ... prefill ...
mx::eval(logits);  // 중요: eval 후 시간 측정
auto t1 = std::chrono::high_resolution_clock::now();
// ... decode loop ...
auto t2 = std::chrono::high_resolution_clock::now();

double prefill_ms = duration_cast<microseconds>(t1-t0).count() / 1000.0;
double decode_ms = duration_cast<microseconds>(t2-t1).count() / 1000.0;
```

---

## 4.3 최적화 기회

Phase 4 완료 후 검토:
1. **mx::compile** — 반복 subgraph를 JIT 최적화
2. **Quantization** — 4-bit/8-bit quantized weights (MLX 내장 지원)
3. **Prompt caching** — KV cache 재사용
4. **Batch decode** — 여러 시퀀스 동시 생성

---

## 상태
- [ ] 4.1 MLX Python reference 토큰 생성 (mlx-vlm)
- [ ] 4.2 mlx_cpp validation test 구현
- [ ] 4.3 Token-level exact match 달성
- [ ] 4.4 성능 벤치마크 (TTFT, tok/s)
- [ ] 4.5 py_cpp vs mlx_cpp 크로스 비교
