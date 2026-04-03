# Qwen3.5 → C++ / Metal GPU Inference 프로젝트

## 최종 목표
HuggingFace transformers의 Qwen3.5 구현을 C++로 포팅하여 Metal GPU에서 inference를 실행하고,
PyTorch MPS baseline 대비 성능을 최적화한다.

## 범위
- **inference only** (train은 추후 별도 고려)
- **VLM 포함** — text (`Qwen3_5ForCausalLM`) → VLM (`Qwen3_5ForConditionalGeneration`) 순서로 구현
- 기존 `Mac/cpu/` 코드 재사용 **없음** — 완전 새로 구현

## 파일 매핑

| Python (원본)               | C++ (포팅 대상)    | 역할                         |
|-----------------------------|--------------------|------------------------------|
| `configuration_qwen3_5.py`  | `config.h`         | 모델 하이퍼파라미터 상수     |
| `tokenization_qwen3_5.py`   | `tokenization.h`   | BPE 토크나이저               |
| `modeling_qwen3_5.py`       | `qwen3_5.h`        | 전체 모델 (expanded, self-contained) |
| `modular_qwen3_5.py`        | `modular.h`        | 상위 조합 / 진입점           |

## 검증 및 최적화 지표
- Python(transformers) 출력과 C++ 출력의 **수치 일치** 확인 (logits / token 단위)
- `prefill_ms`, `decode_ms`, `wall_ms`, `throughput (tok/s)` 계측 및 최적화

## 의존성 조사 방법

### 원칙
`modeling_qwen3_5.py`가 auto-generated로 모든 클래스가 인라인되어 있으므로, 이 파일을 **1차 기준 소스**로 사용한다.
VLM(Vision) 관련 클래스(`Qwen3_5VisionModel`, `Qwen3_5VisionAttention` 등)도 이 파일에 이미 포함되어 있다.

- `qwen3/`, `qwen3_next/`, `qwen3_vl/`의 `.py`는 **다운로드하지 않는다** — `modeling_qwen3_5.py`에 전부 인라인됨
- `modular_qwen3_5.py`는 "이 클래스가 원래 어디서 왔는지" 확인할 때 참조용으로만 사용
- 이 파일만으로 부족한 transformers 유틸(`cache_utils`, `masking_utils`, `modeling_rope_utils` 등)만 on-demand로 GitHub raw URL에서 fetch

### 조사 깊이
- **torch 연산 레벨까지** 조사한다 (`F.softmax`, `F.silu`, `torch.matmul`, `nn.Conv1d` 등)
- 코드가 공개되어 있지 않은 부분은 동작을 최대한 유사하게 재구현한다. 그리고 재구현한 코드는 반드시 표기 후 정리해놓기
- 조사 결과는 `docs/{repo_name}/`에 모듈별로 정리해 둔다

### 조사 대상 분류

| 카테고리 | 예시 | 처리 방식 |
|----------|------|-----------|
| qwen3_5 고유 | `Qwen3_5Attention`, `Qwen3_5GatedDeltaNet` | `modeling_qwen3_5.py`에서 직접 읽음 |
| transformers 유틸 | `Cache`, `DynamicCache`, `create_causal_mask`, `ROPE_INIT_FUNCTIONS` | GitHub에서 해당 파일 fetch → 조사 → docs/ 정리 |
| torch 연산 | `F.softmax`, `nn.Linear`, `torch.rsqrt` | 조사 불필요 — Metal shader / C++ 직접 구현 |
| tokenizers 라이브러리 | `BPE`, `Regex`, `pre_tokenizers` | 인터페이스만 파악 → C++로 BPE 직접 구현 |

## Metal Shader 전략

### 2단계 접근

**Stage 1 — PyTorch MPS 동작 재현 (baseline)**
PyTorch가 `device="mps"`일 때 각 연산을 내부적으로 어떻게 Metal에 디스패치하는지 조사하고,
그 구조를 그대로 C++/Metal로 옮겨서 **output token이 Python과 일치**하는 것을 먼저 확인한다.

**Stage 2 — Custom 최적화 (experiments)**
baseline이 동작하면, 연산별로 커스텀 최적화를 실험한다. → Phase 5 참조

## 목차

| Phase | 파일 | 내용 |
|-------|------|------|
| 0 | [phase0.md](phase0.md) | 의존성 조사 |
| 1 | [phase1.md](phase1.md) | config.h |
| 2 | [phase2.md](phase2.md) | tokenization.h |
| 3 | [phase3.md](phase3.md) | qwen3_5.h — Text model (bottom-up) |
| 3v | [phase3v.md](phase3v.md) | qwen3_5.h — Vision model + VLM |
| 4 | [phase4.md](phase4.md) | 검증 (baseline 확정) |
| 5 | [phase5.md](phase5.md) | 최적화 실험 (EXP-1 ~ EXP-7) |
