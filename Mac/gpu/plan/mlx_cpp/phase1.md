# MLX→C++ Port — Phase 1: Config & Tokenization

## 목표
config.py를 config.h로 변환한다. Tokenization은 py_cpp 포트와 동일하므로 재사용한다.

## 결과

### 1.1 Config — py_cpp 재사용
Config 구조체는 프레임워크 독립적이다. MLX와 PyTorch 경로 모두 같은 `config.json`을 읽으므로
기존 `models/qwen3_5_py_cpp/config.h`를 그대로 재사용한다.

`models/qwen3_5_mlx/config.h` → `#include "../qwen3_5_py_cpp/config.h"`

재사용 가능한 이유:
- 동일한 JSON 파일 (text_config 중첩 구조)
- 같은 필드: hidden_size, intermediate_size, GDN fields, rope_parameters
- `full_attention_interval` → `layer_types[]` 생성 로직 이미 구현됨
- `rope_parameters` dict 파싱 (rope_type, mrope_section 등) 이미 구현됨

### 1.2 Tokenization — py_cpp 재사용
- `models/qwen3_5_py_cpp/tokenization.h` 그대로 사용 가능
- 모델 프레임워크와 무관한 레이어

## 테스트
- `test/test_mlx_config.cpp` — 66 assertions, MLX 경로로 include하여 검증
  - TextConfig: 모든 GDN 필드, RoPE 파라미터, layer_types 패턴
  - full_attention_interval 생성 테스트 (12 layer, interval=4)
  - VisionConfig: depth, hidden_size, out_hidden_size
- `make test` 결과: 76 (py_cpp config) + 66 (mlx config) + 23 (tokenizer) = 165 passed, 0 failed

## 상태
- [x] config.h 구현 (py_cpp 재사용)
- [x] test_mlx_config 테스트 통과 (66/66)
- [x] 기존 테스트 regression 없음
