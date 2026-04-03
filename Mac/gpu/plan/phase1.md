# Phase 1: config.h

## 목표
`Qwen3_5TextConfig`의 모든 필드를 C++ struct로 옮기고, JSON config 파일을 파싱하여 로드한다.

## 원본
- `configuration_qwen3_5.py` → `Qwen3_5TextConfig`, `Qwen3_5VisionConfig`, `Qwen3_5Config`
- inference only이므로 `Qwen3_5TextConfig`만 우선 구현

## 구현 항목

### Qwen3_5TextConfig struct
```
vocab_size: 248320
hidden_size: 4096
intermediate_size: 12288
num_hidden_layers: 32
num_attention_heads: 16
num_key_value_heads: 4
hidden_act: "silu"
max_position_embeddings: 32768
rms_norm_eps: 1e-6
head_dim: 256
attention_bias: false
attention_dropout: 0.0
tie_word_embeddings: false

# linear attention 관련
linear_conv_kernel_dim: 4
linear_key_head_dim: 128
linear_value_head_dim: 128
linear_num_key_heads: 16
linear_num_value_heads: 32
layer_types: ["linear_attention", ...] (패턴 기반 생성)

# RoPE 관련
rope_parameters: { rope_type, rope_theta, mrope_section, partial_rotary_factor }
```

### JSON 파싱
- safetensors / HuggingFace 형식의 `config.json` 로드
- `layer_types` 자동 생성 로직 (`full_attention_interval` 기반)

## 결과물
- `models/qwen3_5/config.h`

## 상태
- [x] struct 정의
- [x] JSON 파싱
- [x] layer_types 생성 로직
- [x] 테스트 (Python config와 필드별 비교) — 68 tests passed
