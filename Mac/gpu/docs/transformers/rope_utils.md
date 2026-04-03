# modeling_rope_utils.py 분석

## 원본 위치
`transformers/modeling_rope_utils.py`

## import 사용처
```python
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
```

## ROPE_INIT_FUNCTIONS 매핑

```python
ROPE_INIT_FUNCTIONS = {
    "linear":  _compute_linear_scaling_rope_parameters,
    "dynamic": _compute_dynamic_ntk_parameters,
    "yarn":    _compute_yarn_parameters,
    "longrope":_compute_longrope_parameters,
    "llama3":  _compute_llama3_parameters,
    "proportional": _compute_proportional_rope_parameters,
}
```

## Qwen3.5에서의 RoPE 사용

Qwen3.5TextConfig defaults (추정):
- `rope_type = "default"` → ROPE_INIT_FUNCTIONS에 없으므로 기본 방식
- `rope_theta = 1000000.0` (일반적인 Qwen 값)

**기본 RoPE inv_freq 계산** (ROPE_INIT_FUNCTIONS에 "default" 키 없음 → Embedding 클래스에서 직접 계산):
```python
inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
```

**Vision RoPE** (`Qwen3_5VisionRotaryEmbedding`):
```python
inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
```
- 3D MRoPE (temporal, height, width) 지원
- `position_ids`가 [batch, 3, seq_len] 형태

## dynamic_rope_update 데코레이터

RoPE forward에 적용하는 데코레이터. 2가지 분기:
1. `"dynamic"` rope_type: 시퀀스가 길어지면 NTK scaling으로 inv_freq 재계산
2. `"longrope"` rope_type: long/short factor로 inv_freq 전환

**Qwen3.5 해당 여부**: 
- 기본 rope_type이 "default"이면 → 데코레이터가 no-op (forward만 통과)
- config에서 확인 필요하지만, 대부분의 Qwen 모델은 "default" 사용

## RoPE 적용 (apply_rotary_pos_emb)

modeling_qwen3_5.py에 인라인 정의됨 (transformers 유틸 아님):
```python
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed
```

`rotate_half`: 전반부와 후반부를 교환하고 후반부를 negate:
```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
```

## C++ Metal 구현 전략

### inv_freq 초기화 (CPU에서 1회)
```cpp
// head_dim = config.head_dim (예: 128)
// theta = config.rope_theta (예: 1000000.0)
std::vector<float> inv_freq(head_dim / 2);
for (int i = 0; i < head_dim / 2; i++) {
    inv_freq[i] = 1.0f / powf(theta, (2.0f * i) / head_dim);
}
```

### cos/sin 테이블 사전 계산 (CPU에서 1회, max_seq_len까지)
```cpp
// freqs[pos][dim] = pos * inv_freq[dim]
// cos_table[pos][dim] = cos(freqs[pos][dim])
// sin_table[pos][dim] = sin(freqs[pos][dim])
```
→ Metal buffer에 업로드하여 kernel에서 참조

### RoPE 적용 (Metal kernel)
```metal
// threadgroup당 1개의 (q_idx, head_idx) 처리
// 각 thread: dim 차원의 2개 원소 (rotate_half 패턴)
float q0 = q[..., d];
float q1 = q[..., d + half_dim];
float c = cos_table[pos][d];
float s = sin_table[pos][d];
q_out[..., d]           = q0 * c - q1 * s;
q_out[..., d + half_dim] = q1 * c + q0 * s;
```

### Vision 3D MRoPE
- position_ids가 [batch, 3, seq_len] → 3개의 독립적 cos/sin
- head_dim을 3등분하여 각각에 temporal/height/width RoPE 적용
- 이건 Phase 3v에서 구현

### 구현하지 않을 것
- dynamic_rope_update의 "dynamic" / "longrope" 분기 (Qwen3.5가 "default" 사용 시)
- `linear`, `yarn`, `llama3` 등 다른 RoPE 타입
- RotaryEmbeddingConfigMixin 검증 로직
