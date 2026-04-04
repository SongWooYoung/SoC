# Phase 0: 의존성 조사

## 목표
구현에 필요한 모든 외부 의존성을 조사하고 정리한다.
이후 Phase에서 코드를 짤 때 추가 조사 없이 바로 구현에 들어갈 수 있는 상태를 만든다.

---

## 0a. 소스 파일 개요 (models/qwen3_5_mlx/)

| 파일 | 라인수 | 역할 |
|------|--------|------|
| `__init__.py` | 6 | re-export (Model, LanguageModel, VisionModel, configs, processor patch) |
| `config.py` | 101 | TextConfig, VisionConfig, ModelConfig dataclass |
| `language.py` | ~700 | 전체 텍스트 모델 (RotaryEmb, Attention, GDN, MLP, DecoderLayer, Model, LanguageModel) |
| `gated_delta.py` | 283 | GatedDeltaNet 핵심: Metal kernel JIT + ops fallback |
| `qwen3_5.py` | 138 | VLM composite (vision + language 결합), Qwen3VLModel 상속 |
| `vision.py` | 5 | pass-through (Qwen3VLVisionModel 상속) |

---

## 0b. 전체 import 목록 (파일별)

### `__init__.py`
```python
from ..base import install_auto_processor_patch
from ..qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from .config import ModelConfig, TextConfig, VisionConfig
from .qwen3_5 import LanguageModel, Model, VisionModel
```

### `config.py`
```python
import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from ..base import BaseModelConfig                        # -> mlx_vlm/models/base.py
from ..qwen3_vl.config import VisionConfig as Qwen3VLVisionConfig  # -> mlx_vlm/models/qwen3_vl/config.py
```

### `language.py` (핵심)
```python
from typing import Any, Optional
import mlx.core as mx                                     # -> mlx C++ (libmlx.a)
import mlx.nn as nn                                       # -> mlx Python nn (직접 합성)
from mlx_lm.models.activations import swiglu              # -> .repo_cache/mlx-lm/
from mlx_lm.models.gated_delta import gated_delta_update  # -> .repo_cache/mlx-lm/
from ..base import (                                      # -> mlx_vlm/models/base.py
    LanguageModelOutput,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from ..cache import ArraysCache, KVCache                  # -> mlx_vlm/models/cache.py -> mlx_lm/models/cache.py
from .config import ModelConfig, TextConfig
```

### `gated_delta.py`
```python
from functools import partial
from typing import Optional, Tuple
import mlx.core as mx
import mlx.nn as nn
```

### `qwen3_5.py`
```python
from typing import Optional
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from ..base import InputEmbeddingsFeatures               # -> mlx_vlm/models/base.py
from ..qwen3_vl import Model as Qwen3VLModel             # -> mlx_vlm/models/qwen3_vl/qwen3_vl.py
from ..qwen3_vl import processing_qwen3_vl               # -> mlx_vlm/models/qwen3_vl/processing_qwen3_vl.py
from ..qwen3_vl.qwen3_vl import masked_scatter           # -> mlx_vlm/models/qwen3_vl/qwen3_vl.py
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel
```

### `vision.py`
```python
from ..qwen3_vl import VisionModel as Qwen3VLVisionModel  # -> mlx_vlm/models/qwen3_vl/vision.py
```

---

## 0c. 의존성 트리 (소스 -> 업스트림)

```
models/qwen3_5_mlx/
+-- __init__.py
|     +-- mlx_vlm/models/base.py --------------- (A)
|     +-- mlx_vlm/models/qwen3_vl/processing_qwen3_vl.py -- (F, Vision-only)
|     +-- .config
|     +-- .qwen3_5
|
+-- config.py
|     +-- mlx_vlm/models/base.py --------------- (A) BaseModelConfig
|     +-- mlx_vlm/models/qwen3_vl/config.py ---- (E) VisionConfig
|
+-- language.py  <-- 핵심 파일
|     +-- mlx.core ----------------------------- (1) libmlx.a
|     +-- mlx.nn ------------------------------- (2) 직접 합성
|     +-- mlx_lm/models/activations.py --------- (B) swiglu
|     +-- mlx_lm/models/gated_delta.py --------- (C) gated_delta_update
|     +-- mlx_vlm/models/base.py --------------- (A) create_attention_mask 외
|     +-- mlx_vlm/models/cache.py -------------- (D) -> re-export from mlx_lm/models/cache.py
|
+-- gated_delta.py
|     +-- mlx.core ----------------------------- (1)
|     +-- mlx.nn ------------------------------- (2)
|
+-- qwen3_5.py (VLM composite)
|     +-- mlx.core ----------------------------- (1)
|     +-- mlx.nn ------------------------------- (2)
|     +-- numpy -------------------------------- (빌드 불필요, masked_scatter에서만 사용)
|     +-- mlx_vlm/models/base.py --------------- (A) InputEmbeddingsFeatures
|     +-- mlx_vlm/models/qwen3_vl/qwen3_vl.py - (G) Model, masked_scatter
|     +-- mlx_vlm/models/qwen3_vl/vision.py --- (H) VisionModel
|
+-- vision.py -> mlx_vlm/models/qwen3_vl/vision.py -- (H)
```

---

## 0d. 업스트림 의존성 상세

### (1) mlx.core -- MLX C++ 라이브러리

**소스**: `.repo_cache/mlx/` (Apple MLX)
**빌드 산출물**: `build/libmlx.a` (34MB) + `build/mlx/backend/metal/kernels/mlx.metallib`
**링크 플래그**: `-lmlx -framework Metal -framework Foundation -framework Accelerate -framework QuartzCore`

#### 사용되는 모든 mx.core 연산

| Python API | C++ API (mlx::core::) | 사용 파일 | 상태 |
|-----------|------------------------|----------|------|
| `mx.arange(start, stop, step, dtype)` | `arange(start, stop, step, dtype)` | language | verified |
| `mx.add(a, b)` | `add(a, b)` or `operator+` | language | verified |
| `mx.array(data)` | `array(data, shape, dtype)` | language, qwen3_5 | verified |
| `mx.broadcast_to(a, shape)` | `broadcast_to(a, shape)` | language, qwen3_5 | verified |
| `mx.concatenate(arrays, axis)` | `concatenate(arrays, axis)` | language | verified |
| `mx.cos(x)` | `cos(x)` | language | verified |
| `mx.cumsum(a, axis)` | `cumsum(a, axis)` | language | verified |
| `mx.exp(x)` | `exp(x)` | gated_delta | verified |
| `mx.expand_dims(a, axis)` | `expand_dims(a, axis)` | language, gated_delta | verified |
| `mx.log(x)` | `log(x)` | language | verified |
| `mx.matmul(a, b)` | `matmul(a, b)` | (nn.Linear 경유) | verified |
| `mx.ones(shape, dtype)` | `ones(shape, dtype)` | language | verified |
| `mx.ones_like(a)` | `ones_like(a)` | language | verified |
| `mx.repeat(a, n, axis)` | `repeat(a, n, axis)` | gated_delta | verified |
| `mx.reshape(a, shape)` | `reshape(a, shape)` | language | verified |
| `mx.sigmoid(x)` | `sigmoid(x)` | language, gated_delta | verified |
| `mx.sin(x)` | `sin(x)` | language | verified |
| `mx.split(a, indices, axis)` | `split(a, indices, axis)` | language | verified |
| `mx.stack(arrays, axis)` | `stack(arrays, axis)` | language, gated_delta | verified |
| `mx.sum(a, axis)` | `sum(a, axis)` | language | verified |
| `mx.swapaxes(a, ax1, ax2)` | `swapaxes(a, ax1, ax2)` | language | verified |
| `mx.tile(a, reps)` | `tile(a, reps)` | language | verified |
| `mx.where(cond, a, b)` | `where(cond, a, b)` | language, gated_delta | verified |
| `mx.zeros(shape, dtype)` | `zeros(shape, dtype)` | language, gated_delta | verified |
| `mx.zeros_like(a)` | `zeros_like(a)` | language | verified |
| `mx.conv1d(x, w, stride, pad, dil, groups)` | `conv1d(x, w, stride, pad, dil, groups)` | (nn.Conv1d 경유) | verified |
| `mx.random.uniform(low, high, shape)` | `random::uniform(shape, low, hi, dtype)` | language (init only) | verified |
| `x.astype(dtype)` | `astype(x, dtype)` | language, gated_delta | verified |
| `x.reshape(shape)` | `reshape(x, shape)` | language | verified |
| `x.transpose(axes)` | `transpose(x, axes)` | language | verified |
| `x.flatten()` | `flatten(x)` | language | verified |

#### fast 연산 (mlx/fast.h)

| Python API | C++ API (mlx::core::fast::) | 사용 파일 | 상태 |
|-----------|------------------------------|----------|------|
| `mx.fast.rms_norm(x, w, eps)` | `rms_norm(x, w, eps)` | language | verified |
| `mx.fast.rope(x, dims, ...)` | `rope(x, dims, ...)` | (language.py는 수동 RoPE) | verified |
| `mx.fast.scaled_dot_product_attention(Q,K,V,scale,mask)` | `scaled_dot_product_attention(Q,K,V,scale,mask,mask_mode)` | (base.py 경유) | verified |
| `mx.fast.metal_kernel(name, inputs, outputs, source)` | `metal_kernel(name, inputs, outputs, source, ...)` | gated_delta | verified |

#### Metal / Device

| Python API | C++ API | 비고 |
|-----------|---------|------|
| `mx.metal.is_available()` | `mlx::core::metal::is_available()` | GPU 가용성 확인 |
| `mx.default_device()` | `mlx::core::default_device()` | 현재 디바이스 |
| `mx.gpu` | `mlx::core::Device(mlx::core::Device::gpu)` | 디바이스 비교 |
| `mx.compile(fn, shapeless)` | `mlx::core::compile(fn)` | JIT 컴파일 데코레이터 |

#### I/O (safetensors 로딩)

| Python API | C++ API | 비고 |
|-----------|---------|------|
| `mx.load(path)` / `mx.load_safetensors(path)` | `mlx::core::load_safetensors(path)` | SafetensorsLoad = pair<unordered_map<string,array>, unordered_map<string,string>> |
| `mx.eval(array)` | `mlx::core::eval(array)` | lazy evaluation 실행 |

#### 발견된 API 차이 (검증 테스트에서 확인)

| 항목 | Python | C++ | 비고 |
|------|--------|-----|------|
| SDPA mask_mode | "none" | "" (빈 문자열) | 유효값: "causal", "array", "" |
| rope 시그니처 | rope(x, dims, traditional, base, ...) | rope(x, dims, traditional, base, ...) | base가 optional<float>, freqs 맨 뒤 optional |
| Shape 타입 | list/tuple | SmallVector<int32_t> (= Shape) | vector<int>에서 Shape 변환 필요 |
| array 기본 생성자 | 해당 없음 | **없음** | 모든 struct 멤버에 초기값 필수: = mx::array(0.0f) |

---

### (2) mlx.nn -- 직접 합성 대상

**소스**: `.repo_cache/mlx/python/mlx/nn/layers/`
**빌드**: 불필요 (C++에서 mlx.core 연산으로 직접 합성)

#### nn.Module

**원본**: `base.py` -- dict 상속, `__setattr__`로 파라미터 자동 등록
**C++ 전략**: 구조체로 대체 (weight 멤버를 직접 선언)

#### nn.Linear

**원본**: `linear.py`
- **파라미터**: `weight` [out, in], `bias` [out] (optional)
- **forward**: `x @ weight.T + bias` (= `mx.addmm(bias, x, weight.T)`)
- **C++ 합성**:
```cpp
mx::array linear(const mx::array& x, const mx::array& w, const mx::array* bias = nullptr) {
    auto y = matmul(x, transpose(w));
    if (bias) y = y + *bias;
    return y;
}
```

#### nn.Embedding

**원본**: `embedding.py`
- **파라미터**: `weight` [vocab, dim]
- **forward**: `weight[x]` (= `take(weight, x, 0)`)
- **as_linear(x)**: `x @ weight.T` (tied embedding/lm_head)
- **C++ 합성**: `take(weight, indices, 0)`

#### nn.Conv1d

**원본**: `convolution.py`
- **파라미터**: `weight` [out_ch, kernel, in_ch/groups], `bias` [out_ch] (optional)
- **입력 형식**: NLC (batch, length, channels)
- **forward**: `mx.conv1d(x, weight, stride, padding, dilation, groups) + bias`
- **C++ 합성**: `conv1d(x, w, stride, padding, dilation, groups)`
- **주의**: safetensors 저장 형식은 PyTorch [O, I/g, kW]. MLX는 [O, kW, I/g]. swapaxes(w, 1, 2) 필요

#### nn.RMSNorm

**원본**: `normalization.py`
- **파라미터**: `weight` [dim] (ones 초기화)
- **forward**: `mx.fast.rms_norm(x, weight, eps)`
- **C++ 합성**: `fast::rms_norm(x, weight, eps)`
- **주의**: 일부 모델 sanitize에서 weight += 1.0 (norm weight 보정). Qwen3.5는 해당 없음

#### nn.silu

**원본**: `activations.py`
- **구현**: `x * mx.sigmoid(x)` (mx.compile 데코레이터)
- **C++ 합성**: `x * sigmoid(x)`

#### nn.softplus

**원본**: `activations.py`
- **구현**: `mx.where(x > 20, x, mx.log(1 + mx.exp(x)))` (수치 안정성)
- **C++ 합성**: `where(x > 20, x, log(1 + exp(x)))`

---

### (A) mlx_vlm/models/base.py

**소스**: `.repo_cache/mlx-vlm/mlx_vlm/models/base.py` (321줄)

사용되는 export:

| 이름 | 종류 | 구현 상세 | C++ 전략 |
|------|------|----------|----------|
| BaseModelConfig | dataclass | from_dict(params) + to_dict(). inspect.signature로 유효 필드만 추출 | 구조체 (config.h에서 이미 구현) |
| LanguageModelOutput | dataclass | logits: mx.array, hidden_states, cross_attention_states, encoder_outputs (모두 Optional) | 구조체 (logits만 사용) |
| InputEmbeddingsFeatures | dataclass | inputs_embeds: mx.array + 10개 Optional 필드 | VLM phase에서 구현 |
| create_attention_mask | 함수 | mlx_lm/models/base.py로 위임 (아래 A-lm 참조) | 직접 구현 |
| create_ssm_mask | 함수 | mlx_lm/models/base.py로 위임 | GDN에서 사용 |
| scaled_dot_product_attention | 함수 | mlx_lm/models/base.py로 위임 | fast::sdpa 호출 |
| install_auto_processor_patch | 함수 | HF AutoProcessor monkey-patch (Python only) | 불필요 (C++ 추론) |
| ensure_fused_sdpa | 함수 | head_dim을 64/80/128로 패딩 후 sdpa 호출 | Vision에서만 사용 |
| chunked_attention | 함수 | Q를 chunk_size로 분할 후 sdpa 반복 | Vision에서만 사용 |
| expand2square, pixel_shuffle, interpolate | 함수 | 이미지 전처리 유틸 | Vision phase에서 구현 |

---

### (A-lm) mlx_lm/models/base.py

**소스**: `.repo_cache/mlx-lm/mlx_lm/models/base.py` (137줄)

mlx_vlm/models/base.py 가 re-export하는 실제 구현체.

#### create_causal_mask(N, offset=0) -> mx.array
```python
rinds = mx.arange(offset, offset + N)
linds = mx.arange(offset, offset + N) if offset else rinds
mask = linds[:, None] >= rinds[None]
return mask * 0.0 - (1 - mask) * 1e9
```
삼각 마스크 생성. 0.0 / -1e9 값.

#### create_attention_mask(h, cache_offset=None) -> Optional[mx.array]
```python
T = h.shape[1]
if T > 1:
    mask = create_causal_mask(T, cache_offset or 0)
    mask = mask.astype(h.dtype)
    return mask
return None
```
seq_len > 1 (prefill)일 때만 causal mask 반환. decode step에서는 None.

#### create_ssm_mask(h, cache, T) -> Optional[mx.array]
```python
if isinstance(cache, ArraysCache):
    return cache.make_mask(T)
return None
```
ArraysCache (GDN recurrent 상태)에 대한 마스크.

#### scaled_dot_product_attention(q, k, v, scale, mask) -> mx.array
```python
return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
```
단순 위임. C++에서 fast::scaled_dot_product_attention 직접 호출로 대체.

---

### (B) mlx_lm/models/activations.py

**소스**: `.repo_cache/mlx-lm/mlx_lm/models/activations.py` (43줄)

| 이름 | 구현 | C++ 합성 |
|------|------|----------|
| swiglu(gate, x) | @mx.compile(shapeless=True): nn.silu(gate) * x | (gate * sigmoid(gate)) * x |
| xielu(x) | 미사용 (skip) | -- |

---

### (C) mlx_lm/models/gated_delta.py

**소스**: `.repo_cache/mlx-lm/mlx_lm/models/gated_delta.py` (283줄, models/qwen3_5_mlx/gated_delta.py와 동일)

| 이름 | 역할 | C++ 전략 |
|------|------|----------|
| compute_g(A_log, a, dt_bias) | g = exp(-exp(A_log)) * softplus(a + dt_bias) | 직접 합성 |
| _gated_delta_step_ops(q, k, v, g, b, state) | 단일 토큰 recurrent step: state = g * state + (v (x) k) * b | 직접 합성 |
| _gated_delta_ops(q, k, v, g, b, state, mask) | prefill: T 토큰 순차 처리 (for loop) | 직접 합성 |
| gated_delta_update(q, k, v, g, b, state, mask) | Metal kernel -> ops fallback 디스패치 | Metal kernel JIT + fallback |

#### Metal Kernel 상세

**이름**: gated_delta_step_kernel_{hasmask}_{vectorized}
```
4가지 변형: (has_mask=0/1) x (vectorized=0/1)
입력: q[B,Hq,Dk], k[B,Hkv,Dk], v[B,Hkv,Dv], g[B,Hkv], b[B,Hkv], state[B,Hkv,Dv,Dk]
출력: y[B,Hq,Dv], new_state[B,Hkv,Dv,Dk]
GQA: repeat_factor = Hq / Hkv
state 업데이트: state[h][d][e] = g_val * state[h][d][e] + v_val * k_val * b_val
출력: y[h*repeat][d] = sum_e(state[h][d][e] * q_val)
```

**그리드**: (B, Hkv, Dv) -- 배치 x KV헤드 x V차원
**타입 인스턴스화**: float32 전용 (다른 dtype은 ops fallback)

---

### (D) mlx_vlm/models/cache.py -> mlx_lm/models/cache.py

**소스**: `.repo_cache/mlx-vlm/mlx_vlm/models/cache.py` (210줄) + `.repo_cache/mlx-lm/mlx_lm/models/cache.py` (1606줄)

mlx_vlm은 mlx_lm 캐시를 re-export하고 3개 추가 클래스 정의.

#### KVCache (mlx_lm, Qwen3.5 Attention에서 사용)

```
상수: step = 256 (사전 할당 단위)
속성: keys=None, values=None, offset=0
```

| 메서드 | 동작 |
|--------|------|
| update_and_fetch(k, v) | buffer None이면 할당 (256 단위 올림). keys[..., prev:offset, :] = k in-place 쓰기. keys[..., :offset, :] 반환 |
| trim(n) | offset -= min(offset, n). 데이터 삭제 없이 포인터만 후퇴 |
| make_mask(N, window_size) | create_attention_mask(N, offset=self.offset) 호출 |
| empty() | self.keys is None |

**Shape 규약**: [B, n_kv_heads, seq_len, head_dim] -- seq_len은 axis 2

**C++ 구현 전략**:
```cpp
struct KVCache {
    static constexpr int step = 256;
    mx::array keys = mx::array(0.0f);  // placeholder (no default ctor)
    mx::array values = mx::array(0.0f);
    int offset = 0;
    bool is_empty = true;

    std::pair<mx::array, mx::array> update_and_fetch(const mx::array& k, const mx::array& v);
};
```

#### ArraysCache (mlx_lm, GatedDeltaNet에서 사용)

```
속성: cache = [None] * size  (크기 size의 generic 슬롯 배열)
```

| 메서드 | 동작 |
|--------|------|
| __getitem__[idx] | cache[idx] 반환 |
| __setitem__[idx] | cache[idx] = value |
| make_mask(N) | left_padding -> pos >= left_padding[:, None] / lengths -> pos < lengths[:, None] / else None |
| empty() | cache[0] is None |

**C++ 구현 전략**:
```cpp
struct ArraysCache {
    std::vector<std::optional<mx::array>> cache;
    explicit ArraysCache(int size) : cache(size) {}
    mx::array& operator[](int idx) { return cache[idx].value(); }
    void set(int idx, mx::array val) { cache[idx] = std::move(val); }
};
```

#### mlx_vlm 추가 캐시 (Vision phase에서 필요시 구현)

| 클래스 | 용도 |
|--------|------|
| SimpleKVCache | 단순 concat (사전 할당 없음) |
| SlidingWindowCache | max_size 고정 + 슬라이딩 윈도우 |
| StaticKVCache | max_size 고정 + trim 지원 |

#### mlx_lm 기타 캐시 (Language-only에서는 불필요)

| 클래스 | 용도 |
|--------|------|
| ConcatenateKVCache | concat-only (사전 할당 없음) |
| QuantizedKVCache | mx.quantize로 K/V 양자화 저장 |
| RotatingKVCache | 슬라이딩 윈도우 + keep prefix |
| ChunkedKVCache | front-trim + start_position |
| CacheList | 복합 캐시 래퍼 (KVCache + ArraysCache) |
| BatchKVCache | 배치 지원 + left_padding |
| BatchRotatingKVCache | 배치 + 슬라이딩 윈도우 |
| PromptTrie | 프롬프트 prefix 매칭 Trie |
| LRUPromptCache | LRU + PromptTrie 기반 프롬프트 캐시 |

---

### (E) mlx_vlm/models/qwen3_vl/config.py

**소스**: `.repo_cache/mlx-vlm/mlx_vlm/models/qwen3_vl/config.py` (101줄)

#### VisionConfig (Qwen3.5 Vision Encoder)
```python
@dataclass
class VisionConfig(BaseModelConfig):
    depth: int = 32                          # ViT 레이어 수
    hidden_size: int = 1280                  # 내부 차원
    intermediate_size: int = 3420            # MLP 중간 차원
    out_hidden_size: int = 1536              # merger 출력 차원
    num_heads: int = 16                      # 어텐션 헤드 수
    patch_size: int = 14                     # 공간 패치 크기
    spatial_merge_size: int = 2              # 패치 병합 비율
    temporal_patch_size: int = 2             # 시간 패치 크기 (비디오)
    num_position_embeddings: int = 2304      # 위치 임베딩 수 (48x48)
    window_size: int = 112                   # 윈도우 어텐션
    fullatt_block_indexes: list = [7,15,23,31]  # 전체 어텐션 블록
    deepstack_visual_indexes: list = []      # deepstack 인덱스
```

#### TextConfig
```python
@dataclass
class TextConfig(BaseModelConfig):
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    vocab_size: int
    rope_theta: float
    max_position_embeddings: int
    rope_scaling: Dict  # {"type": "default", "mrope_section": [24, 20, 20]}
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    hidden_act: str = "silu"
```

#### ModelConfig
```python
@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vision_token_id: int = 151654
```

**C++ 전략**: config.h에서 cpu track의 config 구조체 재사용 (이미 구현됨)

---

### (F) mlx_vlm/models/qwen3_vl/processing_qwen3_vl.py

**용도**: HuggingFace Processor wrapping (이미지/비디오 전처리 + tokenization)
**C++ 전략**: 불필요 (text-only 추론에서 tokenizer_runtime으로 대체)

---

### (G) mlx_vlm/models/qwen3_vl/qwen3_vl.py

**소스**: `.repo_cache/mlx-vlm/mlx_vlm/models/qwen3_vl/qwen3_vl.py` (166줄)

| 이름 | 역할 | C++ 전략 |
|------|------|----------|
| masked_scatter(embed, mask, features) | image feature를 text embedding에 삽입 | VLM phase에서 구현 |
| Model | VLM 최상위 모듈: vision_tower + language_model | VLM phase에서 구현 |
| Model.get_input_embeddings() | vision 인코딩 -> embed merge -> MRoPE position 계산 | VLM phase에서 구현 |
| Model.sanitize() | HF weight 키 정리 (model.language_model -> language_model.model) | weight_loader에서 구현 |

---

### (H) mlx_vlm/models/qwen3_vl/vision.py

**소스**: `.repo_cache/mlx-vlm/mlx_vlm/models/qwen3_vl/vision.py` (439줄)

| 클래스/함수 | 역할 | 의존 nn 모듈 |
|------------|------|-------------|
| rotate_half(x) | RoPE 보조: [-x2, x1] 결합 | -- |
| apply_rotary_pos_emb_vision(tensor, freqs) | 2D factored rotary embedding 적용 | -- |
| VisionRotaryEmbedding | inv_freq -> outer(seq, inv_freq) | nn.Module |
| PatchEmbed | Conv3d [temporal, patch, patch] 커널로 3D 패치 임베딩 | nn.Conv3d |
| PatchMerger | spatial merge: LayerNorm -> Linear -> GELU -> Linear | nn.LayerNorm, nn.Linear, nn.GELU |
| Attention (vision) | fused QKV, cu_seqlens 기반 per-subsequence SDPA | nn.Linear |
| MLP (vision) | 2-layer: Linear -> GELU(tanh) -> Linear | nn.Linear, nn.GELU |
| Qwen3VLMoEVisionBlock | pre-norm (LayerNorm) + Attention + MLP residual | nn.LayerNorm |
| VisionModel | 32-block ViT + PatchEmbed + pos_embed (learned) + PatchMerger | nn.Embedding (pos), 위 전체 |
| VisionModel.rot_pos_emb() | 2D factored rotary: H/W 주파수 분리 lookup | -- |
| VisionModel.fast_pos_embed_interpolate() | bilinear interpolation of learned position embedding | -- |

**추가 nn 모듈 (Vision에서만 사용)**:
- nn.Conv3d -- PatchEmbed에서 3D convolution
- nn.LayerNorm -- Vision transformer pre-norm
- nn.GELU -- Vision MLP activation (approx="tanh" variant)
- nn.Upsample -- bilinear interpolation (base.py의 interpolate())

**C++ 전략**: Vision encoder는 별도 Phase (Phase 4+)에서 구현. Language-only + text는 Vision 불필요.

---

## 0e. 사용되는 전체 nn 모듈 종합

### Language track (Phase 2 구현 대상)

| nn 모듈 | Python 경로 | 파라미터 | forward | C++ 합성 |
|---------|------------|----------|---------|----------|
| nn.Module | nn/layers/base.py | dict 기반 | -- | struct |
| nn.Linear | nn/layers/linear.py | weight [out,in], bias [out] | x @ W.T + b | matmul(x, transpose(w)) + b |
| nn.Embedding | nn/layers/embedding.py | weight [V,D] | weight[x] | take(weight, x, 0) |
| nn.Conv1d | nn/layers/convolution.py | weight [O,kW,I/g], bias [O] | conv1d(x,w,...) | conv1d(x,w,...) |
| nn.RMSNorm | nn/layers/normalization.py | weight [D] | fast.rms_norm(x,w,eps) | fast::rms_norm(x,w,eps) |
| nn.silu | nn/layers/activations.py | (없음) | x * sigmoid(x) | x * sigmoid(x) |
| nn.softplus | nn/layers/activations.py | (없음) | log(1+exp(x)) | where(x>20, x, log(1+exp(x))) |

### Vision track (Phase 4+ 구현 대상)

| nn 모듈 | Python 경로 | 비고 |
|---------|------------|------|
| nn.Conv3d | nn/layers/convolution.py | PatchEmbed용 |
| nn.LayerNorm | nn/layers/normalization.py | Vision pre-norm |
| nn.GELU | nn/layers/activations.py | Vision MLP (approx="tanh") |
| nn.Upsample | nn/layers/upsample.py | bilinear interpolation |

---

## 0f. mx.core Python -> C++ API 전체 매핑

### Core 연산 (mlx/ops.h)

| Python | C++ (mlx::core::) | 검증 |
|--------|---------------------|------|
| mx.arange(start, stop, step, dtype) | arange(start, stop, step, dtype) | verified |
| mx.add(a, b) | add(a, b) / operator+ | verified |
| mx.array(data) | array({data}, {shape}, dtype) | verified |
| mx.broadcast_to(a, shape) | broadcast_to(a, shape) | verified |
| mx.concatenate(arrays, axis) | concatenate({a,b,...}, axis) | verified |
| mx.conv1d(x, w, s, p, d, g) | conv1d(x, w, s, p, d, g) | verified |
| mx.cos(x) | cos(x) | verified |
| mx.cumsum(a, axis) | cumsum(a, axis) | verified |
| mx.exp(x) | exp(x) | verified |
| mx.expand_dims(a, axes) | expand_dims(a, axes) | verified |
| mx.flatten(a) | flatten(a) | verified |
| mx.log(x) | log(x) | verified |
| mx.matmul(a, b) | matmul(a, b) | verified |
| mx.ones(shape, dtype) | ones(shape, dtype) | verified |
| mx.ones_like(a) | ones_like(a) | verified |
| mx.random.uniform(lo, hi, shape) | random::uniform(shape, lo, hi, dtype) | verified |
| mx.repeat(a, n, axis) | repeat(a, n, axis) | verified |
| mx.reshape(a, shape) | reshape(a, shape) | verified |
| mx.sigmoid(x) | sigmoid(x) | verified |
| mx.sin(x) | sin(x) | verified |
| mx.split(a, indices, axis) | split(a, indices, axis) | verified |
| mx.stack(arrays, axis) | stack({a,b,...}, axis) | verified |
| mx.sum(a, axis) | sum(a, axis) | verified |
| mx.swapaxes(a, ax1, ax2) | swapaxes(a, ax1, ax2) | verified |
| mx.tile(a, reps) | tile(a, reps) | verified |
| mx.transpose(a, axes) | transpose(a, axes) | verified |
| mx.where(cond, a, b) | where(cond, a, b) | verified |
| mx.zeros(shape, dtype) | zeros(shape, dtype) | verified |
| mx.zeros_like(a) | zeros_like(a) | verified |
| mx.eval(a) | eval(a) | verified |
| x.astype(dtype) | astype(x, dtype) | verified |

### Fast 연산 (mlx/fast.h)

| Python | C++ (mlx::core::fast::) | 검증 |
|--------|--------------------------|------|
| mx.fast.rms_norm(x, w, eps) | rms_norm(x, w, eps) | verified |
| mx.fast.rope(x, dims, traditional, base) | rope(x, dims, traditional, base) | verified |
| mx.fast.scaled_dot_product_attention(Q,K,V,scale,mask) | scaled_dot_product_attention(Q,K,V,scale,mask,mask_mode) | verified |
| mx.fast.metal_kernel(name, inputs, outputs, source) | metal_kernel(name, inputs, outputs, source, ...) | verified |

### I/O

| Python | C++ | 검증 |
|--------|-----|------|
| mx.load_safetensors(path) | load_safetensors(path) -> SafetensorsLoad | verified |

### Dtype

| Python | C++ |
|--------|-----|
| mx.float32 | mlx::core::float32 |
| mx.float16 | mlx::core::float16 |
| mx.bfloat16 | mlx::core::bfloat16 |
| mx.int32 | mlx::core::int32 |
| mx.int64 | mlx::core::int64 |
| mx.uint32 | mlx::core::uint32 |
| mx.bool_ | mlx::core::bool_ |

---

## 0g. libmlx.a 빌드

```bash
cd .repo_cache/mlx
cmake -B build -DMLX_BUILD_METAL=ON -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
```

**빌드 결과**:
- build/libmlx.a (34MB static library)
- build/mlx/backend/metal/kernels/mlx.metallib
- 의존성: Metal.framework, Foundation.framework, Accelerate.framework, QuartzCore.framework

**컴파일 플래그**:
```makefile
CXXFLAGS = -std=c++20 -O2 -I../../.repo_cache/mlx
LDFLAGS  = -L../../.repo_cache/mlx/build -lmlx \
           -framework Metal -framework Foundation \
           -framework Accelerate -framework QuartzCore
MLX_METAL_PATH = ../../.repo_cache/mlx/build/mlx/backend/metal/kernels
```

**런타임 환경변수**: MLX_METAL_PATH 설정 필수 (Metal kernel metallib 경로)

---

## 0h. API 검증 결과

test/test_mlx_api.cpp: **22/22 PASS**

검증된 API:
- mx::default_device() = GPU, Metal available
- mx::array 생성 (float32, float16, int32), zeros, ones, arange
- mx::matmul() -- 정확한 값
- mx::fast::rms_norm() -- 정확한 값
- mx::fast::rope() -- shape + position 0 identity
- mx::fast::scaled_dot_product_attention() -- mask_mode="" (not "none"!)
- mx::conv1d() -- depthwise (groups=C)
- mx::sigmoid(), silu (x*sigmoid(x)), softplus (log(1+exp(x)))
- mx::fast::metal_kernel() -- JIT 컴파일 + 실행 성공

---

## 상태

### 0a. 소스 분석
- [x] 소스 파일 다운로드 (models/qwen3_5_mlx/)
- [x] 전체 import 목록 추출 (6개 파일)
- [x] 의존성 트리 작성

### 0b. 업스트림 의존성 조사
- [x] mlx_vlm/models/base.py 조사 -> LanguageModelOutput, create_attention_mask, scaled_dot_product_attention 등
- [x] mlx_lm/models/base.py 조사 -> create_causal_mask (삼각마스크), create_attention_mask (prefill만 마스크)
- [x] mlx_lm/models/activations.py 조사 -> swiglu = silu(gate) * x
- [x] mlx_lm/models/gated_delta.py 조사 -> Metal kernel 4변형 + ops fallback
- [x] mlx_lm/models/cache.py 조사 -> KVCache (step=256, in-place 쓰기), ArraysCache (generic slot)
- [x] mlx_vlm/models/cache.py 조사 -> re-export + SimpleKVCache, SlidingWindowCache, StaticKVCache
- [x] mlx_vlm/models/qwen3_vl/config.py 조사 -> VisionConfig, TextConfig, ModelConfig
- [x] mlx_vlm/models/qwen3_vl/qwen3_vl.py 조사 -> Model, masked_scatter, sanitize
- [x] mlx_vlm/models/qwen3_vl/vision.py 조사 -> VisionModel (32-block ViT + PatchEmbed + PatchMerger)

### 0c. nn 모듈 구현 조사
- [x] nn.Module 구현 파악 -> dict 기반, C++에서 struct 대체
- [x] nn.Linear 구현 파악 -> weight [out,in], forward x @ W.T + b
- [x] nn.Embedding 구현 파악 -> weight [V,D], forward weight[x]
- [x] nn.Conv1d 구현 파악 -> weight [O,kW,I/g], NLC 입력
- [x] nn.RMSNorm 구현 파악 -> fast.rms_norm(x, weight, eps)
- [x] nn.silu / nn.softplus 구현 파악 -> 직접 합성

### 0d. MLX C++ API 매핑
- [x] Core ops 매핑 (31개)
- [x] Fast ops 매핑 (4개)
- [x] I/O ops 매핑 (load_safetensors)
- [x] Dtype 매핑 (7개)

### 0e. 빌드 & 검증
- [x] libmlx.a 빌드 (34MB)
- [x] API 검증 테스트 (22/22 PASS)
