# cache_utils.py 분석

## 원본 위치
`transformers/cache_utils.py`

## import 사용처
```python
from ...cache_utils import Cache, DynamicCache
```

## 클래스 계층 구조

```
CacheLayerMixin (ABC)           # 단일 레이어 KV 캐시 인터페이스
├── DynamicLayer                # 동적 크기 KV 캐시 (기본)
│   ├── DynamicSlidingWindowLayer  # sliding window 지원
│   └── QuantizedLayer          # 양자화 캐시
├── StaticLayer                 # 고정 크기 (torch.compile용)
│   └── StaticSlidingWindowLayer
│
LinearAttentionCacheLayerMixin (ABC)  # 선형 어텐션용 캐시 인터페이스
├── LinearAttentionLayer            # conv_state + recurrent_state 관리
└── LinearAttentionAndFullAttentionLayer  # 하이브리드 (풀어텐션 + 선형어텐션)

Cache                           # 레이어 목록 컨테이너
├── DynamicCache                # 동적 캐시 (기본, config로 레이어 타입 추론)
├── StaticCache                 # 정적 캐시 (torch.compile용)
├── QuantizedCache              # 양자화 캐시
└── EncoderDecoderCache         # 인코더-디코더용
```

## Qwen3.5에서 사용하는 캐시 타입

Qwen3.5는 `layer_types`에 따라 하이브리드 캐시를 사용:
- `"full_attention"` 레이어 → `DynamicLayer` (일반 KV 캐시)
- `"linear_attention"` 레이어 → `LinearAttentionLayer` (conv + recurrent state)

`DynamicCache.__init__`에서 `config.layer_types` 기반으로 자동 결정됨:
```python
if layer_type in ("mamba", "conv", "linear_attention", "moe"):
    layers.append(LinearAttentionLayer())
elif layer_type == "hybrid":
    layers.append(LinearAttentionAndFullAttentionLayer())
else:
    layers.append(DynamicLayer())
```

## DynamicLayer (KV 캐시) 핵심 로직

```python
def update(self, key_states, value_states):
    # 텐서 shape: [batch, num_heads, seq_len, head_dim]
    if not self.is_initialized:
        self.lazy_initialization(key_states, value_states)
    self.keys = torch.cat([self.keys, key_states], dim=-2)      # seq_len 차원 concat
    self.values = torch.cat([self.values, value_states], dim=-2)
    return self.keys, self.values
```

**C++ 구현 시 주의사항**:
- decode 시 매 토큰마다 `cat` → 비효율적. pre-allocated buffer + pointer 방식 추천
- `get_seq_length()` → `keys.shape[-2]`
- `get_mask_sizes(query_length)` → `(self.get_seq_length() + query_length, 0)`

## LinearAttentionLayer (GatedDeltaNet 캐시) 핵심 로직

conv_states와 recurrent_states 두 가지를 관리:

```python
def update_conv_state(self, conv_states):
    # shape: [batch, channels, conv_kernel_size]
    if not self.has_previous_state:
        self.conv_states.copy_(conv_states)
    else:
        # roll + copy 방식으로 sliding update
        new = self.conv_states.roll(-num_new_tokens, dims=-1)
        new[:, :, -num_new_tokens:] = conv_states
        self.conv_states.copy_(new)
    return self.conv_states

def update_recurrent_state(self, recurrent_states):
    # shape: [batch, num_heads, k_dim, v_dim]
    self.recurrent_states.copy_(recurrent_states)
    return self.recurrent_states
```

## C++ Metal 구현 전략

### KV Cache (full attention 레이어용)
```
struct KVCache {
    // pre-allocated Metal buffer
    MTL::Buffer* keys;    // [batch, num_heads, max_seq, head_dim]
    MTL::Buffer* values;  // [batch, num_heads, max_seq, head_dim]
    int current_length;   // 현재까지 저장된 seq 길이
};
```
- `update()`: `current_length` 위치에 새 KV write → `current_length++`
- attention 계산 시: `[0..current_length]` 범위만 참조

### Linear Attention Cache (GatedDeltaNet 레이어용)
```
struct LinearCache {
    MTL::Buffer* conv_states;       // [batch, conv_dim, kernel_size]
    MTL::Buffer* recurrent_states;  // [batch, num_heads, k_dim, v_dim]
    bool has_previous;
};
```
- `update_conv()`: ring buffer 또는 roll + copy
- `update_recurrent()`: in-place overwrite

### 구현하지 않을 것
- `StaticCache` (torch.compile 전용)
- `QuantizedCache` (Stage 1에서 불필요)
- `EncoderDecoderCache` (Qwen3.5 VLM은 decoder-only)
- `DynamicSlidingWindowLayer` (Qwen3.5에 sliding window 없음)
- offloading / prefetching (단일 GPU)
- beam search reorder (greedy/sampling만)
