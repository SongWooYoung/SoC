# MLX→C++ Port — Phase 2: Core Modules 구현

## 목표
language.py + gated_delta.py의 모든 모듈을 MLX C++ API로 구현한다.

## 빌드 설정

```makefile
# Makefile에 추가 필요
MLX_ROOT = ../../.repo_cache/mlx
CXX_FLAGS = -std=c++20 -O2
MLX_INC = -I$(MLX_ROOT)
MLX_LIB = -L$(MLX_ROOT)/build -lmlx
FRAMEWORKS = -framework Metal -framework Foundation -framework Accelerate -framework QuartzCore
METALLIB = $(MLX_ROOT)/build/mlx/backend/metal/kernels/mlx.metallib
```

## 공통 헬퍼 (mlx_helpers.h)

```cpp
#include <mlx/mlx.h>
namespace mx = mlx::core;

// nn.silu(x) = x * sigmoid(x)
inline mx::array silu(const mx::array& x) {
    return x * mx::sigmoid(x);
}

// nn.softplus(x) = log(1 + exp(x))
inline mx::array softplus(const mx::array& x) {
    return mx::log(1.0f + mx::exp(x));
}

// swiglu(gate, x) = silu(gate) * x
inline mx::array swiglu(const mx::array& gate, const mx::array& x) {
    return silu(gate) * x;
}

// nn.Linear forward: matmul(x, w.T) + bias
inline mx::array linear(const mx::array& x, const mx::array& w,
                         const std::optional<mx::array>& bias = std::nullopt) {
    auto out = mx::matmul(x, mx::transpose(w));
    if (bias) out = out + *bias;
    return out;
}

// nn.Embedding forward: take(weight, indices, axis=0)
inline mx::array embedding(const mx::array& weight, const mx::array& indices) {
    return mx::take(weight, indices, 0);
}
```

---

## 2.1 Qwen3_5RotaryEmbedding

**Python**: `language.py` Qwen3_5RotaryEmbedding

```
파라미터:
  dim: int (= head_dim * partial_rotary_factor)
  base: 10000
  mrope_section: [11, 11, 0]

가중치: 없음 (inv_freq는 계산값)

Forward:
  Input: x (any shape, dtype 참조용), position_ids [3, B, S] or [B, S]
  Output: (cos, sin) — 각각 [1, S, dim]
```

**C++ 구현 계획**:
```cpp
struct RotaryEmbedding {
    int dim;
    float base;
    std::vector<int> mrope_section; // [11, 11, 0]
    mx::array inv_freq; // shape [dim/2]

    RotaryEmbedding(int dim, float base=10000, std::vector<int> mrope_section={11,11,0})
        : dim(dim), base(base), mrope_section(mrope_section) {
        auto arange = mx::arange(0, dim, 2, mx::float32);
        inv_freq = 1.0f / mx::power(mx::array(base), arange / float(dim));
    }

    // position_ids: [3, B, S] or [B, S]
    // returns: (cos, sin) each [1, S, dim]
    std::pair<mx::array, mx::array> operator()(mx::Dtype dtype, const mx::array& position_ids);
};
```

**핵심 로직**:
1. position_ids를 [3, B, S]로 확장 (2D면 [1,B,S] → [3,B,S] expand)
2. inv_freq_expanded = inv_freq.reshape(1, dim/2, 1) 브로드캐스트
3. freqs = matmul(inv_freq_expanded, position_ids_expanded) → [3, S, dim/2]
4. apply_interleaved_mrope: mrope_section=[11,11,0]에 따라 3개 컴포넌트를 11,11,0 dim씩 분배 후 interleave
5. emb = concatenate(freqs, freqs) → [1, S, dim]
6. return (cos(emb).astype(dtype), sin(emb).astype(dtype))

**Note**: `apply_interleaved_mrope`는 position 축을 따라 section별로 interleave. 텍스트 전용이면 position_ids = [S] 1D → [3,1,S] 브로드캐스트.

---

## 2.2 Qwen3_5RMSNormGated

**Python**: `language.py` Qwen3_5RMSNormGated

```
가중치: weight [hidden_size] — 초기값 ones

Forward:
  Input: hidden_states [B, S, D], gate [B, S, D] (optional)
  Output: [B, S, D]

연산:
  x = mx.fast.rms_norm(hidden_states, weight, eps)
  if gate: x = swiglu(gate, x)
  return x.astype(hidden_states.dtype)
```

**C++ 구현**:
```cpp
struct RMSNormGated {
    mx::array weight; // [D]
    float eps;

    mx::array operator()(const mx::array& x,
                          const std::optional<mx::array>& gate = std::nullopt) {
        auto out = mx::fast::rms_norm(x, weight, eps);
        if (gate) out = swiglu(*gate, out);
        return mx::astype(out, x.dtype());
    }
};
```

---

## 2.3 Qwen3_5MLP

**Python**: `language.py` Qwen3_5MLP

```
가중치:
  gate_proj: [intermediate_size, hidden_size] (bias=False)
  up_proj:   [intermediate_size, hidden_size] (bias=False)
  down_proj: [hidden_size, intermediate_size] (bias=False)

Forward:
  return down_proj(swiglu(gate_proj(x), up_proj(x)))
```

**C++ 구현**:
```cpp
struct MLP {
    mx::array gate_proj_w, up_proj_w, down_proj_w;

    mx::array operator()(const mx::array& x) {
        auto gate = linear(x, gate_proj_w);
        auto up = linear(x, up_proj_w);
        return linear(swiglu(gate, up), down_proj_w);
    }
};
```

---

## 2.4 Qwen3_5Attention (Full Attention 레이어)

**Python**: `language.py` Qwen3_5Attention

```
Config에서:
  num_attention_heads, num_key_value_heads, head_dim
  scale = head_dim ** -0.5
  attention_bias: bool
  partial_rotary_factor: float

가중치:
  q_proj: [num_heads * head_dim * 2, hidden_size]  ← 2배! (gate용)
  k_proj: [num_kv_heads * head_dim, hidden_size]
  v_proj: [num_kv_heads * head_dim, hidden_size]
  o_proj: [hidden_size, num_heads * head_dim]
  q_norm: RMSNorm(head_dim)
  k_norm: RMSNorm(head_dim)
  rotary_emb: RotaryEmbedding

Forward(x, mask, cache, position_ids):
  1. q_out = q_proj(x) → [B, S, num_heads*head_dim*2]
  2. split → queries [B, S, num_heads, head_dim], gate [B, S, num_heads*head_dim]
  3. keys = k_proj(x).reshape(B, S, num_kv_heads, head_dim)
  4. values = v_proj(x).reshape(B, S, num_kv_heads, head_dim)
  5. queries = q_norm(queries), keys = k_norm(keys)
  6. transpose(B,H,S,D)
  7. cos, sin = rotary_emb(values, position_ids)
  8. apply_multimodal_rotary_pos_emb(q, k, cos, sin) — partial RoPE
  9. cache.update(keys, values) → keys, values (with history)
  10. output = scaled_dot_product_attention(q, k, v, scale, mask)
  11. transpose back → [B, S, num_heads*head_dim]
  12. return o_proj(output * sigmoid(gate))  ← output gating!
```

**핵심 포인트**:
- q_proj 출력이 2× → 절반은 attention query, 절반은 output gate (sigmoid)
- partial RoPE: dim * partial_rotary_factor 까지만 회전, 나머지는 그대로
- GQA: num_kv_heads < num_attention_heads → SDPA에서 자동 broadcast

**C++ 구현**:
```cpp
struct Attention {
    int num_heads, num_kv_heads, head_dim;
    float scale;
    mx::array q_proj_w, k_proj_w, v_proj_w, o_proj_w;
    mx::array q_norm_w, k_norm_w;
    float norm_eps;
    RotaryEmbedding rotary_emb;
    // optional bias arrays

    mx::array operator()(const mx::array& x, const std::optional<mx::array>& mask,
                          KVCache* cache, const mx::array& position_ids) {
        auto B = x.shape(0), S = x.shape(1);

        // 1. Q projection → split into queries + gate
        auto q_out = linear(x, q_proj_w);
        auto queries = slice(q_out, ..., 0, num_heads*head_dim);
        auto gate = slice(q_out, ..., num_heads*head_dim, end);
        queries = reshape(queries, {B, S, num_heads, head_dim});

        // 2. K, V projections
        auto keys = reshape(linear(x, k_proj_w), {B, S, num_kv_heads, head_dim});
        auto values = reshape(linear(x, v_proj_w), {B, S, num_kv_heads, head_dim});

        // 3. Normalize Q, K
        queries = mx::fast::rms_norm(queries, q_norm_w, norm_eps);
        keys = mx::fast::rms_norm(keys, k_norm_w, norm_eps);

        // 4. Transpose → [B, H, S, D]
        queries = transpose(queries, {0, 2, 1, 3});
        keys = transpose(keys, {0, 2, 1, 3});
        values = transpose(values, {0, 2, 1, 3});

        // 5. RoPE (partial)
        auto [cos, sin] = rotary_emb(x.dtype(), position_ids);
        apply_rotary(queries, keys, cos, sin);

        // 6. KV cache update
        if (cache) std::tie(keys, values) = cache->update(keys, values);

        // 7. SDPA
        auto output = mx::fast::scaled_dot_product_attention(
            queries, keys, values, scale, "none", mask);

        // 8. Transpose back + output gating
        output = transpose(output, {0, 2, 1, 3});
        output = reshape(output, {B, S, num_heads * head_dim});
        return linear(output * sigmoid(gate), o_proj_w);
    }
};
```

---

## 2.5 KVCache

**Python**: mlx_lm cache — `update_and_fetch(keys, values)`

```cpp
struct KVCache {
    mx::array keys, values;   // accumulated [B, H, total_S, D]
    int offset = 0;

    std::pair<mx::array, mx::array> update(const mx::array& new_k, const mx::array& new_v) {
        if (offset == 0) {
            keys = new_k;
            values = new_v;
        } else {
            keys = mx::concatenate({keys, new_k}, /*axis=*/2);
            values = mx::concatenate({values, new_v}, /*axis=*/2);
        }
        offset += new_k.shape(2);
        return {keys, values};
    }
};
```

---

## 2.6 Qwen3_5GatedDeltaNet (핵심)

**Python**: `language.py` Qwen3_5GatedDeltaNet + `gated_delta.py`

```
Config에서:
  num_v_heads = linear_num_value_heads   (16)
  num_k_heads = linear_num_key_heads     (4)
  head_k_dim  = linear_key_head_dim      (128)
  head_v_dim  = linear_value_head_dim    (256)
  conv_kernel_size = linear_conv_kernel_dim (4)
  key_dim = head_k_dim * num_k_heads     (512)
  value_dim = head_v_dim * num_v_heads   (4096)
  conv_dim = key_dim * 2 + value_dim     (5120)

가중치:
  conv1d_w: [conv_dim, 1, kernel_size] (depthwise, groups=conv_dim)
  in_proj_qkv_w: [key_dim*2 + value_dim, hidden_size]
  in_proj_z_w: [value_dim, hidden_size]
  in_proj_b_w: [num_v_heads, hidden_size]
  in_proj_a_w: [num_v_heads, hidden_size]
  dt_bias: [num_v_heads] — 학습 파라미터
  A_log: [num_v_heads] — log-domain state matrix
  norm: RMSNormGated(head_v_dim)
  out_proj_w: [hidden_size, value_dim]
```

### Forward 상세

```
Input: inputs [B, S, hidden_size], mask [B, S] optional, cache
Output: [B, S, hidden_size]

1. mixed_qkv = in_proj_qkv(inputs)           → [B, S, key_dim*2+value_dim]
2. z = in_proj_z(inputs).reshape(B,S,Hv,Dv)  → [B, S, 16, 256]
3. b = in_proj_b(inputs)                      → [B, S, 16]
4. a = in_proj_a(inputs)                      → [B, S, 16]

5. Conv1d with state management:
   conv_state from cache → [B, kernel_size-1, conv_dim]  (= [B, 3, 5120])
   if mask: mixed_qkv *= mask.unsqueeze(-1)
   conv_input = concat([conv_state, mixed_qkv], axis=1) → [B, S+3, 5120]
   cache[0] = conv_input[:, -(kernel_size-1):]  → update state
   conv_out = silu(conv1d(conv_input))           → [B, S, 5120]

6. Split conv_out → q [B,S,Hk,Dk], k [B,S,Hk,Dk], v [B,S,Hv,Dv]
   q = conv_out[..., :key_dim].reshape(B,S,Hk,Dk)          → [B,S,4,128]
   k = conv_out[..., key_dim:key_dim*2].reshape(B,S,Hk,Dk) → [B,S,4,128]
   v = conv_out[..., key_dim*2:].reshape(B,S,Hv,Dv)        → [B,S,16,256]

7. Q/K Normalization (asymmetric):
   inv_scale = Dk ** -0.5  (= 1/sqrt(128) ≈ 0.0884)
   q = (inv_scale**2) * rms_norm(q, None, 1e-6)  → scale by 1/Dk
   k = inv_scale * rms_norm(k, None, 1e-6)        → scale by 1/sqrt(Dk)
   NOTE: rms_norm(x, None, eps) = RMSNorm without learnable weight

8. Gated delta update:
   beta = sigmoid(b)                       → [B, S, Hv]
   g = exp(-exp(A_log) * softplus(a + dt_bias))  → [B, S, Hv]
   state from cache                        → [B, Hv, Dv, Dk] = [B,16,256,128]

   For each timestep t=0..S-1:
     state = state * g[:,t,:,None,None]    → decay
     kv_mem = (state * k[:,t,:,None,:]).sum(-1) → [B,Hv,Dv]
     delta = (v[:,t] - kv_mem) * beta[:,t,:,None]
     state = state + k[:,t,:,None,:] * delta[:,:,:,None]
     y[:,t] = (state * q[:,t,:,None,:]).sum(-1)  → [B,Hv,Dv]

   OR Metal kernel (inference): grid=(32, Dv, B*Hv), threadgroup=(32,4,1)

9. cache[1] = state (updated)
10. out = norm(y, z)  → RMSNormGated with z as gate → swiglu(z, rms_norm(y))
11. return out_proj(out.reshape(B, S, value_dim))
```

### Metal Kernel 구현

**4 variants**: `gated_delta_step[_vec][_mask]`

```cpp
// C++ 호출 패턴 (metal_kernel JIT):
auto gd_kernel = mx::fast::metal_kernel(
    kernel_name,                    // "gated_delta_step" etc.
    {"q","k","v","g","beta","state_in","T"[,"mask"]},  // input names
    {"y","state_out"},              // output names
    metal_source,                   // ~50줄 Metal 코드
    "",                             // header
    true                            // ensure_row_contiguous
);

auto [y, new_state] = gd_kernel(
    {q, k, v, g, beta, state, mx::array(T)},  // inputs
    {{B, T, Hv, Dv}, state.shape()},           // output shapes
    {input_dtype, state_dtype},                 // output dtypes
    {32, Dv, B*Hv},                            // grid
    {32, 4, 1},                                // threadgroup
    {{"InT", input_dtype}, {"StT", state_dtype},
     {"Dk", Dk}, {"Dv", Dv}, {"Hk", Hk}, {"Hv", Hv}}  // template args
);
```

### ArraysCache (GDN용)

```cpp
struct ArraysCache {
    mx::array conv_state;      // [B, kernel_size-1, conv_dim]
    mx::array recurrent_state; // [B, Hv, Dv, Dk]
    bool initialized = false;
};
```

---

## 2.7 Qwen3_5DecoderLayer

**Python**: `language.py` Qwen3_5DecoderLayer

```
is_linear = (layer_idx + 1) % full_attention_interval != 0

가중치:
  linear_attn: GatedDeltaNet (if is_linear)
  self_attn: Attention (if !is_linear)
  input_layernorm: RMSNorm(hidden_size)
  post_attention_layernorm: RMSNorm(hidden_size)
  mlp: MLP(hidden_size, intermediate_size)

Forward:
  if is_linear:
    r = linear_attn(input_layernorm(x), mask, cache)
  else:
    r = self_attn(input_layernorm(x), mask, cache, position_ids)
  h = x + r
  return h + mlp(post_attention_layernorm(h))
```

**C++ 구현**:
```cpp
struct DecoderLayer {
    bool is_linear;
    std::unique_ptr<GatedDeltaNet> linear_attn; // is_linear인 경우
    std::unique_ptr<Attention> self_attn;         // !is_linear인 경우
    mx::array input_ln_w, post_ln_w;
    float norm_eps;
    MLP mlp;

    mx::array operator()(const mx::array& x, const std::optional<mx::array>& mask,
                          void* cache, const mx::array& position_ids) {
        auto normed = mx::fast::rms_norm(x, input_ln_w, norm_eps);
        mx::array r;
        if (is_linear) {
            r = (*linear_attn)(normed, mask, static_cast<ArraysCache*>(cache));
        } else {
            r = (*self_attn)(normed, mask, static_cast<KVCache*>(cache), position_ids);
        }
        auto h = x + r;
        auto mlp_out = mlp(mx::fast::rms_norm(h, post_ln_w, norm_eps));
        return h + mlp_out;
    }
};
```

---

## 레이어 구성 (Qwen3.5-4B, full_attention_interval=4)

| layer_idx | (idx+1)%4 | type | 캐시 타입 |
|-----------|-----------|------|-----------|
| 0 | 1 | GDN (linear) | ArraysCache |
| 1 | 2 | GDN (linear) | ArraysCache |
| 2 | 3 | GDN (linear) | ArraysCache |
| 3 | 0 | Attention (full) | KVCache |
| 4 | 1 | GDN (linear) | ArraysCache |
| ... | ... | ... | ... |
| 35 | 0 | Attention (full) | KVCache |

**총 36 layers**: 27 GDN + 9 Full Attention

---

## 가중치 로딩

MLX safetensors → MLX array 직접 로드:

```cpp
// safetensors 파일에서 key → mx::array 매핑
// weight key 패턴:
//   language_model.model.layers.{i}.linear_attn.conv1d.weight → [conv_dim, 1, kernel_size]
//   language_model.model.layers.{i}.self_attn.q_proj.weight → [num_heads*head_dim*2, hidden_size]
//   language_model.model.layers.{i}.mlp.gate_proj.weight → [intermediate_size, hidden_size]
//   language_model.model.embed_tokens.weight → [vocab_size, hidden_size]
//   language_model.model.norm.weight → [hidden_size]
//   language_model.lm_head.weight → [vocab_size, hidden_size]

// sanitize 변환 (qwen3_5.py):
//   conv1d.weight: shape [D,1,K] 이면 transpose → 확인 필요
//   norm.weight: += 1.0 (RMSNorm offset)
```

---

## 상태
- [ ] 2.0 Makefile 업데이트 (C++20, MLX 링크)
- [ ] 2.1 mlx_helpers.h (silu, softplus, swiglu, linear, embedding)
- [ ] 2.2 RotaryEmbedding (interleaved MRoPE)
- [ ] 2.3 RMSNormGated
- [ ] 2.4 MLP (SwiGLU)
- [ ] 2.5 Attention (QK norm, gate, partial RoPE, SDPA)
- [ ] 2.6 KVCache
- [ ] 2.7 GatedDeltaNet (Conv1d + gated_delta_update)
- [ ] 2.8 Metal kernel JIT (gated_delta_step × 4 variants)
- [ ] 2.9 ArraysCache
- [ ] 2.10 DecoderLayer
- [ ] 2.11 가중치 로더 (safetensors → MLX array)
- [ ] 2.T 모듈별 단위 테스트
