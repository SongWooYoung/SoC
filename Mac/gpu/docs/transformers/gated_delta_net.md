# GatedDeltaNet Fallback 구현 분석

## 위치
`modeling_qwen3_5.py` 인라인 (lines 210-355)

## 개요
Qwen3.5는 하이브리드 어텐션: full attention + GatedDeltaNet (linear attention).
`causal_conv1d`와 `fla` 라이브러리가 없으면 순수 PyTorch fallback을 사용.
C++/Metal 구현은 이 fallback을 기반으로 한다.

## 핵심 함수 3개

### 1. `torch_causal_conv1d_update` (decode용)
```python
def torch_causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None):
    # hidden_states: [batch, channels, seq_len=1]
    # conv_state:    [batch, channels, kernel_size]  (ring buffer)
    # weight:        [channels, kernel_size]         (depthwise conv)
    
    concat = cat([conv_state, hidden_states], dim=-1)
    conv_state.copy_(concat[:, :, -state_len:])        # state 업데이트
    out = F.conv1d(concat, weight.unsqueeze(1), bias, groups=channels)
    out = F.silu(out[:, :, -seq_len:])
    return out
```

**Metal 구현**: depthwise conv1d + silu 퓨즈드 커널 1개

### 2. `torch_chunk_gated_delta_rule` (prefill용)
```python
def torch_chunk_gated_delta_rule(query, key, value, g, beta, 
                                  chunk_size=64, initial_state=None,
                                  output_final_state=False, use_qk_l2norm_in_kernel=False):
```

알고리즘 단계:
1. **l2norm**: Q, K에 L2 normalization (use_qk_l2norm_in_kernel=True)
2. **transpose**: [B, seq, heads, dim] → [B, heads, seq, dim]
3. **pad**: 시퀀스를 chunk_size 배수로 패딩
4. **scale**: query *= 1/sqrt(k_dim)
5. **chunk reshape**: [B, H, seq, D] → [B, H, num_chunks, chunk_size, D]
6. **decay mask**: `g.cumsum` → `exp(g[i] - g[j])` → lower triangular
7. **intra-chunk attention**:
   - `attn = -(k_beta @ key.T) * decay_mask`  (chunk 내부 KV 상호작용)
   - triangular solve/fixpoint iteration으로 안정화
   - `value = attn @ v_beta`
   - `k_cumdecay = attn @ (k_beta * g.exp())`
8. **inter-chunk recurrence**: 각 chunk를 순회하며
   - `recurrent_state = recurrent_state * g_decay + k.T @ v_new`
   - `output = q @ recurrent_state + intra_chunk_attn @ v_new`
9. **reshape back**: [B, H, seq, D] → [B, seq, H*D]

**핵심 텐서 크기**:
- query/key: [B, H, seq, k_dim]  (k_dim ≈ 192~256)
- value:     [B, H, seq, v_dim]  (v_dim ≈ 128)
- g, beta:   [B, H, seq]         (scalar per position per head)
- recurrent_state: [B, H, k_dim, v_dim]  (~50KB per head)

### 3. `torch_recurrent_gated_delta_rule` (decode용)
```python
def torch_recurrent_gated_delta_rule(query, key, value, g, beta,
                                      initial_state, output_final_state,
                                      use_qk_l2norm_in_kernel=False):
```

토큰 단위 순환:
```python
for i in range(seq_len):  # seq_len = 1 during decode
    g_t = g[:, :, i].exp()
    beta_t = beta[:, :, i]
    
    # 1. state에 decay 적용
    state = state * g_t
    # 2. state에서 현재 key에 대한 value 읽기
    kv_mem = (state * k_t).sum(dim=-2)  # [B, H, v_dim]
    # 3. 오차 보정 (delta rule)
    delta = (v_t - kv_mem) * beta_t
    # 4. state 업데이트
    state = state + k_t.outer(delta)  # [B, H, k_dim, v_dim]
    # 5. query로 output 읽기
    output = (state * q_t).sum(dim=-2)  # [B, H, v_dim]
```

**Metal 구현**: decode 시 seq_len=1이므로 loop 불필요, 단일 matmul 커널

## Qwen3_5GatedDeltaNet 모듈 구조

```
Input: hidden_states [B, seq, hidden_size]
  │
  ├── in_proj_qkv: Linear(hidden → key_dim*2 + value_dim)  → mixed_qkv
  ├── in_proj_z:   Linear(hidden → value_dim)               → z (gating)
  ├── in_proj_b:   Linear(hidden → num_v_heads)             → beta (sigmoid)
  └── in_proj_a:   Linear(hidden → num_v_heads)             → alpha → g (decay)
  │
  ├── conv1d: DepthwiseConv1d(conv_dim, kernel=4)           → mixed_qkv (after conv)
  │                                                            or causal_conv1d_fn
  ├── split → query, key, value
  ├── reshape → [B, seq, heads, head_dim]
  │
  ├── g = -A_log.exp() * softplus(alpha + dt_bias)          → decay rate
  ├── GQA: repeat_interleave if num_v_heads > num_k_heads
  │
  ├── chunk_gated_delta_rule (prefill)                      → core_attn_out, state
  │   or recurrent_gated_delta_rule (decode)
  │
  ├── norm(core_attn_out, z)                                → RMSNormGated
  └── out_proj: Linear(value_dim → hidden_size)             → output
```

## C++ Metal 구현 전략

### Prefill (chunk_gated_delta_rule)
- 가장 복잡한 부분. chunk_size=64 단위로 처리
- intra-chunk: [64, k_dim] × [k_dim, 64] matmul → Metal matmul kernel
- inter-chunk: recurrent state update → Metal 1D kernel
- decay mask 계산: cumsum + exp → 별도 kernel 또는 prefill의 일부

### Decode (recurrent_gated_delta_rule) 
- seq_len=1이므로 훨씬 단순
- state *= g_decay: elementwise
- kv_mem = sum(state * k, dim=-2): reduction
- delta = (v - kv_mem) * beta: elementwise
- state += outer(k, delta): outer product
- output = sum(state * q, dim=-2): reduction
- → 퓨즈드 Metal kernel 1개로 가능

### Conv1d (causal_conv1d)
- depthwise separable conv: groups = channels
- kernel_size = 4 (매우 작음)
- → decode: conv_state rollshift + dot product 4개
- → prefill: F.conv1d groups=channels → Metal depthwise conv kernel

### RMSNormGated
- `output = rms_norm(x) * silu(z)`로 x와 z를 함께 처리
- standard RMSNorm + elementwise silu multiply

## 수치 주의사항
- `g`, `beta`, `alpha`는 float32로 계산 (fp16 overflow 방지)
  - `g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)`
- Q, K에 L2 norm 적용 (`use_qk_l2norm_in_kernel=True`)
- `l2norm` eps = 1e-6
