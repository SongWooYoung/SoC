# Phase 3: qwen3_5.h — Text Model (bottom-up)

## 목표
`modeling_qwen3_5.py`의 **텍스트 모델** 서브모듈을 C++/Metal로 구현한다.
PyTorch MPS 방식을 그대로 재현하여 output이 일치하는 것이 우선이다.
Vision 모델 및 VLM은 Phase 3v에서 다룬다.

## 원본
- `modeling_qwen3_5.py` (96KB, auto-generated, self-contained)

## 구현 순서 (bottom-up)

### 3a. Qwen3_5RMSNorm
- `x * rsqrt(mean(x^2) + eps)`
- weight는 `(1 + weight)` 형태 (0으로 초기화하여 1-centered)
- inference에서는 단순한 연산

### 3b. Qwen3_5TextRotaryEmbedding (RoPE + MRoPE)
- inv_freq 계산 → cos/sin 테이블 생성
- MRoPE: 3차원 position (temporal, height, width) → interleaved
- `mrope_section: [11, 11, 10]`
- `apply_rotary_pos_emb()` — q, k에 cos/sin 적용
- `partial_rotary_factor: 0.25` — head_dim의 25%만 회전

### 3c. Qwen3_5MLP
- `gate_proj(x)` → SiLU → `* up_proj(x)` → `down_proj()`
- 3개의 Linear layer

### 3d. Qwen3_5Attention (full attention)
- q_proj → (query, gate) split → q_norm
- k_proj → k_norm
- v_proj
- RoPE 적용
- KV cache update
- attention: `softmax(Q @ K^T / sqrt(d)) @ V`
- gated output: `output * sigmoid(gate)`
- o_proj

### 3e. Qwen3_5GatedDeltaNet (linear attention)
- in_proj_qkv, in_proj_z, in_proj_b, in_proj_a
- Conv1d (causal, kernel_size=4)
- gated delta rule:
  - prefill: `chunk_gated_delta_rule` (chunked, O(n) complexity)
  - decode: `fused_recurrent_gated_delta_rule` (single-step recurrent)
- Qwen3_5RMSNormGated (norm before gate)
- out_proj
- **중요**: `modeling_qwen3_5.py`에 fallback 구현이 인라인되어 있음 (`torch_chunk_gated_delta_rule`, `torch_recurrent_gated_delta_rule`)

### 3f. Qwen3_5DecoderLayer
- `layer_type` 분기: `"full_attention"` → 3d, `"linear_attention"` → 3e
- input_layernorm → token mixer → residual add → post_attention_layernorm → MLP → residual add

### 3g. Qwen3_5TextModel
- embed_tokens (Embedding)
- rotary_emb (3b)
- layers[] (3f × num_hidden_layers)
- norm (3a)
- position_ids 생성 (4D: text + temporal + height + width)
- causal mask 생성
- linear attention mask 처리
- KV cache (DynamicCache) 관리

### 3h. Qwen3_5ForCausalLM
- TextModel (3g)
- lm_head (Linear)
- generate loop: prefill → decode → sampling → EOS 확인

## 결과물
- `models/qwen3_5/modeling.h`

## 상태
- [ ] 3a. RMSNorm
- [ ] 3b. RoPE + MRoPE
- [ ] 3c. MLP
- [ ] 3d. Attention
- [ ] 3e. GatedDeltaNet
- [ ] 3f. DecoderLayer
- [ ] 3g. TextModel
- [ ] 3h. ForCausalLM + generate
