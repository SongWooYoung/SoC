# Qwen3.5-9B Architecture Notes

This document records the currently known `Qwen3.5-9B` architecture facts that matter for `Mac/gpu`.
It is the working reference for the `qwen3_5` loader, runner, modules, and tests.

## Sources

- Hugging Face `transformers` `Qwen3_5TextConfig` docs:
  - https://huggingface.co/docs/transformers/model_doc/qwen3_5
- Official Hugging Face `Qwen/Qwen3.5-9B` README:
  - https://huggingface.co/Qwen/Qwen3.5-9B/blame/main/README.md
- MLX converted model card used only as a secondary consistency check:
  - https://huggingface.co/Brooooooklyn/Qwen3.5-9B-unsloth-mlx

## Confirmed Core Dimensions

From the `transformers` `Qwen3_5TextConfig` docs:

- `vocab_size = 248320`
- `hidden_size = 4096`
- `intermediate_size = 12288`
- `num_hidden_layers = 32`
- `num_attention_heads = 16`
- `num_key_value_heads = 4`
- `head_dim = 256`
- `hidden_act = silu`
- `rms_norm_eps = 1e-6`
- `linear_conv_kernel_dim = 4`
- `linear_key_head_dim = 128`
- `linear_value_head_dim = 128`
- `linear_num_key_heads = 16`
- `linear_num_value_heads = 32`

## Layer Layout

From the official `Qwen/Qwen3.5-9B` README:

- hidden layout is:
  - `8 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))`

That implies:

- total layers: `32`
- every 4-layer group contains:
  - 3 linear/recurrent layers
  - 1 full attention layer

A practical execution interpretation is:

- layers `0,1,2`: Gated DeltaNet blocks
- layer `3`: Gated Attention block
- repeat every 4 layers

This pattern must be encoded explicitly in the `qwen3_5` architecture spec.

## Attention vs. Linear Blocks

### Gated Attention

From the official README:

- Q heads: `16`
- KV heads: `4`
- head dim: `256`
- rotary position embedding dim: `64`

This is closer to a conventional grouped-query attention path and can reuse more of the current `qwen3` attention/operator stack.

### Gated DeltaNet

From the official README and `transformers` config:

- linear attention heads:
  - `16` QK heads
  - `32` V heads
- linear key/value head dimensions:
  - `128`
- `linear_conv_kernel_dim = 4`

Implication:

- this is not standard softmax attention
- it likely needs model-specific recurrent/state-space execution and state storage
- the current `qwen3` KV-cache-only decode model is not sufficient by itself

## Context Length and RoPE

The actual current `Qwen/Qwen3.5-9B` `config.json` exposes:

- `max_position_embeddings = 262144`
- `rope_parameters.mrope_interleaved = true`
- `rope_parameters.mrope_section = [11, 11, 10]`
- `rope_parameters.rope_theta = 10000000`
- `rope_parameters.partial_rotary_factor = 0.25`

The official README also states:

- context length `262,144` natively
- extensible up to `1,010,000`

Working interpretation:

- the currently shipped config and the README agree on the native `262,144` target
- runtime should still treat long-context behavior as rope/runtime mediated rather than as a simplistic single constant
- `partial_rotary_factor = 0.25` over `head_dim = 256` implies a rotary dimension of `64`

## What This Means for `Mac/gpu`

### Reusable from `qwen3`

- embeddings
- RMSNorm
- FFN matmul paths
- LM head
- grouped-query attention operator pieces for the gated-attention layers
- profiling/runtime scaffolding

### Not reusable as-is

- block scheduling
- homogeneous per-layer assumptions
- decode planner
- cache/state semantics

### New model-specific pieces required

- `Qwen3_5ArchitectureSpec`
- `Qwen3_5StateLayout`
- layer-type routing
- `Qwen3_5GatedDeltaNet` module
- `Qwen3_5GatedAttention` module
- recurrent linear-state storage
- `qwen3_5`-specific decode planner

## Weight Source Recommendation

Recommended primary source for bring-up:

- official HF `Qwen/Qwen3.5-9B` safetensors/config

Recommended secondary source for optimization ideas:

- MLX quantized conversions

Recommended role for GGUF:

- only as a validation/cross-check source during early implementation

Reason:

- bring-up should start from the least transformed representation
- quantized or converted formats are better used after architecture bring-up is stable

## State Layout Notes

Before real Metal execution, `Mac/gpu` should treat the linear-attention state as separate from the full-attention KV cache.

Current working layout:

- each Gated DeltaNet layer owns a recurrent matrix state:
  - `[linear_num_key_heads, linear_key_head_dim, linear_value_head_dim]`
- for `Qwen3.5-9B`:
  - `[16, 128, 128]`
- state dtype should default to `float32`, matching `mamba_ssm_dtype`

Each Gated DeltaNet layer should also own a convolution history buffer:

- `[linear_num_value_heads, linear_value_head_dim, linear_conv_kernel_dim]`
- for `Qwen3.5-9B`:
  - `[32, 128, 4]`

This is the initial loader/runner contract. The exact update equations can refine it later.

Important contract detail:

- the state layout object should carry the recurrent state dtype and byte width explicitly
- `Qwen3.5-9B` currently points to `mamba_ssm_dtype=float32`, so the first implementation still allocates float32 state
- callers should not hardcode `sizeof(float)` if they consume `Qwen3_5StateLayout`

## SSD Offloading Assessment

`danveloper/flash-moe` is a strong Apple Silicon reference, but it should not be copied directly as the first strategy for `Qwen3.5-9B`.

Why:

- `flash-moe` wins because it streams only the active MoE experts from SSD.
- `Qwen3.5-9B` is dense, so a comparable dense per-token SSD streaming path would need to move much more data every token.
- `flash-moe` also reports that SSD DMA and GPU compute do not overlap profitably on Apple Silicon because they share the memory controller.

Implication for `Mac/gpu`:

- do not make per-token SSD streaming the primary path for dense `Qwen3.5-9B`
- if RAM pressure is too high on a 32GB Mac mini, treat offloading as a later optional mode for:
  - mmap-backed cold weights
  - staged host-memory windows
  - RAM-capped loader policy
  - no default assumption that dense SSD reads can be overlapped profitably with GPU execution

That is a separate bring-up track, not the first implementation path.

## Immediate Implementation Guidance

### Loader

The loader should first recover:

- layer count
- layer pattern
- gated-attention sizes
- gated-deltanet sizes
- FFN sizes
- rope-related config

Before any Metal upload, the `qwen3_5` path should also validate a metadata-only tensor catalog:

- common tensors:
  - `embed_tokens.weight`
  - `norm.weight`
  - `lm_head.weight`
- full-attention layers:
  - `self_attn.{q_proj,k_proj,v_proj,o_proj}.weight`
  - `self_attn.{q_norm,k_norm}.weight`
- gated-deltanet layers:
  - `linear_attn.{norm,in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}.weight`
  - `linear_attn.conv1d.weight`
  - `linear_attn.A_log`
  - `linear_attn.dt_bias`
- shared per-layer tensors:
  - `input_layernorm.weight`
  - `post_attention_layernorm.weight`
  - `mlp.{gate_proj,up_proj,down_proj}.weight`

The validator should accept alias prefixes for:

- `model.language_model.layers.*`
- `model.layers.*`
- `layers.*`

### Runner

The runner should expose metadata immediately even before execution is implemented:

- layer count
- hidden size
- vocab size
- max position / rope config placeholder

### Modules

Initial module scaffold should separate:

- `qwen3_5_gated_deltanet`
- `qwen3_5_gated_attention`
- `qwen3_5_mlp`
- `qwen3_5_block`

### Tests

The first test layer should validate:

- architecture spec defaults
- layer pattern
- manifest recognition
- runner metadata surface
