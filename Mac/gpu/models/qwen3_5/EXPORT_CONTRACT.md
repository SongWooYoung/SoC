# Qwen3.5 Export Contract

This document defines the first `Mac/gpu` export and manifest contract for `Qwen3.5-9B`.

## Source of Truth

- Primary source: Hugging Face `Qwen/Qwen3.5-9B` safetensors + `config.json`
- Export format: existing `Mac/gpu` manifest bundle format produced by `LLM_interpreter/convert_py_to_cpp.py`
- Tensor names: keep raw Hugging Face tensor names in the manifest

Reason:

- the metadata-only loader can validate the bundle before any GPU weight upload
- architecture bring-up stays separate from later tensor repacking or quantization

## Manifest Requirements

The manifest must include:

- top-level `config`
- full `text_config` when present in the HF config
- `tensors[]` records that keep original tensor names

The manifest does not need `qwen3_5`-specific extra fields yet if `config` already carries:

- `model_type`
- `architectures`
- `text_config`
- `layer_types`
- `rope_parameters`
- `mamba_ssm_dtype`

## Required Common Tensor Names

Accepted aliases:

- embedding:
  - `model.language_model.embed_tokens.weight`
  - `model.embed_tokens.weight`
  - `embed_tokens.weight`
- final norm:
  - `model.language_model.norm.weight`
  - `model.norm.weight`
  - `norm.weight`
- lm head:
  - `lm_head.weight`
  - `model.language_model.lm_head.weight`
  - `model.lm_head.weight`

## Required Per-Layer Families

Shared on all layers:

- `input_layernorm.weight`
- `post_attention_layernorm.weight`
- `mlp.gate_proj.weight`
- `mlp.up_proj.weight`
- `mlp.down_proj.weight`

For full-attention layers:

- `self_attn.q_proj.weight`
- `self_attn.k_proj.weight`
- `self_attn.v_proj.weight`
- `self_attn.o_proj.weight`
- `self_attn.q_norm.weight`
- `self_attn.k_norm.weight`

For gated-deltanet layers:

- `linear_attn.norm.weight`
- `linear_attn.in_proj_qkv.weight`
- `linear_attn.in_proj_z.weight`
- `linear_attn.in_proj_a.weight`
- `linear_attn.in_proj_b.weight`
- `linear_attn.out_proj.weight`
- `linear_attn.conv1d.weight`
- `linear_attn.A_log`
- `linear_attn.dt_bias`

## Prefix Aliases

The metadata-only loader accepts these layer prefixes:

- `model.language_model.layers.{i}.`
- `model.layers.{i}.`
- `layers.{i}.`

This is intentional. It keeps the loader compatible with:

- full multimodal family exports
- text-only stripped exports
- future internal manifest normalization

## Shape Rules Checked in the Metadata-Only Loader

For `Qwen3.5-9B`, the current validator enforces:

- embeddings: `[vocab_size, hidden_size]`
- final norm: `[hidden_size]`
- lm head: `[vocab_size, hidden_size]`
- attention q_proj: `[num_attention_heads * head_dim * 2, hidden_size]`
- attention k_proj/v_proj: `[num_key_value_heads * head_dim, hidden_size]`
- attention o_proj: `[hidden_size, num_attention_heads * head_dim]`
- attention q_norm/k_norm: `[head_dim]`
- deltanet norm: `[linear_value_head_dim]`
- deltanet in_proj_qkv: `[linear_key_dim * 2 + linear_value_dim, hidden_size]`
- deltanet in_proj_z: `[linear_value_dim, hidden_size]`
- deltanet in_proj_a/in_proj_b: `[linear_num_value_heads, hidden_size]`
- deltanet out_proj: `[hidden_size, linear_value_dim]`
- deltanet conv1d weight: `[linear_qkv_dim, 1, linear_conv_kernel_dim]`
- deltanet A_log/dt_bias: `[linear_num_value_heads]`

## What This Contract Deliberately Does Not Do Yet

- no GPU buffer allocation
- no actual weight upload
- no quantized repacking
- no MTP execution support
- no SSD offloading semantics

This stage exists only to prove:

- the export bundle is structurally usable
- the tensor naming contract is stable
- loader failures can happen before Metal execution starts
