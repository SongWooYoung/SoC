# Qwen3.5 GGUF Import Notes

This file records the current `GGUF` observations for `Qwen3.5-9B`, using:

- `bartowski/Qwen_Qwen3.5-9B-GGUF`
- local file: `Qwen_Qwen3.5-9B-Q8_0.gguf`

The goal is to keep the `GGUF` path explicitly separated from the primary HF-first loader contract.

## Current Local File

- file: `Qwen_Qwen3.5-9B-Q8_0.gguf`
- size on disk: about `8.9G`
- GGUF tensor count: `427`
- extracted manifest: `models/cpp/qwen3_5-9b-gguf-q8_0/manifest.json`
- extracted tokenizer runtime: `models/cpp/qwen3_5-9b-gguf-q8_0/tokenizer/tokenizer_runtime.json`
- extracted GGUF metadata dump: `models/cpp/qwen3_5-9b-gguf-q8_0/gguf_metadata.json`

## GGUF Metadata Observed

Key fields seen in the file:

- `general.architecture = qwen35`
- `general.file_type = 7`
- `qwen35.block_count = 32`
- `qwen35.context_length = 262144`
- `qwen35.embedding_length = 4096`
- `qwen35.feed_forward_length = 12288`
- `qwen35.attention.head_count = 16`
- `qwen35.attention.head_count_kv = 4`
- `qwen35.attention.key_length = 256`
- `qwen35.attention.value_length = 256`
- `qwen35.rope.dimension_count = 64`
- `qwen35.rope.dimension_sections = [11, 11, 10, 0]`
- `qwen35.full_attention_interval = 4`
- `qwen35.ssm.conv_kernel = 4`
- `qwen35.ssm.state_size = 128`
- `qwen35.ssm.group_count = 16`
- `qwen35.ssm.time_step_rank = 32`

Tokenizer metadata is also present inside the GGUF:

- `tokenizer.ggml.tokens`
- `tokenizer.ggml.merges`
- `tokenizer.chat_template`
- `tokenizer.ggml.eos_token_id`
- `tokenizer.ggml.padding_token_id`

## Tensor Naming Difference Vs HF Export

The current HF export path uses names like:

- `model.language_model.layers.3.self_attn.q_proj.weight`
- `model.language_model.layers.0.linear_attn.in_proj_qkv.weight`

The current GGUF file instead exposes names like:

- `output.weight`
- `output_norm.weight`
- `token_embd.weight`
- `blk.0.attn_gate.weight`
- `blk.0.attn_norm.weight`
- `blk.0.attn_qkv.weight`
- `blk.0.ssm_a`
- `blk.0.ssm_alpha.weight`
- `blk.0.ssm_beta.weight`
- `blk.0.ssm_conv1d.weight`
- `blk.0.ssm_dt.bias`
- `blk.0.ssm_norm.weight`
- `blk.0.ssm_out.weight`
- `blk.0.ffn_gate.weight`
- `blk.0.ffn_up.weight`
- `blk.0.ffn_down.weight`

This is not a cosmetic difference.

It means:

- HF-style metadata validation cannot be assumed to work on GGUF-derived manifests
- the GGUF adapter must preserve original naming and quantization metadata
- a future quantized `qwen3_5` loader may need a dedicated GGUF-family tensor resolver

There is also a layout difference:

- many GGUF matrices are stored in the opposite orientation relative to the current HF export contract
- example:
  - HF `mlp.down_proj.weight` expects `[hidden, intermediate]`
  - GGUF `ffn_down.weight` is observed as `[intermediate, hidden]`

So the current GGUF path is blocked by both:

- naming differences
- transpose/layout differences

## Current Repo Policy

- HF safetensors/config remain the correctness source of truth
- GGUF import is experimental and adapter-scoped
- GGUF extraction is still useful for:
  - tokenizer extraction
  - file-offset based manifests
  - RAM-friendlier bundle construction
- quantized weight catalog inspection

## Experimental Reverse Conversion

There is now an experimental reverse-conversion helper in:

- `LLM_interpreter/convert_gguf_to_safetensors.py`

Scope:

- `Qwen3.5` only
- `Q8_0` and `F32` tensors only
- explicit GGUF-name -> HF-name mapping
- per-tensor shape validation before writing safetensors
- optional smoke conversion with `--max-tensors`

Current policy:

- use it for metadata validation and small tensor smoke tests first
- do not treat it as a generic GGUF importer
- do not treat the resulting safetensors as correctness-equivalent until runtime parity is proven

## Current Validation Status

The current `gpu_infer --validate-only` result for the GGUF-derived manifest is:

- failure at metadata resolution
- first reported missing tensor: `embed_tokens`

This is expected under the current code because the qwen3_5 metadata loader still targets the HF naming contract first.

## Bartowski Card Notes

The bartowski README documents:

- prompt format with `<think>`
- `Q8_0` size around `9.55GB`
- that `ssm_alpha.weight` and `ssm_beta.weight` were updated to `F32`

That last point matters because recurrent/state tensors may keep higher-precision subpaths even when the bulk weights are quantized.
