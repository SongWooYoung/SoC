# Qwen3.5 Bring-up Status

This file tracks the current real-machine bring-up state for `qwen3_5`.

## 1. HF `Qwen3.5-4B`

### Download and Export

Completed with `LLM_interpreter/model_downloader.py`:

- raw snapshot dir: `models/raw/qwen3_5-4b`
- exported bundle dir: `models/cpp/qwen3_5-4b`

### `Mac/gpu` Validation

Command:

```bash
cd /Volumes/990pro/Documents/SoC/Mac/gpu
./build/bin/gpu_infer \
  --manifest /Volumes/990pro/Documents/SoC/models/cpp/qwen3_5-4b/manifest.json \
  --model-type qwen3_5 \
  --validate-only \
  --json
```

Current result:

- `validate-only`: success
- device: `Apple M4`
- layers: `32`
- hidden size: `2560`
- KV heads: `4`
- vocab size: `248320`
- manifest tensor count: `738`
- exported tensor bytes: `9,319,730,176`
- loaded weights: `true`
- recurrent state bytes: `26,738,688`
- real GPU load wall time:
  - cold-ish run observed around `13.1 s`
  - warm run observed around `3.1 - 3.2 s`

Saved report:

- `Mac/gpu/reports/qwen3_5_4b_validate_only.json`
- `Mac/gpu/reports/qwen3_5_4b_validate_only_loaded.json`

### PyTorch MPS Baseline

Command was run against the local HF snapshot on MPS fp16.

Current baseline:

- load time: `12.921 s`
- prefill: `208.8 ms`
- decode: `6.68 tok/s`
- peak MPS memory: `8473.4 MB`

Saved report:

- `Mac/gpu/reports/qwen3_5_4b_pytorch_mps.json`

### Native `Mac/gpu` Prompt Forward

Current status:

- planner-less transient prompt forward is implemented
- real GPU bundle load still happens before execution
- execution path is currently CPU-reference math inside the `qwen3_5` runner, then logits are written back to the runtime output buffer
- decode caching is still not implemented

Command:

```bash
cd /Volumes/990pro/Documents/SoC/Mac/gpu
./build/bin/gpu_infer \
  --manifest /Volumes/990pro/Documents/SoC/models/cpp/qwen3_5-4b/manifest.json \
  --model-type qwen3_5 \
  --prompt "Hello world" \
  --max-new-tokens 1 \
  --profiling summary \
  --json
```

Current result:

- prompt token ids: `[9419, 1814]`
- deterministic argmax result with `--temperature 0 --top-k 1`:
  - generated token id: `0`
  - decoded token: `"!"`
- default manifest sampling can emit a different token because generation config is not argmax-only
- current hybrid path:
  - host math still owns `Gated DeltaNet` state update and attention core math
  - GPU path is active for embedding, prompt projections, MLP, and LM head
- latest measured timing on the real M4 with argmax flags:
  - prefill wall: `4876.5 ms`
  - prefill tok/s: `0.410`
  - `gpu_ms = 186.6`
  - `wait_ms = 392.6`
  - `command_buffer_count = 258`

Saved report:

- `Mac/gpu/reports/qwen3_5_4b_first_token_gpu.json`

### First-Token Correctness Against PyTorch

Reference prompt:

- prompt: `"Hello world"`
- prompt token ids: `[9419, 1814]`

PyTorch MPS fp16 argmax result:

- next token id: `0`
- decoded token: `"!"`
- prefill wall: `4915.7 ms`
- prefill tok/s: `0.407`

Saved report:

- `Mac/gpu/reports/qwen3_5_4b_first_token_pytorch.json`

Current conclusion:

- native `Mac/gpu` prompt forward now exists
- first-token correctness is matched on the reference prompt when sampler randomness is disabled with `--temperature 0 --top-k 1`
- `gpu_ms` is now real and non-zero on the HF 4B bring-up path
- the remaining gap is still dominated by host-side recurrent/state math and missing decode cache reuse
- one failed experiment is now recorded:
  - moving `post_attention RMSNorm` and `final RMSNorm` onto the shared `RmsNormOp` broke first-token parity for `qwen3_5`
  - the host helper currently applies `1 + weight`, while the shared GPU kernel multiplies by `weight` directly
  - that path is reverted from the default runner until a `qwen3_5`-compatible RMSNorm contract is implemented

### Native `Mac/gpu` Decode Cache Bring-up

Current status:

- `qwen3_5` now has a minimal mixed decode runtime state inside the runner
- full-attention layers keep CPU-side KV history
- `Gated DeltaNet` layers keep CPU-side recurrent matrix state and conv history
- current decode remains hybrid:
  - attention core math and DeltaNet state update stay on CPU
  - `MLP` and `LM head` still use the existing GPU operator path
- this is not wired into generic prompt-cache artifact serialization yet; it is only enough to make native decode run end-to-end

Command:

```bash
cd /Volumes/990pro/Documents/SoC/Mac/gpu
./build/bin/gpu_infer \
  --manifest /Volumes/990pro/Documents/SoC/models/cpp/qwen3_5-4b/manifest.json \
  --model-type qwen3_5 \
  --prompt "Hello world" \
  --max-new-tokens 4 \
  --temperature 0 \
  --top-k 1 \
  --profiling summary \
  --json
```

Current result:

- generated token ids: `[0, 271, 9419, 11]`
- decoded text: `"!\n\nHello,"`
- prefill wall: `4907.6 ms`
- decode wall: `7037.5 ms`
- decode throughput: `0.568 tok/s`
- `gpu_ms = 459.7`
- `wait_ms = 924.4`
- `command_buffer_count = 645`

Saved report:

- `Mac/gpu/reports/qwen3_5_4b_4tok_gpu.json`

So the next work is:

1. optimize the new mixed decode cache path without losing argmax parity
2. move more decode-hot math from CPU into native GPU operators, starting with parity-safe projection slices
3. only after that resume GGUF quantized resolver work

### Projection GPU Drift Narrowing

Recent real-machine findings:

- `SOC_GPU_ENABLE_EXPERIMENTAL_QWEN3_5_DECODE_EMBEDDING_GPU=1`
- `SOC_GPU_ENABLE_EXPERIMENTAL_QWEN3_5_DECODE_OUTPUT_PROJECTION_GPU=1`

preserved argmax parity on the reference prompt but did not improve decode throughput:

- report: `Mac/gpu/reports/qwen3_5_4b_4tok_gpu_decodeproj.json`
- generated token ids: `[0, 271, 9419, 11]`
- decoded text: `"!\n\nHello,"`
- prefill wall: `6051.9 ms`
- decode wall: `7800.5 ms`
- decode throughput: `0.513 tok/s`

By contrast, the broad experimental projection path:

- `SOC_GPU_ENABLE_EXPERIMENTAL_QWEN3_5_PROJECTION_GPU=1`

currently preserves argmax parity on the same prompt and is much faster:

- report: `Mac/gpu/reports/qwen3_5_4b_4tok_gpu_broadproj.json`
- generated token ids: `[0, 271, 9419, 11]`
- decoded text: `"!\n\nHello,"`
- prefill wall: `484.0 ms`
- decode wall: `647.7 ms`
- decode throughput: `6.176 tok/s`
- `gpu_ms = 685.7`
- `wait_ms = 1052.4`
- `command_buffer_count = 1405`

Interpretation:

- the original projection drift is no longer explained by “all GPU projection is always wrong”
- on the current HF 4B bring-up path, the broad projection path can preserve argmax parity on the reference prompt
- the narrower split introduced for `SOC_GPU_ENABLE_EXPERIMENTAL_QWEN3_5_SINGLE_TOKEN_PROJECTION_GPU=1` has not yet reproduced the throughput gain from the broad flag and needs further tracing before promotion

## 2. GGUF `Qwen3.5-9B-Q8_0`

### Download

Downloaded:

- repo: `bartowski/Qwen_Qwen3.5-9B-GGUF`
- file: `Qwen_Qwen3.5-9B-Q8_0.gguf`

Local dir:

- `models/raw/qwen3_5_9b_gguf_q8_0`

### Extraction

Experimental extraction succeeded with:

- `LLM_interpreter/convert_gguf_to_cpp.py`

Outputs:

- `models/cpp/qwen3_5-9b-gguf-q8_0/manifest.json`
- `models/cpp/qwen3_5-9b-gguf-q8_0/tokenizer/tokenizer_runtime.json`
- `models/cpp/qwen3_5-9b-gguf-q8_0/gguf_metadata.json`

### `Mac/gpu` Validation

Command:

```bash
cd /Volumes/990pro/Documents/SoC/Mac/gpu
./build/bin/gpu_infer \
  --manifest /Volumes/990pro/Documents/SoC/models/cpp/qwen3_5-9b-gguf-q8_0/manifest.json \
  --model-type qwen3_5 \
  --validate-only \
  --json
```

Current result:

- failure
- error: `missing required tensor for embed_tokens`

### Why It Fails

The current `qwen3_5` metadata loader assumes HF-style tensor names and HF-style matrix orientation.

The GGUF file currently exposes:

- different tensor names
- quantized tensor dtypes
- transposed matrix layout for many weights

So the current GGUF path is useful for:

- tokenizer extraction
- file metadata extraction
- tensor catalog inspection

but not yet for actual `qwen3_5` runner validation or profiling.

## 3. Immediate Next Step

Before any real GGUF quantized profiling for `qwen3_5`, the next required work is:

1. replace the current host fallback with native GPU operators for the validated HF 4B path
2. implement cached decode for `qwen3_5`
3. then add GGUF-specific tensor resolver / quantized loader support
