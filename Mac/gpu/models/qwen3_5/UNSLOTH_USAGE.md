# Qwen3.5 Usage Notes From Unsloth

This file records the practical usage guidance from:

- https://huggingface.co/unsloth/Qwen3.5-9B-GGUF
- https://unsloth.ai/docs/models/qwen3.5

It exists so `Mac/gpu/models/qwen3_5` keeps a model-local record of prompt format, thinking controls, and sampling guidance.

## Thinking Control

Unsloth documents that the small and mid-size `Qwen3.5` models, including `4B` and `9B`, ship with thinking disabled by default in their recommended chat-template usage.

The documented switch is:

- `enable_thinking=true`
- `enable_thinking=false`

In our runtime today:

- `qwen3_5` defaults to chat-template mode with `enable_thinking=false`
- CLI: `--enable-thinking` opts back into thinking mode

Current mapping:

- Unsloth `enable_thinking=false` -> `gpu_infer --model-type qwen3_5`
- Unsloth `enable_thinking=true` -> `gpu_infer --model-type qwen3_5 --enable-thinking`

Important:

- `Mac/gpu` currently exposes this through prompt/template rendering, not through a model-specific hidden API
- the chat template emitted by GGUF metadata also explicitly branches on `enable_thinking`

## Prompt Shape

The GGUF cards and Unsloth docs keep `<think>` and `</think>` as protected special tokens.
For `enable_thinking=false`, the Hugging Face Qwen3.5 template still emits an empty thinking block before generation.

Representative prompt envelope:

```text
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

Our runtime should continue to preserve:

- `<|im_start|>`
- `<|im_end|>`
- `<think>`
- `</think>`

as protected special tokens in tokenizer/runtime handling.

## Recommended Sampling Settings

Documented Unsloth presets:

### Thinking / General

- `temperature = 1.0`
- `top_p = 0.95`
- `top_k = 20`
- `min_p = 0.0`
- `presence_penalty = 1.5`
- `repetition_penalty = 1.0`

### Thinking / Precise Coding

- `temperature = 0.6`
- `top_p = 0.95`
- `top_k = 20`
- `min_p = 0.0`
- `presence_penalty = 0.0`
- `repetition_penalty = 1.0`

### Non-thinking / General

- `temperature = 0.7`
- `top_p = 0.8`
- `top_k = 20`
- `min_p = 0.0`
- `presence_penalty = 1.5`
- `repetition_penalty = 1.0`

### Non-thinking / Reasoning

- `temperature = 1.0`
- `top_p = 0.95`
- `top_k = 20`
- `min_p = 0.0`
- `presence_penalty = 1.5`
- `repetition_penalty = 1.0`

## Current `Mac/gpu` Support Status

Supported today:

- `enable_thinking` through chat-template handling
- `temperature`
- `top_k`
- benchmark default preset aligned to Unsloth non-thinking/general: `temperature=0.7`, `top_k=20`

Documented but not yet implemented end-to-end in the current runtime:

- `top_p`
- `min_p`
- `presence_penalty`
- `repetition_penalty`

These remain important because they affect practical `Qwen3.5` behavior, especially when comparing against Unsloth, vLLM, or llama.cpp based runs.

## Memory Guidance Mentioned By Unsloth

Unsloth's public guidance states approximate Apple-style memory footprints in this range:

- `Qwen3.5-4B` BF16: around `14 GB`
- `Qwen3.5-9B` BF16: around `19 GB`
- `Qwen3.5-9B` 8-bit: around `13 GB`

For this repo, that supports the current bring-up policy:

- start with HF `4B`
- use quantized `9B` as the first practical path on a `32 GB` Mac mini
- do not make unquantized `9B` the default first target
