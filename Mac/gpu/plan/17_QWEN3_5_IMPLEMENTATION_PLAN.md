# Qwen3.5 Implementation Plan

## Goal

`Mac/gpu`에 `qwen3`와 별도로 `qwen3_5` 모델 계열을 수용할 수 있는 실행 경로를 만든다.
첫 목표 모델은 `Qwen3.5-9B`이며, bring-up은 correctness 우선, 최적화는 그 다음 단계로 둔다.

## Source Strategy

### Recommended source of truth

- Primary: official Hugging Face `Qwen/Qwen3.5-9B` safetensors/config
- Secondary reference: `transformers` `Qwen3_5TextConfig`
- Performance/quantization reference: MLX quantized conversions
- Cross-check only: GGUF Q8_0 / Q4_K* cards

### Why not use GGUF as the primary source

- GGUF is useful for deployment and cross-checking layer metadata, but it is already a transformed representation.
- `Mac/gpu` currently owns its own manifest/export/runtime path, so model bring-up should start from the least transformed source.
- `Qwen3.5` has architecture-specific semantics that are easier to recover from HF config + weights than from GGUF alone.

### Why MLX matters

- MLX is a strong Apple Silicon performance reference.
- MLX quantized exports help identify useful low-bit weight layouts and fused matmul directions.
- MLX should influence later optimization, not the first correctness bring-up.

## Known Architecture Facts

Based on the official Hugging Face `Qwen/Qwen3.5-9B` config/README and the `transformers` `Qwen3_5TextConfig` docs:

- 9B parameter language model with vision-capable family packaging
- Hidden size: `4096`
- FFN intermediate size: `12288`
- Layer count: `32`
- Gated attention heads: `16` query heads, `4` KV heads
- Gated attention head dimension: `256`
- Gated DeltaNet heads:
  - `16` QK heads
  - `32` V heads
  - linear key/value head dimension `128`
- Hidden layout:
  - `8 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))`
- RoPE:
  - shipped `config.json` exposes `max_position_embeddings=262144`
  - `rope_parameters.mrope_section=[11,11,10]`
  - `rope_parameters.partial_rotary_factor=0.25`
  - `mamba_ssm_dtype=float32`

Implication:

- `qwen3` block execution cannot be reused as-is.
- `qwen3_5` needs two distinct sequence operators:
  - linear recurrent block (`Gated DeltaNet`)
  - periodic full attention block (`Gated Attention`)

## Reuse vs. New Work

### Reusable from `qwen3`

- `MetalContext`
- `PipelineCache`
- `BufferArena`
- `DeviceTensor`
- `KVCache` storage primitives
- generic `ModelRunner` interface
- generic `GenerationContext`
- generic `CommandScheduler`
- generic `MetalProfilingMode`
- shared ops where applicable:
  - `RmsNormOp`
  - `MatMulOp`
  - `AffineQmmOp`
  - `LinearOp`
  - `SoftmaxOp`
  - `RoPEOp` for the gated-attention layers only

### New or substantially different for `qwen3_5`

- model loader schema
- block graph
- per-layer type routing
- recurrent linear-state storage
- gated delta block execution
- attention/linear mixed schedule
- decode planner
- prompt-cache serialization format

## Data Source Decision for the First Bring-up

### Phase 1

Use official HF `Qwen/Qwen3.5-9B` as the bring-up source.

Deliverable:

- new exporter path or manifest translation path that captures:
  - hidden layout
  - layer types
  - Gated DeltaNet dimensions
  - Gated Attention dimensions
  - FFN dimensions
  - raw HF tensor names in manifest records so the metadata-only loader can validate the bundle before any GPU load

### Phase 2

Use MLX quantized conversions as optimization reference for:

- low-bit weight packing
- decode-time fused qmm design
- Apple Silicon memory layout choices

### Phase 3

Optionally add GGUF import support if:

- there is a strong need to ingest existing GGUF releases directly
- or loader bring-up from HF proves too expensive

Current direction:

- keep HF safetensors/config as the primary correctness path
- add `GGUF -> soc.cpp manifest` only as an experimental adapter
- do not let GGUF naming/quantization details redefine the HF-first loader contract

### Phase 4

Optionally add a RAM-capped offload mode after correctness if the unquantized 9B checkpoint is too heavy for the target machine.

Constraints:

- `flash-moe` is compelling, but its main win comes from sparse MoE expert activation.
- `Qwen3.5-9B` is dense, so a per-token SSD streaming path would need to touch most of the model far more often.
- `flash-moe` also documents that SSD DMA and GPU compute do not overlap profitably on Apple Silicon because both contend for the memory controller.

Implication:

- do not make SSD-GPU per-token streaming the primary bring-up path for dense `Qwen3.5-9B`
- if RAM pressure is unacceptable on a 32GB Mac mini, consider a later optional loader mode with:
  - mmap-backed cold weight storage
  - staged host windows
  - explicit RAM-cap policy
  - no assumption of profitable SSD/GPU overlap
  - no per-token dense SSD replay as the default execution model

## Execution Plan

### Step 1. Finalize scaffold

- `models/qwen3_5/ARCHITECTURE.md`
- `models/qwen3_5/qwen3_5_loader_adapter.*`
- `models/qwen3_5/qwen3_5_runner.*`
- `models/qwen3_5/modules/*`
- `models/qwen3_5/test/*`

Exit criteria:

- codebase compiles
- registry recognizes `qwen3_5`
- runner surface exists even if execution remains unimplemented

### Step 2. Add architecture spec layer

Introduce an explicit `Qwen3_5ArchitectureSpec` that describes:

- model size / dims
- layer layout
- gated-attention parameters
- gated-deltanet parameters
- rope/runtime config
- recurrent state layout requirements

Exit criteria:

- manifest parsing can produce a `Qwen3_5ArchitectureSpec`
- `qwen3_5` runner can expose metadata from spec
- metadata-only validation can prove the manifest contains the required tensor catalog and shapes without touching Metal buffers

### Step 3. Loader bring-up

Implement manifest/config parsing and GPU tensor loading for:

- embeddings
- final norm
- lm head
- per-layer FFN weights
- per-layer Gated DeltaNet weights
- per-layer Gated Attention weights

Exit criteria:

- loader succeeds on a real `Qwen3.5-9B` export bundle

Current staging update:

- real `Qwen3.5-4B` HF export bundle already exists and passes `--validate-only`
- this 4B path is the current bring-up vehicle for:
  - manifest/spec validation
  - tokenizer/runtime validation
  - load-time profiling
  - future first-token correctness
- `Qwen3.5-9B` remains the next scale-up target after the 4B path is stable

Before real weight loading:

- implement `config.json/manifest -> Qwen3_5ArchitectureSpec`
- define `Qwen3_5StateLayout` so runner, loader, and future planner share one recurrent-state contract
- carry recurrent state dtype and element width in the layout object instead of assuming float32 in every caller
- implement a metadata-only loader that validates:
  - required common tensors
  - per-layer attention vs deltanet tensor families
  - required shapes for stable tensors
  - alias mapping for full `model.language_model.*` exports vs stripped text-only exports

Additional staging work:

- add an experimental GGUF adapter that can:
  - inspect a GGUF file
  - build a soc.cpp manifest with `file_offset`
  - reverse-convert a narrow `Qwen3.5 Q8_0/F32` tensor subset into HF-style safetensors for metadata-first validation
  - keep tensor naming/orientation fixes outside the primary HF-first loader path
  - emit tokenizer runtime data from GGUF metadata
  - preserve GGUF tensor names and quantization metadata
- do not treat that adapter as proof that the native HF/qwen3_5 loader is complete

### Step 4. CPU parity scaffold

Before optimizing Metal execution, add:

- first-token parity tests
- prompt serialization / tokenizer parity checks
- layer-layout validation tests

Exit criteria:

- loader + config + first-token path are stable

### Step 5. GPU execution bring-up

Implement:

- `Qwen3_5GatedDeltaNet`
- `Qwen3_5GatedAttention`
- `Qwen3_5MLP`
- `Qwen3_5Block`
- `Qwen3_5CausalLM`

Execution rules:

- linear layers route to recurrent/state-space path
- every 4th block routes to full attention path
- FFN always follows the main sequence operator

Exit criteria:

- `--layer 0` smoke
- full GPU first-token parity
- minimal decode pass

### Step 6. State management

Add model-specific state storage for:

- recurrent linear state
- attention KV cache

Do not try to force both into the existing `qwen3` decode planner.

Exit criteria:

- prefill/decode reuse works across multiple generated tokens

Initial state layout target:

- one float32 recurrent matrix state per linear-attention head:
  - `[linear_num_key_heads, linear_key_head_dim, linear_value_head_dim]`
- one float32 convolution history buffer per linear-attention layer:
  - `[linear_num_value_heads, linear_value_head_dim, linear_conv_kernel_dim]`

### Step 7. Optimization

Only after correctness:

- evaluate low-bit decode paths
- compare against MLX quantized behavior
- design a `qwen3_5`-specific decode planner

## Current Practical Bring-up Sequence

1. HF `Qwen3.5-4B`
   - download
   - export with `LLM_interpreter`
   - validate with `gpu_infer --validate-only`
   - collect load / manifest / tokenizer timing
   - compare against PyTorch MPS baseline
2. HF `Qwen3.5-4B`
   - implement actual `qwen3_5` forward path
   - obtain first-token correctness and real tok/s
3. Experimental GGUF path
   - ingest `bartowski/Qwen_Qwen3.5-9B-GGUF` `Q8_0`
   - extract tensor catalog, tokenizer, file metadata
   - record naming and shape differences against HF export
   - only then decide whether a native GGUF loader path is worth keeping
4. MLX path
   - add extraction only after the HF runtime path is stable
   - use it mainly for Apple Silicon low-bit layout comparisons

## RAM / Offload Position

For a `32GB` Mac mini, native `Qwen3.5-9B` BF16 is possible but not comfortable once the rest of the system is active.
Because of that:

- first bring-up favors `Qwen3.5-4B` HF
- `9B` should prefer a quantized experimental path before any dense BF16 default path
- SSD offloading remains a later optional mode, not the first loader implementation

## Known Emerging Risk

Actual GGUF tensor naming for `Qwen3.5-9B` does not mirror the current HF-style metadata contract one-to-one.
That means:

- the GGUF adapter must document and preserve the original tensor families
- the qwen3_5 loader must not assume that HF tensor names and GGUF tensor names are the same thing
- profiling a GGUF-derived manifest may stop at metadata/load validation until a quantized weight path exists
- separately optimize:
  - full attention layers
  - Gated DeltaNet layers
  - FFN

## Initial Risks

### Risk 1. Incorrect assumption that `qwen3_5` is “just qwen3 with different sizes”

This is false.
The layer semantics differ.

### Risk 2. Multimodal top-level packaging

`Qwen3.5-9B` ships as a top-level conditional generation model with nested `text_config` and `vision_config`.
`Mac/gpu` should intentionally bring up the text path first and ignore vision execution until the text stack is stable.

### Risk 3. Quant format temptation too early

If bring-up starts from MLX or GGUF first, architecture debugging will get mixed with format conversion debugging.

### Risk 4. Reusing `qwen3` planner

The `qwen3` decode planner assumes homogeneous transformer blocks.
`qwen3_5` should get its own planner.

## Success Criteria

### Bring-up success

- `qwen3_5` manifest resolves through registry
- loader creates a runner with correct metadata
- basic inference path can initialize and fail only on intentionally unimplemented operators

### Correctness success

- first-token parity against CPU/reference path
- stable multi-token decode

### Performance success

- quantify Apple Silicon performance against MLX and PyTorch reference paths
- explicitly decide whether any later RAM-cap/offload path is worth building for dense 9B
- decide whether `qwen3_5` needs:
  - low-bit decode first
  - linear-state optimization first
  - scheduler work first
