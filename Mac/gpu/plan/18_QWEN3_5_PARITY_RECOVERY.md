# Qwen3.5 Parity Recovery Plan

This document changes the `qwen3_5` priority from throughput to output quality.

Current status:

- `qwen3_5` can load HF `Qwen3.5-4B`
- prompt formatting is now aligned to the HF / Unsloth non-thinking template
- output quality is still poor
- the current implementation is a hybrid runner with large portions of model math reimplemented manually in [`qwen3_5_runner.cpp`](/Volumes/990pro/Documents/SoC/Mac/gpu/models/qwen3_5/qwen3_5_runner.cpp)

That means the immediate goal is not "make it faster".
The immediate goal is "make it mathematically behave like `transformers` first".

## Ground Truth

Reference implementation:

- local `transformers` source:
  - [`modeling_qwen3_5.py`](/Volumes/990pro/Documents/SoC/.venv/lib/python3.14/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py)

Important reference sections:

- `Qwen3_5GatedDeltaNet`
- `Qwen3_5Attention`
- `Qwen3_5RMSNorm`
- `Qwen3_5RMSNormGated`
- `Qwen3_5DecoderLayer`

## Why Parity Is Currently Fragile

The current runner is not a thin execution wrapper around reference operators.
It reimplements several numerically sensitive pieces inline:

- `Qwen3_5RmsNorm`
- `Qwen3_5RmsNormGated`
- `ApplyPartialRoPE`
- `CausalDepthwiseConv1dSiLU`
- `ComputeDeltaDecay`
- DeltaNet recurrent update
- manual query/gate splitting
- manual head repetition and state layout

Those are all inside [`qwen3_5_runner.cpp`](/Volumes/990pro/Documents/SoC/Mac/gpu/models/qwen3_5/qwen3_5_runner.cpp).

## Top Suspected Correctness Risks

Ordered by likelihood.

### 1. DeltaNet recurrent rule mismatch

Evidence:

- `transformers` uses `chunk_gated_delta_rule` / `fused_recurrent_gated_delta_rule` with `use_qk_l2norm_in_kernel=True`
- current C++ runner manually normalizes Q/K, manually computes `g`, manually decays state, manually updates recurrent state
- current decode path has separate prompt and decode implementations, increasing drift risk

Relevant code:

- HF:
  - `Qwen3_5GatedDeltaNet.forward`
  - `torch_chunk_gated_delta_rule`
  - `torch_recurrent_gated_delta_rule`
- C++:
  - `ForwardGatedDeltaNetPrompt`
  - `ForwardGatedDeltaNetDecode`

Expected failure signature:

- first token often plausible
- multi-token drift increases quickly
- repeated chat markers and structural corruption appear after a few decode steps

### 2. Gated RMSNorm contract mismatch

Evidence:

- HF `Qwen3_5RMSNorm` uses `weight=zeros`, then multiplies by `(1 + weight)`
- HF `Qwen3_5RMSNormGated` applies norm, then `weight * hidden`, then `F.silu(gate)`
- current C++ `Qwen3_5RmsNormGated` computes `gate / (1 + exp(-gate))`, which matches `silu(gate)`, but it is still hand-rolled and could drift due to ordering / dtype
- a previous note already says shared `RmsNormOp` broke parity for `qwen3_5`

Relevant code:

- HF:
  - `Qwen3_5RMSNorm.forward`
  - `Qwen3_5RMSNormGated.forward`
- C++:
  - `Qwen3_5RmsNorm`
  - `Qwen3_5RmsNormGated`

### 3. Full-attention path mismatch

Evidence:

- HF attention uses `q_proj -> split(query, gate) -> q_norm/k_norm -> rotary -> cache.update -> attention interface -> sigmoid(gate) -> o_proj`
- current C++ attention path is manually reconstructed with host-side grouped attention, hand-rolled partial rotary, manual cache append, manual gate sigmoid

Relevant code:

- HF:
  - `Qwen3_5Attention.forward`
- C++:
  - `ForwardGatedAttentionPrompt`
  - `ForwardGatedAttentionDecode`

### 4. mRoPE / RoPE application mismatch

Evidence:

- HF uses `Qwen3_5TextRotaryEmbedding` and `apply_rotary_pos_emb`
- current C++ uses `ApplyPartialRoPE` with a simplified rotary path
- `Qwen3.5` has `partial_rotary_factor=0.25`
- HF also applies `mrope_interleaved` and `mrope_section`
- the current C++ helper does not clearly mirror that contract

Expected failure signature:

- attention layers diverge even when linear-attention layers are disabled or bypassed

### 5. Conv1d path mismatch in DeltaNet

Evidence:

- HF uses `causal_conv1d_fn` or `F.silu(self.conv1d(...))` with sequence/cached update behavior
- current C++ does its own `CausalDepthwiseConv1dSiLU` and separate history management via `BuildDeltaConvSequence` / `UpdateDeltaConvHistory`

Expected failure signature:

- prompt and decode differ even when recurrent update is bypassed
- one-token decode path diverges from full prompt path

### 6. Prefill structure mismatch

Evidence:

- HF does a true whole-sequence prefill pass
- the current C++ `qwen3_5` path mixes prompt-time custom math and decode-time state logic
- reviewer feedback identified this as a likely structural correctness issue even before optimization

Expected failure signature:

- first-token parity can look acceptable on a tiny prompt
- longer prompts already diverge before long decode
- prompt-only layer outputs mismatch even when sampler is removed

### 7. Hybrid execution structure itself

Evidence:

- current C++ `qwen3_5` path mixes CPU reference math, GPU projections, GPU MLP, GPU LM head, and custom cache logic
- that is not how HF `transformers` structures the model
- broad GPU projection currently changes outputs a lot, which suggests the current hybrid boundaries are not mathematically stable enough

## Working Hypothesis

The main problem is not the sampler.
The main problem is that `qwen3_5` is currently a partial handwritten port of the model math, especially around DeltaNet and attention.

Sampling can mask or amplify output drift, but it is not the root cause.

## Recovery Strategy

### Phase 1. Make parity measurable

Add two tools:

1. `transformers` reference probe
   - dump per-layer intermediate tensors for a fixed prompt
   - include:
     - embedding output
     - input layernorm output
     - token-mixer output
     - post-attention layernorm output
     - MLP output
     - hidden state after residual
     - final norm
     - logits

2. C++ parity dump mode
   - dump the same intermediate tensors for the same prompt
   - begin with small prompt and first token only

Success criterion:

- we can identify the first layer and the first sub-stage where C++ diverges materially from HF

### Phase 2. Isolate the first wrong stage

Run ablations in this order:

1. prompt-only, first-token, one layer
2. prompt-only, first-token, all layers
3. decode single-step from cached prompt
4. multi-token decode

At each stage, compare:

- hidden-state cosine similarity
- max absolute diff
- top-1 logit agreement
- top-10 overlap

### Phase 3. Replace handwritten math with transformers-closer structure

If the first divergence is in DeltaNet:

- rebuild `qwen3_5` around a more direct translation of the HF `Qwen3_5GatedDeltaNet.forward` contract
- unify prompt and decode through one shared logical path instead of separate prompt/decode math functions

If the first divergence is in full attention:

- rebuild the attention branch to mirror HF ordering exactly
- avoid separate reinterpretations of Q/gate splitting, RMSNorm ordering, and rotary layout

If the first divergence is already present during prompt prefill:

- stop optimizing decode
- rebuild prompt execution to match the HF layer contract first

## Restart Criterion

Restarting from a transformers-closer structure is justified if either of these is true:

- the first divergence occurs before the first layer finishes, or
- multiple early handwritten helper functions each show independent mismatch

That would mean the current partial port is too bespoke, and patching individual symptoms will be slower than rebuilding `qwen3_5` around the HF layer contract.

## Immediate Experiments

These are the first required experiments.

1. `transformers` prompt-only layer dump for `Hello world`
2. C++ prompt-only layer dump for the same prompt
3. compare first divergence among:
   - embedding
   - input RMSNorm
   - token mixer
   - post-attention RMSNorm
   - MLP
   - final norm
   - logits
4. DeltaNet-only isolation:
   - run only first linear-attention layer and compare recurrent update outputs
5. Attention-only isolation:
   - run only first full-attention layer and compare attention outputs before `o_proj`
6. Position-sensitivity isolation:
   - same suffix, different prefix lengths
   - compare first full-attention layer outputs to catch `mRoPE` drift
7. True-prefill isolation:
   - compare whole-prompt logits before decode
   - if this diverges, fix prefill structure before more GPU work

## Explicit Non-Goals For This Phase

- throughput
- scheduler tuning
- q4/q8 optimization
- GGUF parity
- SSD offloading

All of those come after output quality recovery.
