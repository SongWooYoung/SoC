# Decode Follow-Up

- Base trace: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan_0/operator_hook_test/base_stage_trace.json
- Compiled-ops trace: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/gated_delta_kernel_followup/compiled_ops_stage_trace.json
- Metal-kernel trace: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/metal_kernel_stage_trace.json
- No-trace 20-prompt comparison: /Volumes/990pro/Documents/SoC/Mac/gpu/test/result/qwen3_5_9b_mlx_lib_vs_custom.json
- Base-vs-custom decode gap report: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/base_custom_decode_gap.md
- Base split trace: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/base_stage_trace_split.json
- Custom split trace: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/custom_stage_trace_split.json
- Base projection microbench: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/base_decode_projection_microbench.json
- Custom projection microbench: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/custom_decode_projection_microbench.json
- Projection/attention follow-up: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/projection_attention_followup.md
- Compiled-ops linear-attention microbench: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/linear_attention_compiled_microbench.json
- Metal-kernel linear-attention microbench: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/linear_attention_metal_microbench.json

## Experiment 1: Model-Core Trace After Metal Kernel Port

| Backend | Avg Prefill ms | Avg Decode ms/tok | Avg Wall ms | Avg Tok/s |
|--------|---------------:|------------------:|------------:|----------:|
| base trace | 514.607 | 115.162 | 4200.318 | 7.670 |
| custom compiled_ops trace | 466.022 | 151.419 | 5311.723 | 6.031 |
| custom metal_kernel trace | 337.394 | 143.549 | 4931.232 | 6.494 |

- Metal kernel keeps the prefill win and improves decode over compiled_ops, but decode still stays about `+28.387 ms/tok` behind the base trace.
- The metal-kernel port therefore fixed a real part of decode, but it did not remove the main remaining decode gap by itself.

## Experiment 2: Runner Benefit Versus Model-Core Gap

| Backend | Trace Decode ms/tok | No-trace Decode ms/tok | Runner Benefit |
|--------|--------------------:|-----------------------:|---------------:|
| base | 115.162 | 85.665 | -29.497 |
| custom metal_kernel | 143.549 | 111.496 | -32.053 |

- The official runner and the custom runner both recover about `30 ms/tok` once trace-forced synchronizations are removed.
- That means `mx.async_eval`, chunking, and runner-level scheduling are useful, but they are not the main differential root cause anymore.
- The remaining decode gap is still mostly inside the model-core path, not in the outer generation loop.

## Experiment 3: Remaining Decode Stage Delta, Base Trace Versus Custom Metal Trace

The stage totals are not exclusive because `full_attention` includes `rope` and `kv_cache_update`, and `linear_attention` includes nested cache work. Even with that limitation, the remaining pattern is clear.

| Stage | Base sync ms | Custom metal sync ms | Delta ms |
|------|-------------:|---------------------:|---------:|
| mlp | 1951.633 | 2298.435 | 346.802 |
| lm_head | 351.627 | 422.090 | 70.463 |
| full_attention | 226.909 | 261.191 | 34.282 |
| kv_cache_update | 103.386 | 108.837 | 5.451 |
| sampler_sync(argmax/item) | 10.628 | 8.676 | -1.952 |
| rope | 68.560 | 50.181 | -18.379 |
| linear_attention | 859.887 | 580.943 | -278.944 |

- After the Metal kernel port, the outer `linear_attention` sync is no longer slower than base.
- `KVCache::update_and_fetch()` is only slightly above base, so classic full-attention KV concatenation is now secondary, not primary.
- The biggest remaining positive deltas are `mlp` and `lm_head`, with a smaller but persistent `full_attention` delta.

## Experiment 4: Linear-Attention Microbenchmark, Compiled Ops Versus Metal Kernel

Legacy cache mode, same model-sized synthetic shapes.

| Stage | Compiled ops dispatch ms | Metal kernel dispatch ms | Compiled ops sync ms | Metal kernel sync ms |
|------|-------------------------:|-------------------------:|---------------------:|---------------------:|
| gated_delta_update | 5.779 | 1.830 | 244.187 | 50.760 |
| in_proj_qkv | 0.061 | 0.038 | 140.686 | 164.365 |
| in_proj_z | 0.124 | 0.090 | 80.090 | 90.579 |
| out_proj | 0.065 | 0.041 | 83.260 | 110.823 |
| conv1d | 0.275 | 0.208 | 30.176 | 31.690 |

- The Metal kernel crushes the recurrent core itself: `gated_delta_update` sync drops by about `-79.2%`, dispatch by about `-68.3%`.
- Once that recurrent core becomes cheap, the largest remaining linear-attention sync terms are no longer gated-delta. They become the quantized projection path around it: `in_proj_qkv`, `in_proj_z`, and `out_proj`.
- This means the decode bottleneck moved upstream/downstream of the recurrent kernel into quantized matmul-heavy projection code.

## Interpretation

The purpose of this follow-up is not to jump straight to optimization candidates. It is to identify where decode differs from the base official MLX path even after porting the upstream recurrent kernel behavior.

The current evidence says the decode gap is still reproduced inside model-core execution, not mostly in the outer runner:

- trace decode gap is still about `+28.387 ms/tok`, while no-trace decode gap is `+25.831 ms/tok`.
- `mlp` and `lm_head` remain consistent positive decode deltas versus base.
- `full_attention` stays slightly positive, while outer `linear_attention` sync is now below base.
- nested `linear_cache_update` is still strongly positive, so that path still differs from base even though outer linear-attention total no longer leads the gap.

## Current Decode Difference Findings

- The custom runner is not the main explanation for the decode gap. Base and custom both recover about `30 ms/tok` when trace-forced sync is removed.
- The stages that repeatedly stay above base in decode are `linear_cache_update` (nested timing), `mlp`, `lm_head`, and to a lesser extent `full_attention`.
- The stages that do not currently explain the remaining decode gap are outer `linear_attention`, `rope`, and sampler sync.

## Experiment 5: Projection Microbench And Split Decode Trace

- isolated decode projection microbench with real weights did **not** reproduce a slower custom kernel path: `mlp` sync was `1.975 -> 1.924 ms`, and `lm_head` sync was `11.504 -> 10.965 ms`.
- that means the earlier full-model `mlp`/`lm_head` delta is unlikely to be explained by the raw projection kernel alone.
- split decode trace shows `linear_cache_update` remains positive, and most of that comes from `linear_cache_conv_state_update` (`+73.450 ms`) rather than `linear_cache_rec_state_update` (`+28.001 ms`).
- split full-attention trace shows the remaining positive sub-stage deltas are concentrated in `full_attention_q_proj` (`+26.863 ms`), `full_attention_cache_update` (`+8.036 ms`), and `full_attention_o_proj` (`+7.956 ms`), while `full_attention_sdpa` itself is effectively flat.
- the next experiments should therefore explain graph/context differences around those stages, rather than assuming the isolated `mlp` or `lm_head` kernels are slower by themselves.

## Experiment 6: Graph-Context Root Cause Follow-Up

- once linear attention was split like full attention, the earlier `linear_cache_update` signal changed meaning: `linear_cache_conv_state_update` was no longer above base (`-9.637 ms`), and `linear_cache_update` itself became negative (`-22.483 ms`).
- the actual positive linear-path deltas moved to `linear_attention_in_proj_qkv` (`+70.213 ms`), `linear_attention_in_proj_z` (`+45.991 ms`), and `linear_attention_gated_delta` (`+14.346 ms`). `linear_attention_out_proj` was instead below base (`-67.462 ms`).
- this means the earlier positive cache-update signal was largely graph-attribution noise: once the surrounding projections were timed separately, conv-state update itself was not the remaining custom-vs-base decode penalty.
- a separate A/B on full-attention KV cache semantics also narrowed the attention-side conclusion. Switching custom KV cache from per-step concat to a base-like `step_buffer` reduced trace `full_attention_cache_update` by `-5.925 ms`, but it did **not** improve end-to-end no-trace decode (`108.990 -> 110.330 ms/tok`).
- therefore the current evidence points away from full-attention KV concat and away from conv-state update as the primary remaining cause. The tighter explanation is that the residual decode difference is concentrated in the linear-attention projection path inside full-model graph context, especially `in_proj_qkv` and `in_proj_z`.

## Experiment 7: Live Decode Projection Replay

- live decode activation을 official model에서 직접 캡처한 뒤 `in_proj_qkv`와 `in_proj_z`를 1:1 replay해 보면, official `QuantizedLinear`와 manual `mx.quantized_matmul`는 사실상 같은 비용이었다.
- `in_proj_qkv` 평균 sync는 base module `0.656 ms`, manual `quantized_matmul` `0.652 ms`였고, 평균 max abs diff는 `0.0`이었다.
- `in_proj_z` 평균 sync는 base module `0.452 ms`, manual `quantized_matmul` `0.443 ms`, `manual quantized_matmul + reshape` `0.442 ms`였고, 평균 max abs diff도 모두 `0.0`이었다.
- `in_proj_z`의 reshape only cost는 `0.044 ms`라서 trace에서 보인 `in_proj_z` delta를 설명하기에는 너무 작다.
- `dequantize + dense matmul`은 `in_proj_qkv` `2.206 ms`, `in_proj_z` `1.277 ms`로 quantized path보다 훨씬 비쌌다. 즉 남은 decode 차이는 "manual quantized path가 원래 느리다"거나 "reshape가 비싸다"로는 설명되지 않는다.
- 현재 가장 타당한 해석은, `linear_attention_in_proj_qkv`와 `linear_attention_in_proj_z`의 양의 delta가 raw projection kernel 자체가 아니라 full-model graph context 안의 scheduling / fusion / dependency composition 차이에서 생긴다는 것이다.

## Experiment 8: Live Decode Subgraph Replay

- projection 하나씩이 아니라 `projection bundle` (`in_proj_qkv / in_proj_z / in_proj_a / in_proj_b`) 전체를 live decode sample로 replay해도 결과는 같았다. official bundle sync는 `0.854 ms`, manual quantized bundle sync는 `0.850 ms`였다.
- 더 나아가 full `linear-attention block` 전체를 official path와 "projection만 manual quantized path로 교체한" path로 replay해도 평균 sync가 둘 다 `1.161 ms`였다.
- projection bundle의 `qkv/z/a/b` diff와 full block output diff는 모두 `0.0`이었다.
- 즉 base 내부에서는 quantized projection primitive를 manual path로 교체해도 block-level scheduling cost가 달라지지 않는다. 이 결과는 남은 base-vs-custom decode delta가 projection primitive 선택 자체가 아니라, custom full-model graph에서 projection 주변 연산이 compose/schedule 되는 방식에 있다는 해석을 더 강하게 만든다.