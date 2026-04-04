# Linear Path Follow-Up

- Generated at: 2026-04-05T00:24:39
- Base trace: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan_0/operator_hook_test/base_stage_trace.json
- Baseline custom trace: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan_0/operator_hook_test/custom_stage_trace.json
- Arrays-style custom trace: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/linear_path_followup/arrays_cache_stage_trace.json
- Linear microbench: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/linear_path_followup/linear_attention_microbench.json
- Quantized path bench: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/linear_path_followup/quantized_path_bench.json

## Experiment 1: Arrays-Style Linear Cache Update

| Backend | Avg Prefill ms | Avg Decode ms/tok | Avg Wall ms | Avg Tok/s |
|--------|---------------:|------------------:|------------:|----------:|
| base | 514.607 | 115.162 | 4200.318 | 7.670 |
| custom baseline | 1844.759 | 159.884 | 6961.741 | 5.018 |
| custom arrays-style | 2290.630 | 160.046 | 7413.102 | 4.978 |

- Prefill linear_cache_update sync: baseline 320.026 ms -> arrays-style 342.619 ms.
- Decode linear_cache_update sync: baseline 997.365 ms -> arrays-style 998.766 ms.
- Prefill linear_attention dispatch: baseline 325.507 ms -> arrays-style 347.663 ms.
- Decode linear_attention dispatch: baseline 1020.052 ms -> arrays-style 1021.050 ms.

## Experiment 2: Linear Attention Microbenchmark

Synthetic shapes match the model config. This isolates host-side dispatch and stage sync without full-model noise.

### Legacy vs Arrays-Style

| Stage | Legacy dispatch ms | Legacy sync ms | Arrays dispatch ms | Arrays sync ms |
|------|-------------------:|---------------:|-------------------:|---------------:|
| conv1d | 0.494 | 32.582 | 0.229 | 33.866 |
| conv_cache_update | 0.107 | 28.244 | 0.109 | 24.168 |
| gated_delta_update | 6.706 | 359.820 | 6.039 | 353.433 |
| in_proj_a | 0.065 | 30.015 | 0.063 | 29.285 |
| in_proj_b | 0.056 | 31.931 | 0.057 | 31.507 |
| in_proj_qkv | 0.379 | 138.483 | 0.066 | 136.348 |
| in_proj_z | 0.099 | 82.654 | 0.104 | 82.791 |
| k_norm | 0.211 | 28.104 | 0.202 | 34.153 |
| norm_gated | 0.219 | 31.119 | 0.216 | 31.691 |
| out_proj | 0.068 | 85.294 | 0.076 | 85.339 |
| q_norm | 0.194 | 27.828 | 0.214 | 29.049 |
| rec_state_update | 0.040 | 2.589 | 0.037 | 2.383 |

## Experiment 3: Quantized lm_head vs MLP

| Path | Dispatch ms | Sync ms |
|------|------------:|--------:|
| mlp_prefill | 0.021 | 8.238 |
| mlp_decode | 0.002 | 2.027 |
| lm_head_prefill | 0.003 | 50.339 |
| lm_head_decode | 0.002 | 12.579 |

## Interpretation

Arrays-style conv cache update does not materially reduce `linear_cache_update`, so the major cost is likely outside the simple conv-state tail handling.
The isolated linear-attention microbenchmark still shows small local improvements in `conv_cache_update`, `conv1d`, `in_proj_qkv`, and `gated_delta_update` under arrays-style cache handling, but that local win does not translate into a full-model trace win. That implies the dominant full-model cost is the broader recurrent path and graph composition around `linear_attention`, not just the conv-state tail update itself.
In isolated quantized-path benchmarking, `lm_head` is heavier than a single `MLP` call for prefill-sized inputs. However, the full model still spends more total time in `mlp` because that path is executed once per layer while `lm_head` is executed once per token step.
If `linear_attention` dispatch stays high even after arrays-style cache update, the anomaly is not just cache tail maintenance; it points to the broader operator composition inside `GatedDeltaNet` such as repeated quantized projections, conv1d scheduling, and recurrent update graph construction.
