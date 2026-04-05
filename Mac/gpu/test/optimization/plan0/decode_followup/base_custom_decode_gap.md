# Base vs Custom Decode Gap

- Generated at: 2026-04-05T01:18:50
- Base trace: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan_0/operator_hook_test/base_stage_trace.json
- Custom trace: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/metal_kernel_stage_trace.json
- No-trace comparison: /Volumes/990pro/Documents/SoC/Mac/gpu/test/result/qwen3_5_9b_mlx_lib_vs_custom.json
- Scope: decode에서 base official MLX와 custom이 어디서 다르게 느려지는지만 정리한다.

## Overall Decode Gap

- No-trace 최신 평균: base 85.665 ms/tok, custom 111.496 ms/tok, delta +25.831 ms/tok.
- Trace 평균: base 115.162 ms/tok, custom 143.548 ms/tok, delta +28.386 ms/tok.
- Runner benefit: base 29.497 ms/tok, custom 32.052 ms/tok.

## Prompt-Level Decode Delta

| Prompt | Base decode ms/tok | Custom decode ms/tok | Delta ms/tok |
|-------|--------------------:|---------------------:|-------------:|
| short_03 | 115.340 | 142.701 | 27.361 |
| short_10 | 114.969 | 142.938 | 27.969 |
| long_03 | 115.113 | 141.151 | 26.038 |
| long_09 | 115.225 | 147.404 | 32.179 |

## Decode Sync Delta by Stage

상위 양의 delta stage: linear_cache_update, mlp, lm_head.

| Stage | Avg delta ms | Min delta ms | Max delta ms | Positive prompts |
|------|-------------:|-------------:|-------------:|-----------------:|
| linear_cache_update | 815.073 | 798.295 | 849.157 | 4/4 |
| mlp | 346.802 | 308.915 | 406.431 | 4/4 |
| lm_head | 70.463 | 55.891 | 79.569 | 4/4 |
| full_attention | 34.282 | 28.316 | 44.242 | 4/4 |
| kv_cache_update | 5.451 | 1.088 | 10.443 | 4/4 |
| input_embeddings | -0.459 | -0.914 | 0.027 | 1/4 |
| final_norm | -0.775 | -1.171 | -0.148 | 0/4 |
| sampler_sync(argmax/item) | -1.952 | -5.287 | -0.559 | 0/4 |
| position_ids | -7.159 | -9.233 | -6.077 | 0/4 |
| rope | -18.379 | -21.920 | -13.952 | 0/4 |
| linear_attention | -278.944 | -288.641 | -271.099 | 0/4 |

## Decode Dispatch Delta by Stage

host-side dispatch 차이가 큰 stage: linear_attention, sampler_sync(argmax/item), final_norm.

| Stage | Avg delta ms | Min delta ms | Max delta ms | Positive prompts |
|------|-------------:|-------------:|-------------:|-----------------:|
| linear_attention | 794.061 | 777.022 | 827.513 | 4/4 |
| sampler_sync(argmax/item) | -0.004 | -0.010 | 0.002 | 1/4 |
| final_norm | -0.094 | -0.133 | -0.077 | 0/4 |
| linear_cache_update | -0.097 | -0.129 | -0.044 | 0/4 |
| lm_head | -0.109 | -0.120 | -0.097 | 0/4 |
| input_embeddings | -0.145 | -0.152 | -0.140 | 0/4 |
| position_ids | -0.520 | -0.535 | -0.486 | 0/4 |
| kv_cache_update | -1.860 | -1.947 | -1.702 | 0/4 |
| mlp | -4.524 | -4.992 | -4.120 | 0/4 |
| rope | -5.917 | -6.031 | -5.752 | 0/4 |
| full_attention | -31.057 | -37.889 | -20.766 | 0/4 |

## Findings

- `linear_cache_update`는 decode sync에서 평균 +815.073 ms로 4/4 prompt 모두 base보다 높다. 다만 이 stage는 nested 계측이라 outer `linear_attention`와 단순 합산하면 안 된다.
- `mlp`는 decode sync에서 평균 +346.802 ms로 4/4 prompt 모두 base보다 느리다.
- `lm_head`는 decode sync에서 평균 +70.463 ms로 4/4 prompt 모두 느리다.
- `full_attention`는 평균 +34.282 ms로 남아 있지만, outer `linear_attention`는 평균 -278.944 ms라서 decode gap의 주범이 linear_attention 전체라고 보기는 어렵다.
- dispatch 기준으로는 `linear_attention` 평균 +794.061 ms가 가장 커서 host-side graph composition 차이도 따로 남아 있다.
- 결론: 현재 decode gap은 runner가 아니라 model-core 차이로 재현되며, 가장 일관되게 벌어지는 구간은 `mlp`, `lm_head`, 그리고 일부 `full_attention`/`linear_cache_update`다.

