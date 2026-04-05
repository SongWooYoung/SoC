# Projection And Attention Follow-Up

- Generated at: 2026-04-05T01:32:55
- Base split trace: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/base_stage_trace_split.json
- Custom split trace: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/custom_stage_trace_split.json
- Base projection microbench: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/base_decode_projection_microbench.json
- Custom projection microbench: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/custom_decode_projection_microbench.json

## Decode Projection Microbench

- MLP average sync: base 1.975 ms, custom 1.924 ms, delta -0.051 ms.
- LM head decode sync: base 11.504 ms, custom 10.965 ms, delta -0.539 ms.

## Linear Cache Split Decode

| Stage | Base dispatch ms | Custom dispatch ms | Delta dispatch ms | Base sync ms | Custom sync ms | Delta sync ms |
|------|------------------:|-------------------:|------------------:|-------------:|---------------:|--------------:|
| linear_cache_update | 0.495 | 0.561 | 0.066 | 765.037 | 866.489 | 101.452 |
| linear_cache_conv_state_update | 0.216 | 0.543 | 0.327 | 516.482 | 589.932 | 73.450 |
| linear_cache_rec_state_update | 0.279 | 0.018 | -0.261 | 248.555 | 276.556 | 28.001 |

## Full Attention Split Decode

| Stage | Base dispatch ms | Custom dispatch ms | Delta dispatch ms | Base sync ms | Custom sync ms | Delta sync ms |
|------|------------------:|-------------------:|------------------:|-------------:|---------------:|--------------:|
| full_attention | 760.304 | 773.833 | 13.529 | 4.722 | 4.451 | -0.271 |
| full_attention_q_proj | 0.563 | 0.140 | -0.423 | 169.966 | 196.829 | 26.863 |
| full_attention_k_proj | 0.651 | 0.223 | -0.428 | 74.578 | 80.103 | 5.525 |
| full_attention_v_proj | 0.673 | 0.243 | -0.430 | 71.128 | 76.870 | 5.742 |
| full_attention_q_norm | 0.894 | 0.155 | -0.739 | 51.276 | 50.788 | -0.488 |
| full_attention_k_norm | 0.888 | 0.126 | -0.762 | 49.773 | 47.404 | -2.369 |
| full_attention_rope | 7.066 | 0.587 | -6.479 | 65.727 | 49.825 | -15.902 |
| full_attention_cache_update | 1.890 | 0.143 | -1.747 | 60.118 | 68.154 | 8.036 |
| full_attention_sdpa | 1.126 | 0.327 | -0.799 | 67.952 | 67.719 | -0.233 |
| full_attention_o_proj | 1.064 | 0.312 | -0.752 | 120.417 | 128.373 | 7.956 |

## Findings

- `linear_cache_conv_state_update` delta는 73.450 ms이고, `linear_cache_rec_state_update` delta는 28.001 ms다.
- isolated decode projection microbench에서는 `mlp`와 `lm_head` 자체가 base보다 느리지 않았다. 따라서 full-model trace의 `mlp`/`lm_head` delta는 순수 projection kernel 자체보다 상위 graph/context 차이와 더 관련 있을 가능성이 높다.
- `full_attention` 내부에서는 `full_attention_q_proj`, `full_attention_cache_update`, `full_attention_o_proj`가 양의 sync delta로 남고, `full_attention_sdpa` 자체는 거의 차이가 없었다.
- 이 문서는 base와 custom이 decode에서 실제로 갈라지는 sub-stage만 다시 확인하기 위한 follow-up 산출물이다.

