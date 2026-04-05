# Graph Context Root Cause

- Generated at: 2026-04-05T09:22:15
- Base split trace: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/base_stage_trace_split.json
- Custom legacy split trace: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/custom_stage_trace_linear_split.json
- Custom step-buffer split trace: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/custom_stage_trace_linear_split_step_buffer.json
- Custom no-trace legacy: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/custom_no_trace_legacy_4prompt.json
- Custom no-trace step-buffer: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/decode_followup/custom_no_trace_step_buffer_4prompt.json

## Base vs Custom Legacy: Linear Path Decode Delta

- trace decode average: base 184.890 ms/tok, custom 195.010 ms/tok, delta +10.120 ms/tok.

| Stage | base_sync_ms | custom_sync_ms | Delta ms |
|------|---------:|---------:|---------:|
| linear_attention_in_proj_qkv | 497.425 | 567.638 | +70.213 |
| linear_attention_in_proj_z | 329.677 | 375.668 | +45.991 |
| linear_attention_gated_delta | 194.821 | 209.167 | +14.346 |
| linear_attention_conv1d | 144.174 | 152.126 | +7.952 |
| linear_attention_norm_gated | 146.251 | 151.019 | +4.768 |
| linear_attention | 12.566 | 12.438 | -0.128 |
| linear_attention_q_norm | 144.885 | 142.359 | -2.526 |
| linear_attention_in_proj_a | 161.130 | 157.317 | -3.813 |
| linear_attention_in_proj_b | 175.296 | 170.817 | -4.479 |
| linear_attention_k_norm | 147.067 | 140.530 | -6.537 |
| linear_cache_conv_state_update | 153.746 | 144.109 | -9.637 |
| linear_cache_rec_state_update | 25.051 | 12.204 | -12.847 |
| linear_cache_update | 178.797 | 156.314 | -22.483 |
| linear_attention_out_proj | 454.051 | 386.589 | -67.462 |

## Base vs Custom Legacy: Attention Path Decode Delta

| Stage | base_sync_ms | custom_sync_ms | Delta ms |
|------|---------:|---------:|---------:|
| full_attention_q_proj | 168.368 | 189.593 | +21.225 |
| full_attention_cache_update | 59.892 | 65.997 | +6.105 |
| full_attention | 5.546 | 4.872 | -0.674 |
| full_attention_sdpa | 67.561 | 65.889 | -1.672 |
| full_attention_o_proj | 144.288 | 129.757 | -14.531 |

## Step-Buffer Effect On Attention Path

- custom trace decode average: legacy 195.010 ms/tok, step-buffer 198.283 ms/tok, delta +3.273 ms/tok.
- custom no-trace decode average: legacy 108.990 ms/tok, step-buffer 110.330 ms/tok, delta +1.340 ms/tok.

| Stage | legacy_sync_ms | step_sync_ms | Delta ms |
|------|---------:|---------:|---------:|
| full_attention_cache_update | 65.997 | 60.072 | -5.925 |
| full_attention_o_proj | 129.757 | 133.829 | +4.072 |
| full_attention_q_proj | 189.593 | 192.877 | +3.284 |
| full_attention_sdpa | 65.889 | 68.707 | +2.818 |
| full_attention | 4.872 | 4.663 | -0.209 |

## Findings

- legacy full-model linear path에서 base 대비 가장 큰 양의 delta는 linear_attention_in_proj_qkv (+70.213 ms), linear_attention_in_proj_z (+45.991 ms), linear_attention_gated_delta (+14.346 ms)였다.
- `linear_cache_conv_state_update` delta는 -9.637 ms였지만, `linear_attention_in_proj_qkv` delta는 +70.213 ms, `linear_attention_out_proj` delta는 -67.462 ms였다. 즉 conv-state update 하나만의 문제가 아니라 그 앞뒤 projection path 전체가 같이 비싸다.
- attention cache를 step-buffer로 바꾸면 trace 기준 `full_attention_cache_update`는 -5.925 ms, `full_attention_q_proj`는 +3.284 ms, `full_attention_o_proj`는 +4.072 ms 바뀐다.
- 그런데 같은 변경의 no-trace decode 평균은 +1.340 ms/tok로 거의 개선되지 않았다. 따라서 attention cache concat은 trace-forced sync에서는 큰 영향을 주지만, 현재 end-to-end decode gap의 주원인이라고 보기는 어렵다.
- 현재 증거상 남은 핵심 차이는 full-attention KV cache 자체보다는 linear attention 안의 quantized projection/conv/out-proj 경로와 그 주변 graph-context 비용이 더 크다. conv-state update delta도 존재하지만, 그것만 따로 떼어 최상위 원인으로 보기는 어렵다.

