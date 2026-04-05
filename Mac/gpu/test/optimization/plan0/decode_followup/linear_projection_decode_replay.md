# Linear Projection Decode Replay

- Generated at: 2026-04-05T09:33:54
- Model dir: /Volumes/990pro/.lmstudio/models/mlx-community/Qwen3.5-9B-MLX-8bit
- Prompt suite: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan_0/operator_hook_test/prompt_suite.json
- Selected decode steps: [0, 15, 30]
- Samples: 288

## in_proj_qkv

- base module sync: 0.656 ms
- manual quantized_matmul sync: 0.652 ms
- dequantize only sync: 1.482 ms
- dense matmul on predequantized weight sync: 0.998 ms
- dequantize + dense matmul sync: 2.206 ms
- avg base vs manual max abs diff: 0.000000

## in_proj_z

- base module sync: 0.452 ms
- manual quantized_matmul sync: 0.443 ms
- manual quantized_matmul + reshape sync: 0.442 ms
- reshape only sync: 0.044 ms
- dequantize only sync: 0.884 ms
- dense matmul on predequantized weight sync: 0.638 ms
- dequantize + dense matmul sync: 1.277 ms
- avg base vs manual max abs diff: 0.000000
- avg base vs manual reshape max abs diff: 0.000000

## Findings

- live decode context replay에서 official `QuantizedLinear`와 manual `mx.quantized_matmul`는 `in_proj_qkv` 기준 0.656 vs 0.652 ms, `in_proj_z` 기준 0.452 vs 0.443 ms였다.
- `in_proj_z`의 reshape only cost는 0.044 ms라서 projection 본체에 비해 매우 작다.
- `dequantize + dense matmul`은 `in_proj_qkv`에서 2.206 ms, `in_proj_z`에서 1.277 ms로 manual quantized path보다 훨씬 크거나 같다면, delta는 dequantize boundary나 reshape보다 full-model graph context 쪽일 가능성이 높다.
- 이 문서는 live cache context에서 나온 실제 decode activation을 기준으로 official base 모듈과 custom manual quantized path를 1:1 replay하기 위한 산출물이다.

