# Linear Subgraph Decode Replay

- Generated at: 2026-04-05T09:47:26
- Model dir: /Volumes/990pro/.lmstudio/models/mlx-community/Qwen3.5-9B-MLX-8bit
- Prompt suite: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan_0/operator_hook_test/prompt_suite.json
- Selected decode steps: [0, 15, 30]
- Samples: 288

## Projection Bundle

- official sync: 0.854 ms
- manual quantized sync: 0.850 ms
- avg qkv max abs diff: 0.000000
- avg z max abs diff: 0.000000
- avg a max abs diff: 0.000000
- avg b max abs diff: 0.000000

## Full Linear-Attention Block

- official sync: 1.161 ms
- manual projection sync: 1.161 ms
- avg output max abs diff: 0.000000

## Findings

- projection bundle 단위에서도 official과 manual quantized path는 0.854 vs 0.850 ms로 거의 같다.
- full linear-attention block에서도 projection만 manual path로 바꾼 replay는 1.161 vs 1.161 ms로 거의 같다.
- full block output diff도 0.000000라서, base 내부에서는 projection primitive를 바꿔도 block-level scheduling cost가 거의 달라지지 않는다.
- 따라서 현재 남은 base-vs-custom decode delta는 projection primitive 자체보다, custom full-model graph 안에서 projection 주변 연산이 어떻게 compose/schedule 되는지에서 확인해야 한다.

