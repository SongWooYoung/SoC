# Layer Split Benchmark

`gpu_infer --layer k`의 실측 benchmark를 `Hello world`, `max_new_tokens=64` 조건으로 돌린 결과다.

실행 artifact는 `Mac/gpu/reports/layer_split_benchmark.md`와 `Mac/gpu/reports/layer_split_benchmark/layer_*.json` 아래에 남는다. 이 문서는 현재 확인된 결과를 repo-stable로 고정한 것이다.

## Setup

1. Prompt: `Hello world`
2. max_new_tokens: `64`
3. Cases: `1..27` + `-1` (`-1`은 full GPU)
4. Device: Apple M1 Pro

## Results

| requested_layer | mode | resolved_gpu_layers | wall_ms | gpu_ms | tokens_per_sec |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | hybrid | 1 | 115582.00 | 656.53 | 0.554 |
| 2 | hybrid | 2 | 113000.00 | 1168.05 | 0.566 |
| 3 | hybrid | 3 | 111045.00 | 1473.84 | 0.576 |
| 4 | hybrid | 4 | 108117.00 | 1673.45 | 0.592 |
| 5 | hybrid | 5 | 105629.00 | 1911.85 | 0.606 |
| 6 | hybrid | 6 | 103603.00 | 2264.36 | 0.618 |
| 7 | hybrid | 7 | 101152.00 | 2439.12 | 0.633 |
| 8 | hybrid | 8 | 99055.00 | 2645.73 | 0.646 |
| 9 | hybrid | 9 | 96701.30 | 3038.72 | 0.662 |
| 10 | hybrid | 10 | 94088.00 | 3302.68 | 0.680 |
| 11 | hybrid | 11 | 92285.20 | 4387.02 | 0.694 |
| 12 | hybrid | 12 | 88856.70 | 3784.29 | 0.720 |
| 13 | hybrid | 13 | 87765.80 | 4824.04 | 0.729 |
| 14 | hybrid | 14 | 84674.50 | 4782.91 | 0.756 |
| 15 | hybrid | 15 | 83590.80 | 5731.56 | 0.766 |
| 16 | hybrid | 16 | 80709.20 | 6341.23 | 0.793 |
| 17 | hybrid | 17 | 78098.50 | 6953.29 | 0.819 |
| 18 | hybrid | 18 | 74818.00 | 6763.43 | 0.855 |
| 19 | hybrid | 19 | 74057.30 | 8228.73 | 0.864 |
| 20 | hybrid | 20 | 70817.60 | 8664.56 | 0.904 |
| 21 | hybrid | 21 | 67509.40 | 9060.14 | 0.948 |
| 22 | hybrid | 22 | 63265.70 | 8548.94 | 1.012 |
| 23 | hybrid | 23 | 62047.50 | 9913.43 | 1.031 |
| 24 | hybrid | 24 | 59178.20 | 10126.10 | 1.081 |
| 25 | hybrid | 25 | 59767.50 | 12664.00 | 1.071 |
| 26 | hybrid | 26 | 54666.20 | 11013.70 | 1.171 |
| 27 | hybrid | 27 | 52601.20 | 11708.00 | 1.217 |
| -1 | full-gpu | 28 | 35665.70 | 25851.30 | 1.794 |

## Readout

1. prefix GPU layer 수가 커질수록 wall time이 거의 단조 감소했다.
2. `layer 27` hybrid도 `1.217 tok/s`로 빨라지지만, full GPU `-1`의 `1.794 tok/s`가 여전히 가장 빠르다.
3. hybrid에서는 GPU time이 늘어도 CPU suffix와 hidden handoff 비용이 남아서 end-to-end wall time이 full GPU보다 크다.
4. `layer 24 -> 25` 구간은 약간의 흔들림이 보이지만 전체 추세는 GPU prefix 확대가 유리한 방향이다.

## Reproduce

```sh
cd /Users/song-ganghui/Documents/SoC/Mac/gpu
make benchmark-layer-split
```