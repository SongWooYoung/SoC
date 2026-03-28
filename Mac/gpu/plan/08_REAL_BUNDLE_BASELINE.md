# Real Bundle Baseline

## Scope

이 문서는 `Mac/gpu/test/integration/test_real_bundle_regression.mm`를 실제 Qwen3-0.6B export bundle에 대해 실행해서 얻은 baseline을 기록한다.

측정 목적은 아래 세 가지다.

1. CPU reference와 GPU runtime의 multi-token sequence parity를 고정한다.
2. raw prompt와 chat template prompt 둘 다 tokenizer/runtime contract에 묶는다.
3. 현재 CPU:GPU wall time, host CPU time, temporary working-set proxy를 기록한다.

## Validation Status

검증 시점의 regression은 아래를 만족했다.

1. Raw prompt case first-token parity 통과
2. Chat template case first-token parity 통과
3. Raw prompt CPU/GPU generated token sequence parity 통과
4. Chat template CPU/GPU generated token sequence parity 통과
5. `GenerationContext.Generate()` sequence가 manual GPU loop와 일치
6. compared top-k candidate 기준 max abs logit diff가 `1e-3` 이내

이 상태에 도달하기 전 핵심 mismatch는 CPU attention path가 Qwen3의 `q_norm` / `k_norm`를 반영하지 않던 점이었다. CPU reference에 해당 norm을 추가한 뒤 real-bundle regression이 수렴했다.

## Environment

실측 환경:

1. Device: Apple M1 Pro
2. Unified memory: yes
3. Thread execution width: 32
4. Recommended max working set size: `12713115648` bytes
5. Manifest: `../../models/cpp/qwen3-0.6b/manifest.json`
6. Default prompt: `Summarize how CPU and GPU scheduling differ in one short paragraph.`
7. Max new tokens: `8`

## Measured Results

### Raw Prompt

1. Prompt tokens: `14`
2. Generated tokens: `8`
3. First token: CPU=`7281`, GPU=`7281`
4. Max abs logit diff on compared candidates: `0.000027`
5. Generated text: ` Also, explain why the CPU and GPU`

Timing:

1. CPU total wall: `29408.264 ms`
2. GPU context total wall: `1810.261 ms`
3. CPU total CPU time: `29186.529 ms`
4. GPU context host CPU time: `235.517 ms`
5. CPU active ratio: `0.992`
6. GPU context host CPU ratio: `0.130`
7. CPU:GPU wall ratio: `16.245x`
8. CPU:GPU host-CPU ratio: `123.925x`
9. GPU temp peak bytes: `1057816`
10. GPU working-set ratio: `0.000083`

### Chat Template Prompt

1. Prompt tokens: `35`
2. Generated tokens: `8`
3. First token: CPU=`32313`, GPU=`32313`
4. Max abs logit diff on compared candidates: `0.000044`
5. Generated text: `Okay, the user wants a summary of`

Timing:

1. CPU total wall: `57590.725 ms`
2. GPU context total wall: `1965.040 ms`
3. CPU total CPU time: `57291.670 ms`
4. GPU context host CPU time: `238.052 ms`
5. CPU active ratio: `0.995`
6. GPU context host CPU ratio: `0.121`
7. CPU:GPU wall ratio: `29.308x`
8. CPU:GPU host-CPU ratio: `240.669x`
9. GPU temp peak bytes: `2738200`
10. GPU working-set ratio: `0.000215`

## Interpretation

현재 baseline에서 읽을 수 있는 결론은 아래와 같다.

1. correctness는 first-token 수준이 아니라 full generated sequence 수준까지 CPU와 GPU가 맞는다.
2. chat template prompt는 raw prompt보다 prompt length가 늘어나지만 GPU wall time 증가는 제한적이다.
3. CPU reference는 거의 전 시간을 host CPU에서 소모한다.
4. GPU path는 wall time이 더 짧을 뿐 아니라 host CPU 점유도 크게 낮다.
5. temporary arena peak는 recommended working set 대비 매우 작아서, 현재 regression prompt 길이에서는 memory pressure가 병목이 아니다.

## Caveats

이 baseline의 GPU utilization 수치는 하드웨어 occupancy counter가 아니다.

현재 test가 기록하는 것은 아래 proxy다.

1. CPU utilization proxy: process CPU time / wall time
2. GPU memory pressure proxy: temporary arena peak / `recommended_max_working_set_size`

즉, 실제 SM occupancy나 GPU busy percentage를 직접 의미하지는 않는다.

## Reproduction

기본 실행:

```sh
cd Mac/gpu
make integration
```

report artifact:

1. `Mac/gpu/build/reports/test_real_bundle_regression_report.md`

유용한 override:

```sh
SOC_QWEN_PROMPT='A shorter prompt' \
SOC_QWEN_MAX_NEW_TOKENS=4 \
SOC_GPU_REPORT_PATH=build/reports/custom_report.md \
make integration
```
