# Test Results

## Saved Location

현재 GPU real-bundle regression 결과는 아래 두 위치에 저장했다.

1. Runtime-generated report
   `Mac/gpu/build/reports/test_real_bundle_regression_report.md`

2. Repo-stable summary
   `Mac/gpu/plan/09_TEST_RESULTS.md`

3. Prompt sweep summary
   `Mac/gpu/build/reports/test_real_bundle_sweep_summary.md`

4. Prompt sweep per-case reports
   `Mac/gpu/build/reports/sweep_cases/*.md`

5. Quick regression report
   `Mac/gpu/build/reports/quick/test_real_bundle_regression_report.md`

6. Quick sweep summary
   `Mac/gpu/build/reports/quick/test_real_bundle_sweep_summary.md`

7. Quick sweep per-case reports
   `Mac/gpu/build/reports/quick/sweep_cases/*.md`

정리하면:

1. `build/reports/...`는 test 실행 결과 artifact다.
2. `build/reports/quick/...`는 smoke/quick validation artifact다.
3. `plan/09_TEST_RESULTS.md`는 지금 확인된 결과와 재현 경로를 고정해서 보는 문서다.

## What Was Verified

`Mac/gpu/test/integration/test_real_bundle_regression.mm` 기준으로 아래를 확인했다.

1. Raw prompt case CPU/GPU first-token parity
2. Chat template case CPU/GPU first-token parity
3. Raw prompt case multi-token generated sequence parity
4. Chat template case multi-token generated sequence parity
5. `GenerationContext.Generate()`와 manual GPU decode sequence parity
6. top-k candidate 기준 logit drift가 허용 오차 이내

추가로 CPU reference 변경 이후 `Mac/cpu/build/bin/test_nn_modules`도 다시 실행해서 통과를 확인했다.

## Current Measured Baseline

환경:

1. Device: Apple M1 Pro
2. Manifest: `../../models/cpp/qwen3-0.6b/manifest.json`
3. Max new tokens: `8`

### Raw Prompt

1. Prompt tokens: `14`
2. First token parity: CPU=`7281`, GPU=`7281`
3. Max abs compared logit diff: `0.000027`
4. CPU wall: `29408.264 ms`
5. GPU context wall: `1810.261 ms`
6. CPU:GPU wall ratio: `16.245x`

### Chat Template Prompt

1. Prompt tokens: `35`
2. First token parity: CPU=`32313`, GPU=`32313`
3. Max abs compared logit diff: `0.000044`
4. CPU wall: `57590.725 ms`
5. GPU context wall: `1965.040 ms`
6. CPU:GPU wall ratio: `29.308x`

## Why The Result Changed

중간에 드러난 실제 원인은 CPU reference와 GPU implementation의 attention contract 차이였다.

핵심 차이:

1. GPU는 `q_norm` / `k_norm`를 적용하고 있었음
2. CPU는 그 경로를 빠뜨리고 있었음

그래서 CPU 쪽에 `q_norm` / `k_norm` 로딩과 적용을 추가한 뒤, real-bundle regression이 수렴했다.

## Follow-up Work Closed

이전에 남아 있던 5개 항목은 현재 아래 형태로 모두 반영됐다.

1. Top-level regression integration
   workspace root에 `Makefile`을 추가해서 `cd /Users/song-ganghui/Documents/SoC && make regression` 한 번으로 CPU golden regression과 GPU real-bundle regression을 연속 실행할 수 있다.

2. Prompt-length sweep
   `cd /Users/song-ganghui/Documents/SoC && make gpu-sweep` 또는 `cd /Users/song-ganghui/Documents/SoC/Mac/gpu && make integration-sweep`으로 short/medium/long prompt sweep을 자동 실행하고 markdown summary를 남긴다.

3. Per-stage analysis refinement
   `Mac/gpu/test/integration/test_real_bundle_regression.mm` report는 이제 total/prefill/decode wall time, prefill/decode GPU ms, prefill per prompt-token, decode per generated-token, tokens/sec, stage share를 함께 기록한다.

4. Stronger GPU utilization metrics
   GPU utilization은 더 이상 temporary arena proxy만 사용하지 않는다. Metal command-buffer GPU timestamps에서 집계한 `GPU ms`와 `GPU active ratio`를 report에 포함하고, temporary arena working-set ratio는 별도 memory-pressure proxy로 유지한다.

5. Scripted reproducibility
   `Mac/gpu/tools/run_real_bundle_sweep.sh`와 `Mac/gpu/Makefile` target으로 local 재현 경로를 고정했다.

## Reproduction Commands

1. Workspace aggregate regression
   `cd /Users/song-ganghui/Documents/SoC && make regression`

2. GPU real-bundle regression only
   `cd /Users/song-ganghui/Documents/SoC/Mac/gpu && make integration-real-bundle`

3. GPU sweep summary
   `cd /Users/song-ganghui/Documents/SoC && make gpu-sweep`

4. GPU quick regression
   `cd /Users/song-ganghui/Documents/SoC && make regression-gpu-quick`

5. GPU quick sweep
   `cd /Users/song-ganghui/Documents/SoC && make gpu-sweep-quick`

## Report Coverage

현재 markdown report는 각 case마다 아래 항목을 포함한다.

1. CPU / GPU first-token parity
2. CPU / GPU multi-token sequence parity
3. Total wall ms / CPU ms / GPU ms
4. CPU active ratio / GPU active ratio
5. Prefill / decode wall time
6. Prefill / decode GPU ms
7. Prefill ms per prompt token
8. Decode ms per generated token
9. Tokens per second
10. CPU decode share, GPU prefill share, GPU decode share
11. Temporary arena peak bytes / working-set ratio

## Recommended Reading Order

1. 실행 결과 원문: `Mac/gpu/build/reports/test_real_bundle_regression_report.md`
2. baseline 해석: `Mac/gpu/plan/08_REAL_BUNDLE_BASELINE.md`
3. sweep summary: `Mac/gpu/build/reports/test_real_bundle_sweep_summary.md`
4. 현재 저장 위치와 자동화 경로: `Mac/gpu/plan/09_TEST_RESULTS.md`

## Artifact Separation

quick target과 full target은 이제 서로 다른 artifact를 쓴다.

1. `make regression-gpu`는 full regression artifact를 갱신한다.
2. `make regression-gpu-quick`는 `build/reports/quick/...`만 갱신한다.
3. `make gpu-sweep`는 full sweep artifact를 갱신한다.
4. `make gpu-sweep-quick`는 `build/reports/quick/...`만 갱신한다.