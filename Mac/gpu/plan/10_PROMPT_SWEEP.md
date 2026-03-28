# Prompt Sweep

## Scope

이 문서는 GPU real-bundle prompt-length sweep의 목적, 결과 저장 위치, 해석 기준을 repo에 고정해서 남긴다.

## Saved Location

1. Full sweep summary
   `Mac/gpu/build/reports/test_real_bundle_sweep_summary.md`
2. Full sweep per-case reports
   `Mac/gpu/build/reports/sweep_cases/*.md`
3. Quick sweep summary
   `Mac/gpu/build/reports/quick/test_real_bundle_sweep_summary.md`
4. Quick sweep per-case reports
   `Mac/gpu/build/reports/quick/sweep_cases/*.md`

## Coverage

현재 sweep는 아래 세 prompt 길이를 자동 실행한다.

1. `short`
2. `medium`
3. `long`

각 case는 아래 두 입력 형태를 함께 비교한다.

1. Raw prompt
2. Chat template prompt

## What The Sweep Captures

summary는 각 case마다 아래 항목을 기록한다.

1. Raw prompt token count
2. Chat prompt token count
3. Raw CPU:GPU wall ratio
4. Chat CPU:GPU wall ratio
5. Raw GPU active ratio
6. Chat GPU active ratio
7. Per-case markdown report path

## Current Measured Results

### Full Sweep Baseline

실측 환경:

1. Device: Apple M1 Pro
2. Manifest: `../../models/cpp/qwen3-0.6b/manifest.json`
3. Max new tokens: `8`

현재 full summary는 아래 수치를 기록한다.

1. `short`: raw tokens=`12`, chat tokens=`33`, raw CPU:GPU wall=`14.731x`, chat CPU:GPU wall=`27.889x`, raw GPU active ratio=`0.491931`, chat GPU active ratio=`0.531650`
2. `medium`: raw tokens=`31`, chat tokens=`52`, raw CPU:GPU wall=`23.984x`, chat CPU:GPU wall=`37.607x`, raw GPU active ratio=`0.535106`, chat GPU active ratio=`0.571784`
3. `long`: raw tokens=`65`, chat tokens=`86`, raw CPU:GPU wall=`31.725x`, chat CPU:GPU wall=`40.823x`, raw GPU active ratio=`0.563559`, chat GPU active ratio=`0.580876`

### Quick Sweep Baseline

실측 환경:

1. Device: Apple M1 Pro
2. Manifest: `../../models/cpp/qwen3-0.6b/manifest.json`
3. Max new tokens: `1`

현재 quick summary는 아래 수치를 기록한다.

1. `short`: raw tokens=`12`, chat tokens=`33`, raw CPU:GPU wall=`62.147x`, chat CPU:GPU wall=`115.164x`, raw GPU active ratio=`0.528837`, chat GPU active ratio=`0.691915`
2. `medium`: raw tokens=`31`, chat tokens=`52`, raw CPU:GPU wall=`111.102x`, chat CPU:GPU wall=`142.118x`, raw GPU active ratio=`0.681602`, chat GPU active ratio=`0.752300`
3. `long`: raw tokens=`65`, chat tokens=`86`, raw CPU:GPU wall=`148.859x`, chat CPU:GPU wall=`162.505x`, raw GPU active ratio=`0.794300`, chat GPU active ratio=`0.829653`

## Expected Trend

prompt 길이가 길어질수록 일반적으로 아래 경향을 기대한다.

1. prompt token count 증가
2. GPU active ratio 증가
3. CPU:GPU wall ratio 증가 또는 유지
4. temporary arena peak 증가

이 경향은 prefill 비중이 커질수록 GPU 쪽 연산 밀도가 올라가는 현재 runtime 구조와 맞는다.

## Reproduction

full sweep:

```sh
cd /Users/song-ganghui/Documents/SoC
make gpu-sweep
```

quick sweep:

```sh
cd /Users/song-ganghui/Documents/SoC
make gpu-sweep-quick
```

## Interpretation Notes

1. `GPU active ratio`는 Metal command-buffer GPU timestamp 기반 busy share이며 occupancy counter는 아니다.
2. quick sweep은 smoke validation용이라 `max_new_tokens=1`을 사용한다.
3. baseline 비교나 장기 보관은 quick artifact가 아니라 full sweep artifact를 기준으로 보는 편이 안전하다.
4. quick sweep은 decode token 수가 작아서 CPU:GPU wall ratio가 full sweep보다 더 크게 보일 수 있다.
5. full sweep에서도 `short -> medium -> long`으로 갈수록 wall ratio와 GPU active ratio가 함께 증가해, prefill 비중 증가에 따른 GPU 쪽 이득이 유지되는 것을 확인할 수 있다.