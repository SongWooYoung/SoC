# Performance Regression Notes

## Purpose

이 문서는 crash/fault는 아니지만 실제 측정에서 성능을 악화시킨 최적화 시도를 기록한다. 핵심은 "아이디어가 틀렸다"가 아니라 "현재 구현 방식으로는 왜 역효과였는가"를 남기는 것이다.

## 1. GPU Sampler Top-K Default Path

상태: `rejected as default`, `experimental only`

관찰:

1. `SamplerTopK`가 decode 경로에서 가장 큰 병목이었다.
2. 8-token quick run에서 GPU sampler가 있을 때 `wall_ms ~2980`, CPU fallback일 때 `wall_ms ~1807` 수준이었다.
3. profiling 기준 `SamplerTopK gpu_ms`는 8-token run에서 약 `1395 ms`까지 치솟았다.

원인:

1. 현재 `sampler_topk_f32_rowwise`는 single-row scalar scan이다.
2. vocab 전체를 한 줄씩 훑는 커널 구조가 Apple GPU에서 매우 비효율적이다.
3. logits buffer가 host-visible이라면 CPU readback + partial sort가 더 싸다.

현재 정책:

1. 기본값은 CPU sampler
2. `SOC_GPU_ENABLE_EXPERIMENTAL_GPU_SAMPLER=1`일 때만 기존 GPU sampler 사용

관련 파일:

- `Mac/gpu/src/runtime/sampler.cpp`
- `Mac/gpu/src/op/sampler_topk_op.mm`
- `Mac/gpu/shaders/gpu_kernels.metal`

## 2. Layer-Scoped CommandStream Batching

상태: `rejected for now`

관찰:

1. `SOC_GPU_ENABLE_EXPERIMENTAL_COMMAND_STREAM=layer`는 full-range batching보다 안전해 보였지만, 8-token quick run에서도 종료 지연/정지 성향을 보였다.
2. baseline보다 빠르지 않았고, 실전 최적화로 채택할 수 없었다.

원인 가설:

1. giant batching fault를 피하더라도, 현재 구현의 stream lifecycle과 temporary arena 사용 방식이 layer scope와 잘 맞지 않는다.
2. batching granularity를 줄였다고 해서 자동으로 성능이 오르는 것은 아니다.

현재 정책:

1. 유지하더라도 `experimental`
2. 기본 경로는 per-op command buffer

관련 파일:

- `Mac/gpu/src/model/qwen_causal_lm.cpp`
- `Mac/gpu/include/metal/command_stream.h`

## 3. Loader-Time Hot Weight Pretranspose

상태: `tested and reverted`

관찰:

1. attention/MLP projection weight를 loader에서 transpose해 `transpose_rhs=false` 경로를 타게 했을 때 성능이 악화됐다.
2. 8-token quick run에서 `wall_ms`가 대략 `1569 -> 2038 ms`로 나빠졌다.

원인 가설:

1. 현재 matmul kernel은 `transpose_rhs=true`와 기존 row-major weight layout에 더 잘 맞춰져 있다.
2. weight만 바꿔도 충분하지 않았고, kernel load pattern과 tile strategy까지 같이 바꿔야 했다.

교훈:

1. layout prepack은 유효한 방법론일 수 있다.
2. 하지만 loader에서 weight만 transpose하는 것으로는 충분하지 않다.
3. kernel access pattern, function constants, tile policy가 함께 바뀌지 않으면 오히려 느려질 수 있다.

관련 파일:

- `Mac/gpu/src/model/qwen_model_loader.cpp`
- `Mac/gpu/src/op/matmul_op.mm`
- `Mac/gpu/shaders/gpu_kernels.metal`

## 4. Decode Multi-Output-Per-Thread MatMul

상태: `tested and reverted`

관찰:

1. decode matmul에서 `32 threads -> 64 columns`를 한 번에 처리하는 전용 경로를 넣었지만, 실전 inference에서는 악화됐다.
2. 8-token quick run에서 `wall_ms`가 대략 `1620 -> 1960 ms`로 증가했다.

원인 가설:

1. microbenchmark에서는 일부 가능성이 보였지만, 실제 graph에서는 command scheduling, cache behavior, fused bias/residual path까지 합쳐져 다른 결과가 나왔다.
2. 즉 decode GEMV microbench 승자가 곧 end-to-end 승자는 아니었다.

교훈:

1. `microbenchmark win`은 채택 조건이 아니라 후보 선정 도구다.
2. 실전 graph에서 다시 확인하기 전에는 기본 경로에 넣지 않는다.

관련 파일:

- `Mac/gpu/src/op/matmul_op.mm`
- `Mac/gpu/shaders/gpu_kernels.metal`

## 5. Stale `gpu.metallib` Confusion

상태: `mitigated`

관찰:

1. shader source를 바꾼 뒤 오래된 `build/shaders/gpu.metallib`가 남아 있으면, 실행 결과가 갑자기 비정상적으로 빨라지거나 출력이 깨질 수 있었다.
2. 이건 최적화 성공이 아니라 runtime이 새 source와 안 맞는 오래된 metallib를 잡은 경우였다.

대응:

1. 새 runtime 필수 커널이 없으면 source compile로 fallback 하도록 강화했다.
2. 의심될 때는 `build/shaders/gpu.metallib`를 제거하고 다시 실행한다.

관련 파일:

- `Mac/gpu/src/metal/metal_context.mm`

## 6. Decode MLP Gate/Up Fusion

상태: `tested and reverted`

관찰:

1. `gate_proj + up_proj + SiLU*mul`를 decode 전용 단일 커널로 합친 실험은 8-token quick run에서는 개선을 보였다.
2. 하지만 실전 기준으로 잡은 `benchmark-full-gpu-vs-pytorch` 64-token full decode에서는 `~5.13 tok/s -> ~4.97 tok/s`로 오히려 악화됐다.
3. 출력 자체는 동일했고 crash도 없었으므로, 문제는 correctness가 아니라 장시간 decode에서의 효율 저하였다.

원인 가설:

1. 이 fusion은 중간 tensor 하나와 dispatch 하나를 줄여도, 매 token마다 여전히 `gate`와 `up` 전체 weight를 모두 읽어야 한다.
2. 즉 현재 병목인 weight bandwidth를 줄이지 못한 채 threadgroup memory 사용량과 커널 복잡도만 늘어난 셈이다.
3. 짧은 quick run 승리가 긴 decode throughput 승리를 보장하지 않는다는 점이 다시 확인됐다.

현재 정책:

1. decode MLP fusion은 기본 경로에 두지 않는다.
2. 이후 유사 실험은 `8-token quick run`만으로 채택하지 않고, 반드시 `64-token full benchmark`까지 통과해야 한다.
3. 다음 우선순위는 작은 op fusion이 아니라 decode-dominant weight layout/prepack과 그에 맞는 matmul kernel 재설계다.

관련 파일:

- `Mac/gpu/src/module/qwen_mlp.mm`
- `Mac/gpu/shaders/gpu_kernels.metal`
- `Mac/gpu/build/reports/full_gpu_vs_pytorch.md`

## 7. Decode Weight Prepack For MLP + `lm_head`

상태: `tested and reverted`

관찰:

1. decode에서 `MLP(gate/up/down)`와 `lm_head`만 별도 packed layout으로 복제하고, 전용 `matmul_f32_decode_tiled_packed` 경로를 타게 하는 실험을 했다.
2. 출력은 정상이고 `integration-real-bundle`도 통과했다.
3. 하지만 8-token quick run 기준 기존 안정 baseline 대비 성능이 크게 악화됐다.
4. 첫 번째 packed layout은 `wall_ms ~1726 -> ~1986`, `gpu_ms ~635 -> ~799`로 나빠졌다.
5. 두 번째 `32x32 inner-tile` layout도 `wall_ms ~1726 -> ~1950`, `gpu_ms ~635 -> ~802`로 여전히 악화됐다.

원인 가설:

1. 현재 decode 병목은 단순히 weight를 "다시 배치"한다고 해결되지 않았다.
2. `MLP + lm_head`만 따로 packed 복제해도, 실제 Apple GPU에서는 추가 메모리 footprint와 cache behavior 변화가 더 큰 손실을 만들었다.
3. 즉 `prepack`은 유효한 방법론일 수 있지만, 현재 커널/로드 패턴/대상 weight 조합으로는 오히려 느렸다.

교훈:

1. packed decode weight는 "layout을 하나 더 만든다" 수준으로는 채택하지 않는다.
2. 다음에 다시 시도한다면, 대상 weight 범위, kernel load pattern, 그리고 메모리 footprint를 함께 설계해야 한다.
3. quick run에서 이미 큰 regression이 보이면 full benchmark까지 가지 않고 바로 폐기한다.

관련 파일:

- `Mac/gpu/src/model/qwen_model_loader.cpp`
- `Mac/gpu/src/op/matmul_op.mm`
- `Mac/gpu/shaders/gpu_kernels.metal`
- `Mac/gpu/build/reports/quick/packed_decode_check.json`
- `Mac/gpu/build/reports/quick/packed_decode_check_v2.json`

## 8. Decode `tg_width=16` Override For `OProj` And `DownProj`

상태: `tested and reverted`

관찰:

1. `bench_matmul`의 isolated shape에서는 `decode_1x1024x1024_transposed`, `decode_1x1024x2816_transposed`에서 `tg_width=16`이 `32`보다 좋게 나왔다.
2. 이를 근거로 `OProjDecode`와 `DownProjDecode`에만 `preferred_threadgroup_width=16` override를 넣었다.
3. 하지만 실전 8-token quick run에서는 오히려 악화됐다.
4. `OProjDecode`는 대략 `79.5 -> 85.7 ms`, `DownProjDecode`는 `119.6 -> 129.3 ms`, 전체 `gpu_ms`는 `602.9 -> 635.9 ms`로 증가했다.

원인 가설:

1. isolated GEMV microbenchmark 승자가 실제 graph 승자는 아니었다.
2. 실제 decode graph에서는 allocator, command scheduling, 주변 op와의 cache 상호작용까지 합쳐져 다른 최적점이 나온다.
3. 즉 shape만 같다고 해도, full graph에서는 현재 기본 policy가 더 낫다는 뜻이다.

교훈:

1. `bench_matmul`은 후보 선정 도구이지 채택 근거가 아니다.
2. decode hot path 실험은 반드시 `labeled_breakdown.json` 같은 실전 계측으로 다시 확인한다.
3. 다음 커널 실험은 단순 threadgroup width override보다, load/accumulate 자체를 바꾸는 `float4` 벡터화 쪽이 우선이다.

관련 파일:

- `Mac/gpu/src/module/qwen_attention.mm`
- `Mac/gpu/src/module/qwen_mlp.mm`
- `Mac/gpu/build/reports/quick/labeled_breakdown.json`
- `Mac/gpu/build/reports/quick/labeled_breakdown_width16.json`

## 9. `LMHeadDecode` 4-Column-Per-Thread Kernel

상태: `rejected as default`, `experimental only`

관찰:

1. `LMHeadDecode` 전용으로 `32 threads -> 128 columns`를 처리하는 4-column-per-thread kernel을 추가했다.
2. short full benchmark 32-token 1회에서는 전체 `tok/s`가 올라가는 것처럼 보였지만, quick decode 기준으로는 `LMHeadDecode`가 오히려 악화됐다.
3. 실전 8-token quick run에서 `LMHeadDecode`는 대략 `52.7 -> 103.4 ms`로 증가했다.

원인 가설:

1. `lm_head`는 vocab 축이 매우 커서, threadgroup 수를 줄여도 register pressure와 global load pattern 악화가 더 크게 작용했다.
2. micro/full 단일 측정에서 보이는 wall 개선만으로는 kernel 채택 근거가 부족했다.
3. 이 경로는 캐시/occupancy 편차가 커서 머신/런 조건에 따라 들쑥날쑥할 수 있다.

현재 정책:

1. 기본 경로에서는 사용하지 않는다.
2. `SOC_GPU_ENABLE_EXPERIMENTAL_LMHEAD_4COL=1`일 때만 실험적으로 켠다.

관련 파일:

- `Mac/gpu/src/op/matmul_op.mm`
- `Mac/gpu/shaders/gpu_kernels.metal`

## 10. Prebuilt Decode Layer-Stream Replay

상태: `rejected as default`, `unsafe experimental`

관찰:

1. `real buffer-range hazard tracker`와 stage-local scratch arena를 붙인 뒤, `SOC_GPU_ENABLE_EXPERIMENTAL_PREBUILT_DECODE_LAYER_STREAM=1`으로 decode stage를 외부 `CommandStream`에 replay하는 실험을 했다.
2. `REAL_BUNDLE_MAX_NEW_TOKENS=1` smoke는 통과했다.
3. 하지만 `gpu_infer --max-new-tokens 8`는 30초 timeout으로 멈췄고, 실기기에서는 다시 `WindowServer` 문제가 관찰됐다.
4. 즉 single-step smoke success가 multi-token decode safety를 보장하지 않았다.

원인 가설:

1. `buffer id + byte range` tracking 자체는 진단용 infrastructure로 유효하다.
2. 실제 회귀를 만든 건 `RunBlockRange`에 외부 `CommandStream`을 주입해 layer-scope replay를 다시 도입한 점이다.
3. 이것은 과거 `SOC_GPU_ENABLE_EXPERIMENTAL_COMMAND_STREAM=layer` 회귀와 본질적으로 같은 방향이다.
4. 다시 말해 방법론의 문제라기보다, "prebuilt plan을 layer-scope command buffer replay로 쓰는 방식"이 M4 실기기에서 unsafe했다.

교훈:

1. 앞으로 prebuilt graph/plan 최적화는 layer replay가 아니라, 이미 안전성이 확인된 sublayer batch boundary만 대상으로 삼아야 한다.
2. smoke test는 필요조건일 뿐이고, 최소 8-token 이상 decode timeout/hang check가 같이 있어야 한다.
3. `real hazard tracker`와 `plan cache key` 확장은 유지할 가치가 있지만, layer-stream 실행 경로는 기본 비활성화 상태를 유지한다.

관련 파일:

- `Mac/gpu/src/runtime/command_scheduler.cpp`
- `Mac/gpu/src/model/qwen_causal_lm.cpp`
- `Mac/gpu/include/runtime/command_scheduler.h`
- `Mac/gpu/test/errors/01_windowserver_gpu_fault.md`
- `Mac/gpu/build/reports/quick/labeled_breakdown_lmhead_gateup.json`

## 10. Decode `Gate/Up` Fused Projection Kernel

상태: `rejected as default`, `experimental only`

관찰:

1. decode에서 `gate_proj`와 `up_proj`를 한 kernel에서 같이 계산해 입력 로드를 공유하는 경로를 추가했다.
2. dispatch 수는 줄었지만, quick decode 기준 전체 성능은 크게 악화됐다.
3. `GateProjDecode + UpProjDecode` 합을 새 `GateUpDecode`로 바꿨을 때, 기존 합 `~64.8 ms` 대비 새 fused 값은 `~77.1 ms`였다.
4. 전체 `gpu_ms`도 `424.9 -> 785.6 ms`로 악화됐다.

원인 가설:

1. gate/up 두 weight bandwidth는 그대로인데, fused kernel의 register pressure와 load pattern이 나빠졌다.
2. 특히 `hidden=1024`, `intermediate=2816` decode shape에서 occupancy 손실이 크게 났을 가능성이 높다.
3. 입력을 한 번 덜 읽는 이득보다 kernel 복잡도 증가가 더 컸다.

현재 정책:

1. 기본 경로에서는 사용하지 않는다.
2. `SOC_GPU_ENABLE_EXPERIMENTAL_FUSED_GATE_UP=1`일 때만 실험적으로 켠다.

관련 파일:

- `Mac/gpu/src/module/qwen_mlp.mm`
- `Mac/gpu/shaders/gpu_kernels.metal`
- `Mac/gpu/build/reports/quick/labeled_breakdown_gateup_only.json`
