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
- `Mac/gpu/reports/full_gpu_vs_pytorch.md`

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
- `Mac/gpu/reports/quick/packed_decode_check.json`
- `Mac/gpu/reports/quick/packed_decode_check_v2.json`

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
- `Mac/gpu/reports/quick/labeled_breakdown.json`
- `Mac/gpu/reports/quick/labeled_breakdown_width16.json`

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

## 10. Deferred `DecodePostNormMlpBatch` Wait

상태: `tested and kept experimental only`

관찰:

1. `SOC_GPU_ENABLE_EXPERIMENTAL_DEFERRED_MLP_WAIT=1`로 `DecodePostNormMlpBatch`를 block마다 기다리지 않고 token 끝까지 미루는 실험을 했다.
2. profiling상 `DecodePostNormMlpBatch wait_ms`는 사실상 `0`으로 떨어졌다.
3. 하지만 8-token quick run 전체는 오히려 악화됐다.
4. `q4 + safe + block attention` 기준 `wall_ms ~1859 -> ~2801`, `wait_ms ~1840 -> ~2777`로 증가했다.

원인 가설:

1. `wait`를 한 군데서 덜어낸 대신 queue backlog가 다른 batch와 token-end drain으로 이동했다.
2. persistent scratch와 deferred commit 자체는 correctness를 깨지 않았지만, 현재 그래프 크기에서는 in-flight work가 늘면서 오히려 end-to-end wall이 나빠졌다.
3. 즉 "wait를 미룬다"와 "wall이 줄어든다"는 같지 않다.

교훈:

1. deferred wait는 profiling bucket 하나를 예쁘게 만드는 것만으로 채택하지 않는다.
2. token-end drain까지 포함한 wall clock으로 판단해야 한다.
3. queue ordering 기반 async 실험은 향후에도 가능하지만, 현재 구현은 기본 경로로 채택할 수 없다.

관련 파일:

- `Mac/gpu/src/module/qwen_block.mm`
- `Mac/gpu/src/model/qwen_causal_lm.cpp`
- `Mac/gpu/src/metal/metal_context.mm`

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
- `Mac/gpu/reports/quick/labeled_breakdown_lmhead_gateup.json`

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
- `Mac/gpu/reports/quick/labeled_breakdown_gateup_only.json`

## 11. Decode `Q/K/V/O` Q4 Attention Specialization (`16x4`)

상태: `rejected`

관찰:

1. `SOC_GPU_ENABLE_EXPERIMENTAL_Q4_ATTN_SPECIALIZED=1`로 `QProjDecodeQ4`, `KProjDecodeQ4`, `VProjDecodeQ4`, `OProjDecodeQ4`를 전용 `16x4` q4 kernel로 바꿨다.
2. `8-token` quick run은 좋아 보였다. `wall_ms ~619 -> ~478`, `wait_ms ~582 -> ~457`, `DecodeBlockAttentionBatch gpu_ms ~56.0 -> ~48.9`.
3. 하지만 `32-token` full benchmark 3-run 평균은 기존 `q4 MLP specialized` 기준 `25.37 tok/s`에서 `23.86 tok/s`로 내려갔다.
4. 즉 quick improvement가 full decode throughput으로 이어지지 않았다.

원인 가설:

1. `outputs-per-thread=4`, `threadgroup width=16` 조합이 short run에서는 submit/wait를 줄였지만, long decode에서는 register pressure와 cache pressure가 커졌다.
2. 특히 `OProj`를 `Q/K/V`와 같은 specialization에 묶은 점이 full decode에서 불리하게 작용했을 가능성이 높다.
3. 이것은 attention 특화 자체의 문제라기보다, `Q/K/V/O`를 한꺼번에 `16x4`로 몰아넣는 방식이 장기 decode에서 맞지 않았다는 뜻이다.

교훈:

1. attention projection 특화는 `Q/K/V`와 `OProj`를 분리해서 다뤄야 한다.
2. quick run만으로 승격하지 않는다.
3. reviewer 의견대로 `32x2` 또는 `Q/K/V only`부터 좁게 시작하는 편이 안전하다.

관련 파일:

- `Mac/gpu/src/op/affine_qmm_op.mm`
- `Mac/gpu/shaders/gpu_kernels.metal`
- `Mac/gpu/reports/quick/q4_blockattn_logitbatch_qmmmlp_qmmattn_8tok.json`
- `Mac/gpu/reports/full_gpu_vs_pytorch_q4_blockattn_logitbatch_qmmmlp_qmmattn/summary.json`

## 12. Decode `Q/K/V` Q4 Attention Specialization (`32x2`)

상태: `rejected`

관찰:

1. reviewer 조언대로 `Q/K/V`만 `32x2` q4 kernel로 좁힌 variant를 다시 시도했다.
2. `8-token` hang check는 통과했지만, 실행 결과가 비정상적이었다.
3. `gpu_infer --max-new-tokens 8` output에 `generated_token_ids: [0]`이 나타났고, 일부 리포트는 timing이 전부 `0`으로 기록되기도 했다.
4. 즉 이 경로는 성능 이전에 correctness와 계측 신뢰성이 깨졌다.

원인 가설:

1. `32x2` 경로에서 column packing/indexing 또는 write-out 가정이 현재 q4 layout과 맞지 않았을 가능성이 높다.
2. 방법론의 문제라기보다, 현재 구현의 `Q/K/V only` specialization이 안전한 결과를 보장하지 못했다.
3. hang이 없다고 correctness가 보장되는 것은 아니다.

교훈:

1. attention specialization은 반드시 output token correctness와 timing integrity까지 같이 확인해야 한다.
2. `generated_token_ids`나 timing JSON이 조금이라도 이상하면 즉시 폐기한다.
3. 이후 attention 특화는 더 작은 실험 단위나 별도 검증 harness 없이 기본 경로 후보로 올리지 않는다.

관련 파일:

- `Mac/gpu/src/op/affine_qmm_op.mm`
- `Mac/gpu/shaders/gpu_kernels.metal`
- `Mac/gpu/reports/quick/q4_blockattn_logitbatch_qmmmlp_qmmattn2_8tok.json`
- `Mac/gpu/reports/decode_hang_check_q4_blockattn_logitbatch_qmmmlp_qmmattn2.json`

## 13. Softmax SIMD Cooperative Reduction

상태: `tested and reverted`

관찰:

1. `softmax_f32_rowwise`를 32-lane cooperative reduction kernel로 바꿔 `Softmax gpu_ms` 자체는 baseline `~22.83 ms`에서 `~7.95 ms`까지 줄었다.
2. kernel test와 real-bundle regression은 통과했고, 출력 correctness 문제도 없었다.
3. 하지만 `benchmark_full_gpu_vs_pytorch` 32-token 기준 3-run 평균은 full GPU throughput이 `~6.116 tok/s` baseline 대비 `~6.145 tok/s` 수준으로 사실상 동일했다.
4. 같은 측정에서 total `gpu_ms`는 `~1605 -> ~1694 ms`로 오히려 늘었고, `wait_ms`와 `command_buffer_count`도 구조적으로 그대로였다.

원인 가설:

1. 현재 엔진의 지배 병목은 softmax 단일 커널보다 decode projection과 per-op submit/wait overhead다.
2. softmax를 빠르게 만들어도 `17120` command buffer 구조와 decode matmul 비용이 그대로라 end-to-end wall 개선으로 거의 이어지지 않았다.
3. 즉 이 실험은 "kernel hotspot 개선"과 "실제 throughput 개선"이 다를 수 있다는 점을 다시 확인한 사례다.

교훈:

1. 앞으로 softmax 계열 작업은 standalone kernel speedup만으로 채택하지 않는다.
2. 다음 attention 최적화는 softmax 단독 교체보다 `AttentionScore -> Softmax -> AttentionValue`의 bounded submit reduction 또는 더 큰 attention-scope 재구성이 우선이다.
3. per-op GPU ms가 크게 좋아져도 total tok/s가 유의미하게 오르지 않으면 기본 경로에 넣지 않는다.

관련 파일:

- `Mac/gpu/src/op/softmax_op.mm`
- `Mac/gpu/shaders/gpu_kernels.metal`
- `Mac/gpu/reports/full_gpu_vs_pytorch_softmax_simd.md`
- `Mac/gpu/reports/full_gpu_vs_pytorch_softmax_simd_3runs.md`

## 14. Decode Final Logits Batch On Top Of Attention Output Batching

상태: `tested and kept experimental only`

관찰:

1. `SOC_GPU_ENABLE_EXPERIMENTAL_DECODE_FINAL_LOGITS_BATCH=1`를 현재 attention output batching 경로 위에 추가로 켰다.
2. command buffer 수는 `14516 -> 14485`로 소폭 줄었지만, 1-run 비교에서 full GPU throughput은 `~6.753 tok/s`에서 `~6.497 tok/s`로 오히려 내려갔다.
3. `DecodeFinalNormLmHeadBatch` 자체는 정상 동작했고 correctness 문제도 없었지만, total `gpu_ms`와 `wait_ms`는 개선되지 않았다.

원인 가설:

1. final RMSNorm + LMHead는 이미 command buffer 수가 적어서, submit 수를 조금 더 줄여도 전체 wall에 미치는 영향이 작다.
2. 반대로 LMHead stage의 큰 compute cost가 그대로라, 추가 batching 이득보다 queue ordering 변화가 더 크게 작용했을 가능성이 높다.
3. 현재 엔진에서는 final logits batching보다 attention output batching 쪽이 더 우선순위가 높다.

교훈:

1. remaining bounded batching work는 token 마지막 stage보다 decode 중반의 반복 hot path를 먼저 다뤄야 한다.
2. `SOC_GPU_ENABLE_EXPERIMENTAL_DECODE_FINAL_LOGITS_BATCH`는 유지하되, 기본 경로로 승격하지 않는다.

관련 파일:

- `Mac/gpu/src/model/qwen_causal_lm.cpp`
- `Mac/gpu/reports/full_gpu_vs_pytorch_attn_output_batch.md`
- `Mac/gpu/reports/full_gpu_vs_pytorch_attn_output_logitbatch.md`

## 15. Dense `LMHeadDecode` 4-Column Kernel On Top Of Current Bounded Decode Path

상태: `retested and kept experimental only`

관찰:

1. 이미 기본 경로에 올라간 `DecodeAttnPrepBatch`, `KVCacheBlitBatch`, `DecodeAttentionOutputBatch`, `DecodePostNormMlpBatch` 위에 `SOC_GPU_ENABLE_EXPERIMENTAL_LMHEAD_4COL=1`를 다시 올려 봤다.
2. 같은 시점 immediate 3-run baseline 비교에서 기본 경로는 `~13.389 tok/s`, `LMHead 4-col`은 `~13.074 tok/s`였다.
3. profiling상 `LMHeadDecode`도 `~218.7 ms -> ~420.5 ms`로 명확히 악화됐다.

원인 가설:

1. submit overhead가 크게 줄어든 현재 경로에서도, `LMHeadDecode`의 register pressure와 load pattern 손실이 여전히 더 크다.
2. 단일 stage 개선이 아니라 전체가 좋아 보였던 이전 1-run은 노이즈였다.

현재 정책:

1. 기본 경로에서는 사용하지 않는다.
2. `SOC_GPU_ENABLE_EXPERIMENTAL_LMHEAD_4COL=1`는 계속 실험 전용으로만 둔다.

관련 파일:

- `Mac/gpu/src/op/matmul_op.mm`
- `Mac/gpu/shaders/gpu_kernels.metal`
- `Mac/gpu/reports/full_gpu_vs_pytorch_lmhead4col_current_3runs.md`
- `Mac/gpu/reports/full_gpu_vs_pytorch_kvbatch_attnprep_postnorm_3runs_rerun.md`

## 16. Q4 `DownProjDecode` Specialized Kernel On Top Of Current Bounded Decode Path

상태: `rejected`

관찰:

1. `SOC_GPU_ENABLE_EXPERIMENTAL_Q4_DOWNPROJ_SPECIALIZED=1`를 현재 bounded q4 decode 경로 위에 다시 올렸다.
2. benchmark wrapper가 결과 JSON을 UTF-8로 읽지 못하고 실패했다.
3. 남은 산출물의 `generated_text`도 이미 깨져 있어서, 단순 성능 회귀가 아니라 출력 무결성 자체가 깨졌다.

원인 가설:

1. 현재 `affine_qmm_t_4bit_lmhead2`를 downproj에도 재사용하는 specialization이 decode output을 손상시키는 것으로 보인다.
2. 현재 bounded decode scheduling과 결합했을 때도 correctness 문제가 남아 있으므로, 성능 논의 이전에 배제해야 한다.

현재 정책:

1. 기본 경로에서는 절대 사용하지 않는다.
2. downproj 전용 q4 specialization은 별도 correctness 수정 전까지 다시 평가하지 않는다.

관련 파일:

- `Mac/gpu/src/op/affine_qmm_op.mm`
- `Mac/gpu/reports/full_gpu_vs_pytorch_q4_current_qmmdown/gpu_full_run_1.json`

## 17. Dense `DownProjDecode` Width-16 Override On Current Bounded Decode Path

상태: `rejected`

관찰:

1. 현재 dense 기본 경로 위에서 `SOC_GPU_ENABLE_EXPERIMENTAL_DOWNPROJ_WIDTH16=1`로 `DownProjDecode`만 `preferred_threadgroup_width=16`, `preferred_tile_columns=16`으로 좁혔다.
2. `make build-infer`와 `REAL_BUNDLE_MAX_NEW_TOKENS=8 make integration-real-bundle`는 통과했다.
3. 하지만 32-token full benchmark 1-run에서 immediate dense baseline은 `~13.389 tok/s`였고, 이 override를 켜면 `~12.090 tok/s`로 내려갔다.
4. 같은 측정에서 `DecodePostNormMlpBatch gpu_ms`도 `~483.57 ms -> ~613.00 ms`로 악화됐다.

원인 가설:

1. `bench_matmul`의 개별 shape 힌트와 실제 decode graph의 최적점이 달랐다.
2. `DownProjDecode`만 폭을 줄여도 threadgroup scheduling 이득보다 전체 MLP batch 내부의 locality 손실이 더 컸다.
3. 즉 현재 bounded dense path에서는 `DownProjDecode` width를 따로 좁히는 방향이 end-to-end로는 맞지 않았다.

현재 정책:

1. 기본 경로에서는 사용하지 않는다.
2. 해당 override 코드는 제거했다.

관련 파일:

- `Mac/gpu/src/module/qwen_mlp.mm`
- `Mac/gpu/reports/full_gpu_vs_pytorch_dense_detail_current.md`
- `Mac/gpu/reports/full_gpu_vs_pytorch_downproj16_current.md`

## 18. Q4 Decode `Q/K/V`-Only Specialization On Current Bounded Decode Path

상태: `rejected`

관찰:

1. `QProjDecodeQ4`, `KProjDecodeQ4`, `VProjDecodeQ4`만 기존 `affine_qmm_t_4bit_mlp2` `16x2` kernel로 우회하는 좁은 variant를 `SOC_GPU_ENABLE_EXPERIMENTAL_Q4_QKV_SPECIALIZED=1`로 시도했다.
2. `8-token` hang smoke는 통과했고, output도 즉시 깨지지는 않았다.
3. 하지만 현재 q4 baseline 32-token 1-run은 `~19.879 tok/s`였고, 같은 조건에서 variant는 `~15.236 tok/s`로 크게 악화됐다.
4. 같은 측정에서 `DecodeAttnPrepBatch gpu_ms`도 `~103.22 ms -> ~199.67 ms`로 거의 두 배 수준으로 증가했다.

원인 가설:

1. 현재 q4 bounded decode graph에서 `Q/K/V` projection은 `16x2` reuse kernel보다 기존 generic q4 kernel이 더 잘 맞는다.
2. `OProj`를 제외해도, attention prep batch 안에서 Q/K/V projection 세 개를 모두 이 kernel로 바꾸면 register pressure와 memory behavior가 오히려 나빠진다.
3. 과거 기록대로 attention q4 specialization은 quick smoke나 부분 단계가 아니라 full decode benchmark 기준으로만 판단해야 한다.

현재 정책:

1. 기본 경로에서는 사용하지 않는다.
2. 해당 variant 코드는 제거했다.

관련 파일:

- `Mac/gpu/src/op/affine_qmm_op.mm`
- `Mac/gpu/reports/decode_hang_check_q4_current_qkvspecialized.json`
- `Mac/gpu/reports/full_gpu_vs_pytorch_q4_current_qkvspecialized_baseline.md`
- `Mac/gpu/reports/full_gpu_vs_pytorch_q4_current_qkvspecialized.md`

## 19. Chunked Prefill Full-Token Upload Reuse

상태: `tested and reverted`

관찰:

1. `GenerationContext::Prefill()`에서 chunk마다 `std::vector<int>`를 다시 만들고 token buffer를 다시 쓰는 대신, full prompt token buffer를 한 번 올리고 chunk view만 재사용하는 variant를 시도했다.
2. 의도는 unified memory 환경에서 chunked prefill의 host copy / buffer write overhead를 줄이는 것이었다.
3. `test_generation_context`와 `SOC_GPU_PREFILL_STEP_SIZE=4 REAL_BUNDLE_MAX_NEW_TOKENS=1 make integration-real-bundle`는 통과했다.
4. 하지만 같은 조건 실측에서 `GPU context` tok/s는 개선되지 않았다. Raw prompt는 대략 `1.506 -> 1.505 tok/s`, chat template prompt는 `0.685 -> 0.680 tok/s` 수준으로 오히려 소폭 악화됐다.

원인 가설:

1. 현재 chunked prefill 병목은 host-side token upload보다 GPU-side prefill matmul/attention compute와 command-buffer scheduling이다.
2. full prompt를 한 번 쓰고 chunk view를 나누는 변화는 CPU bookkeeping은 줄여도 end-to-end wall을 움직일 만큼 크지 않았다.
3. 즉 unified memory라는 조건만으로 host upload 최적화가 바로 prefill tok/s 개선으로 이어지지는 않았다.

현재 정책:

1. 이 변경은 유지하지 않는다.
2. 다음 prefill 실험은 token upload 재사용보다 prefill attention/matmul working-set과 dispatch 구조를 직접 건드리는 쪽으로 간다.

관련 파일:

- `Mac/gpu/src/runtime/generation_context.cpp`
- `Mac/gpu/reports/test_real_bundle_regression_report.md`

## 20. Decode Hidden-Slot Hazard Relaxation

상태: `tested and reverted`

관찰:

1. prebuilt decode plan에서 hidden ping-pong slot overlap을 진짜 external hazard로 보지 않도록 완화해서, adjacent decode layer stage가 더 많이 같은 `batch_id`를 공유하게 하는 variant를 시도했다.
2. toy plan 기준 `test_generation_context`에서는 layer stage merge 자체는 확인됐다.
3. 하지만 real-model steady-state decode 측정에서는 tok/s가 개선되지 않았다. `SOC_GPU_ENABLE_EXPERIMENTAL_PREBUILT_DECODE_PLAN=1` baseline은 `15.336 tok/s`, 여기에 hidden-slot hazard relaxation까지 켠 variant는 `15.051 tok/s`였다.
4. full-run command buffer / encoder count도 둘 다 각각 `2680 / 8560`으로 같아서, 실제 실행 경로에서 의미 있는 dispatch reduction으로 이어지지 않았다.

원인 가설:

1. 현재 prebuilt decode plan의 metadata 병목은 hidden slot overlap 하나만의 문제가 아니거나, 그 완화가 downstream execution batching에 실질적으로 반영되지 않았다.
2. 즉 metadata 상 `batch_id` merge 가능성이 늘어도, real-model decode wall을 움직일 만큼의 command-buffer reduction은 나오지 않았다.

현재 정책:

1. 이 변경은 유지하지 않는다.
2. decode metadata 최적화는 다음에 plan/runtime 경계에서 실제 merged range가 command-buffer count를 줄이는지 먼저 계측 가능하게 만든 뒤 다시 시도한다.

관련 파일:

- `Mac/gpu/src/runtime/command_scheduler.cpp`
- `Mac/gpu/test/runtime/test_generation_context.mm`
- `Mac/gpu/reports/decode_hidden_relax_baseline.md`
- `Mac/gpu/reports/decode_hidden_relax_experimental.md`

## 21. Safe Decode Encoder Budget 8

상태: `tested and kept default-off`

관찰:

1. safe decode batch를 block-level bounded stream으로 재설계한 뒤, `SOC_GPU_COMMAND_STREAM_ENCODER_BUDGET=8`을 실제 runtime policy로 연결해서 oversized attention batch를 강제로 split하는 variant를 측정했다.
2. baseline reworked q4 safe decode steady-state는 `42.53 tok/s`, `76.26 command buffers/decode-token`이었고, 같은 경로에 budget `8`을 넣으면 `28.00 tok/s`, `104.26 command buffers/decode-token`으로 악화됐다.
3. end-to-end q4 safe decode도 baseline reworked path는 `30.89 tok/s` decode, `27.84 tok/s` total이었지만, budget `8` variant는 `27.27 tok/s` decode, `24.92 tok/s` total로 내려갔다.
4. encoder count 자체는 둘 다 `535~552 encoders/token` 수준으로 같아서, 이번 cap은 encoder density를 줄이지 못하고 command buffer fragmentation만 늘렸다.

원인 가설:

1. 현재 safe decode hot path의 병목은 oversize encoder count 자체보다 attention/MLP 내부 encoder 수가 많다는 구조 문제다.
2. budget `8`은 block attention을 `DecodeAttnPrepBatch`와 `DecodeBlockAttentionBatch`로 다시 쪼개서 wait/flush overhead만 추가했다.
3. 즉 command-stream encoder budget은 더 높은 cap 또는 다른 natural boundary와 결합해야 의미가 있다.

현재 정책:

1. runtime policy budget wiring과 bounded split hook은 유지한다.
2. 기본값은 계속 `0`(disabled)로 둔다.
3. aggressive low cap은 유지하지 않고, 다음 실험은 attention/MLP 내부 encoder 수 자체를 줄이는 방향으로 간다.

관련 파일:

- `Mac/gpu/src/module/qwen_attention.mm`
- `Mac/gpu/src/module/qwen_block.mm`
- `Mac/gpu/include/runtime/runtime_policy.h`
- `Mac/gpu/reports/decode_steady_state_q4_safe_batch_rework_baseline.md`
- `Mac/gpu/reports/decode_steady_state_q4_safe_batch_rework_budget8.md`
- `Mac/gpu/reports/full_gpu_vs_pytorch_q4_safe_batch_rework_baseline.md`
- `Mac/gpu/reports/full_gpu_vs_pytorch_q4_safe_batch_rework_budget8.md`
# Qwen3.5 GPU RMSNorm Regression

Date: 2026-04-02

Scope:
- `Mac/gpu/models/qwen3_5/qwen3_5_runner.cpp`
- `Mac/gpu/models/qwen3_5/qwen3_5_loader_adapter.cpp`

What was attempted:
- Move `post_attention_layernorm -> MLP` and `final_norm -> LMHead` onto the shared GPU `RmsNormOp`.
- Promote `qwen3_5` norm weights to device `float32` during load.

Observed result on real machine:
- first-token parity broke on the reference prompt `"Hello world"`
- expected argmax token: `0` / `"!"`
- regressed token: `220` / `" "`
- wall time also worsened to about `7.18 s`

Why it failed:
- `qwen3_5_runner.cpp` host reference path uses:
  - `output = input * inv_rms * (1 + weight)`
- shared GPU `RmsNormOp` kernel uses:
  - `output = input * inv_rms * weight`
- this is not a generic scheduler or batching issue
- it is a contract mismatch in how `qwen3_5` interprets norm weights

Correct conclusion:
- the problem is not “GPU RMSNorm is impossible”
- the problem is “reusing the shared RMSNorm operator without a `qwen3_5`-compatible weight contract changes the math”

Current policy:
- default `qwen3_5` runner keeps `post_attention RMSNorm` and `final RMSNorm` on the validated host path
- any future GPU RMSNorm re-enable must first do one of:
  - add a `qwen3_5`-specific RMSNorm kernel/parameter mode
  - or prepack a mathematically equivalent device-side norm weight contract with explicit parity validation

# Qwen3.5 Narrow Decode Projection Regression

Date: 2026-04-02

Scope:
- `Mac/gpu/models/qwen3_5/qwen3_5_runner.cpp`

What was attempted:
- Re-enable only the narrowest decode-side GPU slices with:
  - `SOC_GPU_ENABLE_EXPERIMENTAL_QWEN3_5_DECODE_EMBEDDING_GPU=1`
  - `SOC_GPU_ENABLE_EXPERIMENTAL_QWEN3_5_DECODE_OUTPUT_PROJECTION_GPU=1`
- Keep `q/k/v`, DeltaNet `in_proj_*`, RMSNorm, RoPE, and recurrent state update on the host path.

Observed result on real machine:
- argmax parity on `"Hello world"` was preserved
- generated token ids stayed `[0, 271, 9419, 11]`
- but throughput regressed versus the best current experimental path
- measured report:
  - `Mac/gpu/reports/qwen3_5_4b_4tok_gpu_decodeproj.json`
  - prefill wall `6051.9 ms`
  - decode wall `7800.5 ms`
  - decode throughput `0.513 tok/s`

Why it failed:
- the change only moved cheap boundary ops
- it did not move the decode-hot projection families that dominate the current hybrid runner
- upload / readback overhead remained, while extra GPU dispatches were added

Correct conclusion:
- the problem is not “decode-only GPU slices are impossible”
- the problem is “re-enabling only embedding + output projections does not attack the real decode bottleneck”

Current policy:
- keep this path experimental only
- do not promote it as the default `qwen3_5` decode configuration
- prefer validating the broader projection path on multiple prompts before deciding which projection subsets are safe enough to keep
