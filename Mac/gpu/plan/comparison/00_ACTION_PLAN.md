# Comparison-Based Action Plan

## 비교에서 공통으로 나온 결론

### 1. giant batching은 금지

- `llama.cpp`는 graph-aware / hazard-aware batching만 한다.
- `flash-moe`는 수동 encoder 결합이 성능 이득이 없거나 손해였다.
- 우리 코드는 full/layer `CommandStream`에서 실제 fault까지 났다.

결론:

batching은 계속 실험하되, `안전한 op 묶음`과 `hazard 추적` 없이 넓게 적용하면 안 된다.

### 2. end-to-end 검증 없는 micro-optimization은 채택하지 않는다

- `flash-moe`는 isolated win이 full pipeline에서 깨지는 예시를 대량으로 남겼다.
- 우리도 같은 유형의 회귀를 이미 여러 번 겪었다.

결론:

새 커널은 반드시:

- microbench
- quick decode benchmark
- full GPU vs PyTorch benchmark

세 단계를 다 통과해야 채택한다.

### 3. runtime policy 계층을 강화해야 한다

- `mlx-lm`, `mlx-vlm`은 cache, prompt cache, prefill step, memory budget를 runtime policy로 분리한다.
- 우리 코드는 kernel과 runtime policy가 상대적으로 강하게 결합돼 있다.

결론:

지금부터는 커널 튜닝과 함께 runtime policy 정리를 병행해야 한다.

## 바로 실행할 수정 아이디어

### 아이디어 A. hazard-aware bounded scheduler

내용:

- giant `CommandStream` 대신 decode 전용 whitelist batch를 도입한다.
- 첫 단계는 다음 세 묶음만 허용한다.
  - `InputNorm -> QProj/KProj/VProj`
  - `AttentionScore -> Softmax -> AttentionValue`
  - `PostAttnNorm -> GateProj/UpProj`

이유:

- `llama.cpp`에서 배운 핵심은 "많이 묶기"가 아니라 "안전하게 묶기"다.
- 현재 dispatch 수가 과도하므로, 안전 범위 내 축소가 필요하다.

리스크:

- overlap 판정이 틀리면 fault가 재발할 수 있다.

채택 조건:

- `integration-real-bundle` 통과
- quick benchmark 개선
- WindowServer/GPU fault 0건

### 아이디어 B. decode matmul shape-specialized kernel

내용:

- `DownProjDecode`, `OProjDecode`, `LMHeadDecode` 전용 kernel variant를 추가한다.
- generic heuristic 대신 shape-specialized path를 둔다.
- 우선은 `transpose_rhs=true` 유지 + vectorized RHS load 실험을 한다.

이유:

- 현재 실측 병목 대부분이 이 세 projection이다.
- `flash-moe`처럼 특정 data layout/shape에 맞춘 커널이 실전 이득을 줄 가능성이 가장 높다.

리스크:

- microbench와 full benchmark 결과가 엇갈릴 수 있다.

채택 조건:

- quick run에서 해당 label GPU ms 감소
- full benchmark tok/s 증가

### 아이디어 C. KV / prefill runtime policy 분리

내용:

- future `RotatingKVCache` 인터페이스를 먼저 열어 둔다.
- prefill step-size option을 추가한다.
- prompt cache 저장/복원 경로를 설계한다.

이유:

- `mlx-lm`/`mlx-vlm`에서 가장 강한 교훈은 runtime policy 분리다.
- long prompt 실험과 재현성 있는 benchmark에 도움이 된다.

리스크:

- 즉시 tok/s를 크게 올리지는 못할 수 있다.

채택 조건:

- 코드 경로가 복잡해지지 않고 안정성 유지
- prefill memory peak 제어 가능

## 우선순위

1. 아이디어 B
2. 아이디어 A
3. 아이디어 C

이 순서가 맞다. 현재 목표가 `PyTorch MPS 50 tok/s`를 넘는 것이므로, 먼저 가장 큰 병목인 decode matmul을 깎아야 한다. scheduler 재설계는 그 다음이다. cache/prefill policy는 중기 과제다.

## agent 검토 요청 포인트

- `llama.cpp` 관점: bounded scheduler 범위가 적절한가
- `flash-moe` 관점: decode shape specialization이 bandwidth bottleneck에 실제 도움이 되는가
- `mlx-lm` 관점: runtime policy 분리가 지금 시점에 과한지
- `mlx-vlm` 관점: prefill step-size를 먼저 넣는 편이 안정성 면에서 유리한지

## agent 검토 결과

### `llama.cpp` agent 요약

- giant batching이 아니라 dependency-aware graph partition이 핵심이다.
- `row_count == 1` decode path는 별도 `mul_mv` 성격으로 더 세분화할 가치가 있다.
- sampler를 다시 GPU로 올리려면 standalone top-k보다 제한된 sampler chain 방식이 안전하다.

### `flash-moe` agent 요약

- 입력 vector cache, SIMD reduction, 작은 범위 batched kernel은 채택 가치가 있다.
- giant command buffer는 다시 시도하면 안 된다.
- 우리 쪽 최우선은 `MatMul` 병목 완화이며, `threadgroup` input cache + vectorized RHS load가 가장 직접적인 후보다.

### `mlx-lm` agent 요약

- `recommended_max_working_set_size`를 실제 정책에 연결해야 한다.
- `RotatingKVCache`, prompt cache, memory-budget-aware runtime은 중기 과제다.
- 지금 시점에서 성능 1순위는 아니지만 구조 정리 과제로 유지한다.

### `mlx-vlm` agent 요약

- 긴 prefill은 chunked prefill / `prefill_step_size`가 안전장치 역할을 한다.
- giant async overlap은 금지하고, 작은 비동기 경계만 허용해야 한다.
- core LM path와 입력 준비 계층 분리는 장기적으로 필요하다.

## 합의

1. 지금 바로 할 일은 `decode matmul shape specialization`
2. 그 다음은 `hazard-aware bounded scheduler`
3. `KV / prefill / prompt cache`는 안정성 구조 정리 과제로 병행

## 이번 턴에서 실제 반영한 수정

### 변경 내용

- [`shaders/gpu_kernels.metal`](/Volumes/990pro/Documents/SoC/Mac/gpu/shaders/gpu_kernels.metal)에 decode 전용 `vec4` tiled kernel 두 개를 추가했다.
  - `matmul_f32_decode_tiled_vec4`
  - `matmul_f32_f16rhs_decode_tiled_vec4`
- [`src/op/matmul_op.mm`](/Volumes/990pro/Documents/SoC/Mac/gpu/src/op/matmul_op.mm)에서 `decode + transpose_rhs + tiled + inner_dim % 4 == 0 + large column_count` 조건일 때만 이 경로를 선택하도록 했다.
- [`src/metal/metal_context.mm`](/Volumes/990pro/Documents/SoC/Mac/gpu/src/metal/metal_context.mm)에서 stale `metallib`가 새 kernel을 놓치지 않도록 required kernel 목록을 갱신했다.

### 안전성 검증

- `make build-infer` 통과
- `make integration-real-bundle REAL_BUNDLE_MAX_NEW_TOKENS=1` 통과

### quick decode 실측 결과

비교 기준:

- baseline: [`reports/quick/labeled_breakdown.json`](/Volumes/990pro/Documents/SoC/Mac/gpu/reports/quick/labeled_breakdown.json)
- vec4 path: [`reports/quick/labeled_breakdown_vec4.json`](/Volumes/990pro/Documents/SoC/Mac/gpu/reports/quick/labeled_breakdown_vec4.json)

핵심 변화:

- total `gpu_ms`: `602.944 -> 424.887`
- total `wall_ms`: `1736.46 -> 1508.46`
- `DownProjDecode`: `119.624 -> 58.676`
- `OProjDecode`: `79.546 -> 39.927`
- `QProjDecode`: `41.702 -> 25.065`
- `KProjDecode`: `39.594 -> 19.995`
- `VProjDecode`: `39.581 -> 19.838`
- `LMHeadDecode`: `55.973 -> 52.719`

판단:

- 이 변경은 quick decode 기준으로 명확한 개선이다.
- 특히 `OProjDecode`와 `DownProjDecode`가 거의 절반 수준으로 줄었다.
- `LMHeadDecode`는 개선 폭이 작으므로 다음엔 LM head 전용 kernel 분리가 필요하다.

### short full benchmark 결과

- artifact: [`reports/full_gpu_vs_pytorch/summary.json`](/Volumes/990pro/Documents/SoC/Mac/gpu/reports/full_gpu_vs_pytorch/summary.json)
- 조건: `max_new_tokens=32`, `gpu_runs=1`, `pytorch_runs=1`, `dtype=float16`
- C++ full GPU: `5.30 tok/s`
- PyTorch MPS fp16 decode: `46.7 tok/s`

해석:

- 여전히 PyTorch와 격차가 크다.
- 하지만 현재 hot path 기준으론 `decode matmul specialization`이 실제로 먹힌다는 근거를 확보했다.

## 다음 수정 우선순위

1. `transformers`의 `MetalLinear / affine_qmm_t`와 같은 low-bit fused linear 경로 설계
2. giant batching 없이 `hazard-aware bounded scheduler` 설계 시작
3. `KV / prefill / prompt cache` runtime policy 분리

## 추가 비교: `transformers`

- 문서: [`comparison/transformers/README.md`](/Volumes/990pro/Documents/SoC/Mac/gpu/plan/comparison/transformers/README.md)

핵심 결론:

- `transformers`는 우리처럼 직접 Metal runtime을 짜지 않는다.
- Apple Silicon fast path는 `PyTorch MPS`, `SDPA`, 그리고 `MetalLinear + affine_qmm_t` 같은 low-bit Metal kernel integration에서 나온다.
- 즉 PyTorch를 정말 넘기려면, 지금의 dense decode matmul 최적화 다음 단계는 결국 `low-bit fused qmm`다.

## 2026-04-01 GitHub 재감사 요약

이번에는 comparison 문서에 적혀 있던 판단을 GitHub 메인라인 구현으로 다시 검증했다. 대상 4개 repo는 `llama.cpp`, `flash-moe`, `mlx-lm`, `mlx-vlm`이다. 보조 비교로 `transformers`의 Metal quantization도 다시 확인했다.

### 공통 결론 1. prefill과 decode를 같은 정책으로 다루면 안 된다

- `llama.cpp`는 `ubatch + graph reserve`로 prefill을 다루고, decode는 hazard-aware graph encode로 다룬다.
- `mlx-lm`, `mlx-vlm`은 `prefill_step_size`를 별도 노브로 두고, decode는 generation stream 위에서 따로 돈다.
- `flash-moe`도 사실상 decode-heavy 엔진이라, 짧은 seq에서 GPU attention을 아예 끄고 길이 조건을 둔다.

결론:

우리도 `prefill_step_size`와 `decode stage plan`을 분리해야 한다.

### 공통 결론 2. decode 최적화의 핵심은 작은 안전 경계 또는 low-bit fused linear다

- `llama.cpp`는 real mem range hazard tracking으로 안전한 범위만 묶는다.
- `flash-moe`는 low-bit fused matvec와 stage 구조화로 decode를 줄인다.
- `transformers`의 Apple Silicon fast path도 결국 `MetalLinear + affine_qmm_t`다.

결론:

우리 다음 2축은 계속 같다.

1. `real hazard tracker`
2. `low-bit fused decode projection`

### 공통 결론 3. runtime policy가 아직 우리 쪽에서 가장 약하다

- `mlx-lm`은 prompt cache, rotating/quantized KV, working set budget이 있다.
- `mlx-vlm`은 input preparation, embeddings prefill, prefill/completion batch knob가 있다.
- 우리는 kernel과 runtime policy가 아직 더 강하게 결합돼 있다.

결론:

`Mac/gpu`의 중기 과제는 새 kernel만이 아니라 아래 세 가지다.

1. `prefill_step_size`
2. `prompt cache artifact`
3. `KV policy abstraction`

## 적용 우선순위 갱신

1. `real range hazard tracker + decode plan`
2. `prefill_step_size + prompt cache`
3. `low-bit fused decode projection`을 dense/q4 공용 설계로 재정리
4. 이후에만 추가 shape-specialized dense kernel을 본다

이번 GitHub 재감사 기준으로, 현재 우리 문서의 방향 자체는 크게 틀리지 않았다. 다만 빠진 것은 `prefill/decode 분리`, `prompt cache`, `real hazard range` 세 축이고, 다음 반영은 이 세 가지를 중심으로 잡는 것이 맞다.

## 이번 턴의 추가 실험 결과

### `LMHeadDecode` 4-col kernel

- quick decode 기준 회귀
- 기본 비활성화
- `SOC_GPU_ENABLE_EXPERIMENTAL_LMHEAD_4COL=1`일 때만 실험

### `Gate/Up` fused decode kernel

- quick decode 기준 회귀
- 기본 비활성화
- `SOC_GPU_ENABLE_EXPERIMENTAL_FUSED_GATE_UP=1`일 때만 실험

### 현재 기본 경로

- 채택 유지: decode vec4 matmul specialization
- 실험만 유지: `LMHead 4-col`, `Gate/Up fused`

## 이번 턴 결과: `q4 decode projections`

### 구현

- exporter:
  - [`LLM_interpreter/convert_py_to_cpp.py`](/Volumes/990pro/Documents/SoC/LLM_interpreter/convert_py_to_cpp.py)
  - 새 flag: `--metal-quantize-decode-weights`
  - `lm_head`에 더해 `q/k/v/o`, `gate/up/down`의 추가 `qweight/scales/qbiases` copy를 export
- runtime:
  - [`src/op/affine_qmm_op.mm`](/Volumes/990pro/Documents/SoC/Mac/gpu/src/op/affine_qmm_op.mm)
  - [`shaders/gpu_kernels.metal`](/Volumes/990pro/Documents/SoC/Mac/gpu/shaders/gpu_kernels.metal)
  - residual / SiLU를 decode q4 경로에서도 처리 가능하게 확장
- module integration:
  - [`src/module/qwen_attention.mm`](/Volumes/990pro/Documents/SoC/Mac/gpu/src/module/qwen_attention.mm)
  - [`src/module/qwen_mlp.mm`](/Volumes/990pro/Documents/SoC/Mac/gpu/src/module/qwen_mlp.mm)
  - [`src/model/qwen_causal_lm.cpp`](/Volumes/990pro/Documents/SoC/Mac/gpu/src/model/qwen_causal_lm.cpp)
  - env: `SOC_GPU_ENABLE_EXPERIMENTAL_Q4_DECODE=1`

### 실기기 결과

- quick 8-token:
  - baseline vec4: [`reports/quick/labeled_breakdown_vec4.json`](/Volumes/990pro/Documents/SoC/Mac/gpu/reports/quick/labeled_breakdown_vec4.json)
  - q4 decode: [`reports/quick/q4_decode_8tok.json`](/Volumes/990pro/Documents/SoC/Mac/gpu/reports/quick/q4_decode_8tok.json)
  - total `gpu_ms`: `424.887 -> 255.584`
  - total `wall_ms`: `1508.46 -> 984.534`
- full benchmark 32-token:
  - dense baseline: [`reports/full_gpu_vs_pytorch/summary.json`](/Volumes/990pro/Documents/SoC/Mac/gpu/reports/full_gpu_vs_pytorch/summary.json)
  - q4 decode: [`reports/full_gpu_vs_pytorch_q4_decode/summary.json`](/Volumes/990pro/Documents/SoC/Mac/gpu/reports/full_gpu_vs_pytorch_q4_decode/summary.json)
  - C++ full GPU: `6.12 -> 9.65 tok/s`
  - PyTorch MPS fp16: `52.86 tok/s`

### 해석

- `low-bit fused qmm` 방향 자체는 맞다.
- 하지만 `command_buffer_count`가 여전히 `17120`이라서, matmul GPU 시간만 줄여도 PyTorch 격차를 다 메우진 못한다.
- 남은 1순위는 `hazard-aware bounded scheduler`다.

### 다음 우선순위

1. `full-range`가 아닌 `bounded` decode scheduler
2. encoder count cap / flush policy를 가진 `safe CommandStream`
3. q4 decode path를 기본값으로 올릴지, 품질 기준을 먼저 둘지 결정

## 이번 턴 결과: `q4 decode + safe micro-batching`

### 구현

- 새 env:
  - `SOC_GPU_ENABLE_EXPERIMENTAL_Q4_DECODE=1`
  - `SOC_GPU_ENABLE_EXPERIMENTAL_SAFE_DECODE_BATCH=1`
- stage batching:
  - attention prep: `Q/K/V proj + q/k norm + rope`
  - attention context: `attention score + softmax + value + o_proj`
  - post-attn: `post norm + mlp`
- 구현 파일:
  - [`src/module/qwen_attention.mm`](/Volumes/990pro/Documents/SoC/Mac/gpu/src/module/qwen_attention.mm)
  - [`src/module/qwen_mlp.mm`](/Volumes/990pro/Documents/SoC/Mac/gpu/src/module/qwen_mlp.mm)
  - [`src/module/qwen_block.mm`](/Volumes/990pro/Documents/SoC/Mac/gpu/src/module/qwen_block.mm)

### 실측

- quick 8-token:
  - q4 decode only: [`reports/quick/q4_decode_8tok.json`](/Volumes/990pro/Documents/SoC/Mac/gpu/reports/quick/q4_decode_8tok.json)
  - q4 decode + safe batch: [`reports/quick/q4_decode_safe_batch_8tok.json`](/Volumes/990pro/Documents/SoC/Mac/gpu/reports/quick/q4_decode_safe_batch_8tok.json)
  - `wall_ms`: `984.534 -> 628.509`
  - `gpu_ms`: `255.584 -> 191.310`
  - `command_buffer_count`: `4280 -> 1732`
- full benchmark 32-token:
  - q4 decode only: [`reports/full_gpu_vs_pytorch_q4_decode/summary.json`](/Volumes/990pro/Documents/SoC/Mac/gpu/reports/full_gpu_vs_pytorch_q4_decode/summary.json)
  - q4 decode + safe batch: [`reports/full_gpu_vs_pytorch_q4_decode_safe_batch/summary.json`](/Volumes/990pro/Documents/SoC/Mac/gpu/reports/full_gpu_vs_pytorch_q4_decode_safe_batch/summary.json)
  - C++ full GPU: `9.65 -> 16.53 tok/s`
  - PyTorch MPS fp16: `54.05 tok/s`

### 검증

- `make build-infer`
- `make build-tests`
- `SOC_GPU_ENABLE_EXPERIMENTAL_Q4_DECODE=1 SOC_GPU_ENABLE_EXPERIMENTAL_SAFE_DECODE_BATCH=1 make integration-real-bundle REAL_BUNDLE_MAX_NEW_TOKENS=1`

### 해석

- `low-bit fused qmm`과 `bounded decode scheduler` 둘 다 방향이 맞다.
- giant batching 없이도 실기기에서 command buffer 수를 유의미하게 줄일 수 있었다.
- 다음 단계는 이 stage batching을 `prebuilt decode plan`으로 일반화하는 것이다.

## 이번 턴 결과: `prebuilt decode plan`

### 구현

- 새 env:
  - `SOC_GPU_ENABLE_EXPERIMENTAL_PREBUILT_DECODE_PLAN=1`
- scheduler가 decode stage plan을 캐시한다.
- decode hidden state는 persistent ping-pong buffer 2개를 재사용한다.
- abstract slot 기반 hazard tracker로 stage batch id를 계산한다.
- 관련 파일:
  - [`include/runtime/command_scheduler.h`](/Volumes/990pro/Documents/SoC/Mac/gpu/include/runtime/command_scheduler.h)
  - [`src/runtime/command_scheduler.cpp`](/Volumes/990pro/Documents/SoC/Mac/gpu/src/runtime/command_scheduler.cpp)
  - [`src/model/qwen_causal_lm.cpp`](/Volumes/990pro/Documents/SoC/Mac/gpu/src/model/qwen_causal_lm.cpp)

### 실측

- quick 8-token:
  - safe batch only: [`reports/quick/q4_decode_safe_batch_8tok.json`](/Volumes/990pro/Documents/SoC/Mac/gpu/reports/quick/q4_decode_safe_batch_8tok.json)
  - safe batch + plan: [`reports/quick/q4_decode_safe_batch_plan_8tok.json`](/Volumes/990pro/Documents/SoC/Mac/gpu/reports/quick/q4_decode_safe_batch_plan_8tok.json)
  - `wall_ms`: `628.509 -> 661.238`
  - quick 기준 약간 회귀
- full benchmark 32-token:
  - safe batch only: [`reports/full_gpu_vs_pytorch_q4_decode_safe_batch/summary.json`](/Volumes/990pro/Documents/SoC/Mac/gpu/reports/full_gpu_vs_pytorch_q4_decode_safe_batch/summary.json)
  - safe batch + plan: [`reports/full_gpu_vs_pytorch_q4_decode_safe_batch_plan/summary.json`](/Volumes/990pro/Documents/SoC/Mac/gpu/reports/full_gpu_vs_pytorch_q4_decode_safe_batch_plan/summary.json)
  - C++ full GPU: `16.53 -> 16.86 tok/s`

### 판단

- 구조 방향은 맞지만, 현재 이득은 작다.
- `abstract slot` hazard model은 너무 약하다.
- 따라서 `prebuilt decode plan`은 지금 단계에서는 기본값이 아니라 `experimental` 유지가 맞다.
