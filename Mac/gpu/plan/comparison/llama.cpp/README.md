# llama.cpp Comparison

## Repo Snapshot

- Repository: `ggml-org/llama.cpp`
- Local snapshot: `/tmp/soc_compare/llama.cpp`
- Inspected commit: `0fcb376`

## 왜 이 레포를 보는가

우리 코드와 가장 직접적으로 비교할 수 있는 Apple Silicon Metal inference 엔진이다. 특히 `ggml`의 graph executor, command buffer 분할, Metal encoder 배치, hazard 관리 방식은 지금 우리 코드의 병목과 fault 이슈를 다시 설계할 때 가장 중요한 참고점이다.

## 직접 읽은 파일

- `ggml/src/ggml-metal/ggml-metal-context.m`
- `ggml/src/ggml-metal/ggml-metal-ops.cpp`

## 구조 요약

`llama.cpp`는 op를 하나씩 바로 호출하는 imperative runtime이 아니라, 먼저 `ggml_cgraph`를 만들고 그 graph를 Metal backend가 encode하는 구조다. 핵심 차이는 "실행 전에 dependency와 memory range를 알고 있다"는 점이다.

`ggml_metal_context.m` 기준으로 Metal backend는:

- device, queue, library, event, dispatch queue를 분리 관리한다.
- `GGML_METAL_MAX_COMMAND_BUFFERS 8`로 그래프 하나를 여러 command buffer로 나눈다.
- `use_fusion`, `use_concurrency`, `use_graph_optimize`를 별도 토글로 둔다.
- 에러가 난 command buffer가 생기면 backend 전체를 failed state로 두고 더 이상 같은 backend를 재사용하지 않는다.

`ggml-metal-ops.cpp` 기준으로 encode 단계는:

- graph node 범위를 받아 non-empty node만 추린 뒤 encode한다.
- 현재 node가 이전 node들과 병렬로 갈 수 있는지 `mem_ranges`로 판정한다.
- 이전 src/dst와 충돌하면 memory barrier를 넣고 range 추적 상태를 reset한다.
- 충돌이 없으면 같은 encoder 흐름 안에서 concurrent하게 계속 encode한다.
- fusion은 "가능한 op 패턴"에만 제한적으로 적용한다. 무조건 giant fusion을 하지 않는다.

## 우리 코드와의 직접 비교

### 1. 실행 단위

`llama.cpp`

- graph 전체를 알고 시작한다.
- op scheduling 전에 dependency와 memory overlap을 본다.
- 여러 op를 한 encoder / command buffer 안에 넣되, 충돌 시 즉시 barrier를 넣는다.

우리 코드

- [`qwen_causal_lm.cpp`](/Volumes/990pro/Documents/SoC/Mac/gpu/src/model/qwen_causal_lm.cpp)에서 블록을 순차적으로 실행한다.
- 기본 경로는 사실상 per-op command buffer다.
- experimental `CommandStream`은 layer/full 범위를 통째로 묶었고, dependency-aware barrier 없이 giant batch가 되기 쉬웠다.
- 그 결과 M4에서 `WindowServer` fault를 유발했다.

판단:

- 문제는 "batching 자체"보다 "dependency를 모르는 giant batching"이다.
- `llama.cpp`는 batching을 하되, graph-level hazard tracking을 전제로 한다.

### 2. command buffer 전략

`llama.cpp`

- command buffer를 최대 8개까지 쓴다.
- encode를 main thread + worker threads로 나눌 수 있다.
- 하지만 무제한으로 크게 묶지 않는다.
- 분할 기준은 graph/node range다.

우리 코드

- [`metal_context.mm`](/Volumes/990pro/Documents/SoC/Mac/gpu/src/metal/metal_context.mm)에서 command buffer 하나를 commit/wait하고 profiling을 모은다.
- [`qwen_causal_lm.cpp`](/Volumes/990pro/Documents/SoC/Mac/gpu/src/model/qwen_causal_lm.cpp)에서 full-range 또는 layer-range `CommandStream`을 시도했지만 안전하지 않았다.
- 현재 stable baseline은 per-op finalize다.

판단:

- 지금 필요한 것은 `llama.cpp`식 "bounded multi-buffer scheduling"이지, full-range batch가 아니다.
- 특히 decode에서는 `DownProjDecode`, `OProjDecode`, `LMHeadDecode`가 병목이라 그 주변만 제한적으로 묶는 방식이 맞다.

### 3. op encode / fusion

`llama.cpp`

- op encode는 graph node 단위다.
- noop op를 early-skip한다.
- fusion은 op 패턴 검사 후 가능할 때만 한다.
- concurrency는 memory range check를 통과한 node들 사이에서만 허용한다.

우리 코드

- [`matmul_op.mm`](/Volumes/990pro/Documents/SoC/Mac/gpu/src/op/matmul_op.mm)에서 op별로 즉시 pipeline 선택 후 dispatch한다.
- [`qwen_attention.mm`](/Volumes/990pro/Documents/SoC/Mac/gpu/src/module/qwen_attention.mm), [`qwen_mlp.mm`](/Volumes/990pro/Documents/SoC/Mac/gpu/src/module/qwen_mlp.mm)에서 함수 호출 순서가 곧 encode 순서다.
- 일부 fusion 시도는 했지만 graph-aware legality check가 없다.

판단:

- 우리 쪽은 "fuse 가능 여부"보다 "지금 묶으면 안전한가"를 먼저 판단해야 한다.
- decode path에서 micro-fusion보다 scheduler와 dispatch count 축소가 먼저다.

### 4. KV cache / sampling

`llama.cpp`

- sampling은 Metal graph 내부 핵심 경로보다 분리되어 있다.
- backend는 tensor ops에 집중하고 sampling은 상위 레이어에서 다룬다.

우리 코드

- [`sampler.cpp`](/Volumes/990pro/Documents/SoC/Mac/gpu/src/runtime/sampler.cpp)에서 GPU sampler를 만들었지만 M4에서 CPU fallback이 더 빨랐다.
- [`kv_cache.mm`](/Volumes/990pro/Documents/SoC/Mac/gpu/src/runtime/kv_cache.mm)에서는 `Private` buffer에 blit append를 사용한다.

판단:

- sampling을 GPU에 억지로 넣는 것보다, matmul 병목을 먼저 줄이는 현재 방침이 맞다.
- KV append는 지금 방식이 안정적이지만, decode 전체에서 append dispatch 수를 줄일 여지는 남아 있다.

## op 관점 정리

`llama.cpp`가 잘하는 점:

- graph 수준에서 op를 본다.
- memory conflict를 추적한다.
- bounded command buffer 분할을 한다.
- fusion과 concurrency를 분리된 옵션으로 관리한다.

우리 코드가 다른 점:

- op 함수 호출이 곧 스케줄러다.
- hazard tracking이 없다.
- profiling은 좋지만 scheduling layer가 얇다.
- command buffer 수가 많고 dispatch가 잘게 쪼개져 있다.

## 바로 적용 가능한 아이디어

### 채택 후보 1. decode 전용 bounded scheduler

`CommandStream`을 다시 키되 full/layer 모드가 아니라 "허용된 인접 op 묶음"만 encode한다.

예시:

- `RmsNorm -> QKV proj` 범위
- `AttentionScore -> Softmax -> AttentionValue`
- `PostAttnNorm -> GateProj/UpProj`

중요한 점:

- `llama.cpp`처럼 memory hazard table 없이 전역 batch를 만들면 안 된다.
- 먼저 whitelist 기반으로만 묶고, 묶음마다 finalize한다.

### 채택 후보 2. tensor memory range tracker

graph까지 안 가더라도 최소한 `DeviceTensor buffer + offset + size` 기준 overlap tracker를 추가한다. 이게 있어야 "같은 command buffer 안에서 어떤 op를 함께 encode해도 되는가"를 판단할 수 있다.

### 채택 후보 3. profiling label 기반 scheduler feedback

지금 이미 `MatMul`, `RMSNorm`, `KVCacheBlit` 라벨 profiling이 있다. 이걸 scheduler 정책과 연결해서 "decode path에서 자주 반복되고 안전한 라벨 조합"만 실험 대상으로 삼는다.

## 우리 코드에 대한 결론

`llama.cpp`와 비교하면, 지금 우리 쪽 성능 한계는 단순히 Metal kernel 품질만의 문제가 아니다. scheduler 부재, hazard-aware batching 부재, 너무 많은 per-op finalize가 같이 묶여 있다. 반대로 giant batching이 fault를 냈으므로, 다음 방향은 "graph-aware 또는 최소한 hazard-aware bounded batching"이다.

즉 `llama.cpp`에서 배워야 할 핵심은 "많이 묶는다"가 아니라 "안전하다고 증명된 범위만 묶는다"이다.
