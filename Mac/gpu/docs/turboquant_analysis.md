# TurboQuant 분석 및 qwen3_5-mlx 통합 관점 정리

## 개요

이 문서는 GitHub 저장소 `0xSero/TurboQuant`를 기준으로 TurboQuant의 목적, 방법론, 구현 구조, 실제 제약, 그리고 `qwen3_5-mlx`에 합칠 때의 현실적인 적용 범위를 정리한다.

- 원본 repo: `https://github.com/0xSero/TurboQuant`
- 로컬 clone 경로: `/Volumes/990pro/Documents/SoC/.repo_cache/TurboQuant`
- 분석 기준 commit: `7ac9b8d165a3f7d5e6df33b0450bc1f88ec0d4d5`
- 라이선스: `GPL-3.0`

핵심 결론부터 쓰면, TurboQuant는 **full-attention KV cache 압축**에는 흥미로운 설계이지만, 현재 repo의 구현은 **CUDA + PyTorch + vLLM monkey patch**를 전제로 한다. 따라서 `qwen3_5-mlx`에 그대로 이식하는 대상이라기보다, **아이디어와 데이터 구조를 선별적으로 재설계해서 가져오는 대상**에 가깝다.

---

## 1. TurboQuant가 하려는 일

TurboQuant의 목표는 long-context LLM inference에서 KV cache가 차지하는 VRAM을 줄이면서, decode 품질과 속도 저하를 작게 유지하는 것이다.

repo의 주장 기준으로는 다음 두 가지가 핵심이다.

1. full-attention layer의 key/value를 저비트로 압축해서 KV 메모리를 크게 줄인다.
2. decode 시점에는 최근 토큰은 exact buffer로 유지하고, 더 오래된 history는 compressed representation으로 읽는 hybrid attention을 사용한다.

repo README 기준 구조적으로 중요한 전제는 다음과 같다.

- dense transformer에서는 savings가 크다.
- Qwen3.5처럼 **full-attention + linear-attention** 혼합 구조에서는 full-attention layer만 압축 가능하다.
- 따라서 전체 KV 절감률은 모델 구조에 강하게 의존한다.

이 점은 `qwen3_5-mlx`에 특히 중요하다. 현재 우리 모델도 linear-attention(GatedDeltaNet) 계층이 섞여 있기 때문에, TurboQuant를 붙이더라도 전체 상태 메모리 중 **full-attention KV 부분만** 절감된다.

---

## 2. 방법론 핵심 요약

TurboQuant의 key 쪽 핵심은 단순한 uniform quantization이 아니다. 아래 네 단계가 결합되어 있다.

1. **random orthogonal rotation**
2. **Lloyd-Max scalar quantization**
3. **QJL(sign sketch) 기반 residual correction**
4. **unbiased inner-product estimator**

value 쪽은 상대적으로 단순하다.

1. group-wise asymmetric quantization
2. 2-bit 또는 4-bit packing

즉 key는 “dot product 보존”을 더 강하게 의식한 구조이고, value는 “메모리 절감” 중심이다.

### 2.1 Key 압축 원리

TurboQuant는 key 벡터 $x \in \mathbb{R}^d$를 바로 양자화하지 않는다.

먼저 $x$를 unit sphere로 정규화한 뒤 random orthogonal matrix $\Pi$로 회전한다.

$$
y = \Pi \left( \frac{x}{\|x\|} \right)
$$

repo의 설명대로, 이렇게 회전된 좌표의 각 성분은 고차원에서 Beta-like distribution을 따른다. 그래서 각 좌표를 동일한 uniform bin으로 자르는 것보다, 그 분포에 맞춘 **Lloyd-Max codebook**을 쓰는 것이 왜곡(MSE) 측면에서 유리하다.

이 부분 구현은 다음 파일에 있다.

- `turboquant/rotation.py`
- `turboquant/codebook.py`
- `turboquant/quantizer.py`

### 2.2 MSE quantizer

`TurboQuantMSE`는 rotated coordinate를 per-dimension scalar quantization해서 centroid index를 저장한다.

구현 흐름:

1. norm 저장
2. unit normalization
3. `rotate_forward(x, Pi)`
4. `searchsorted`로 centroid bucket index 선택
5. bit-pack하여 저장

복원 시에는:

1. bit unpack
2. centroid lookup
3. inverse rotation
4. original norm 재적용

이 구현은 `turboquant/quantizer.py`의 `TurboQuantMSE`에 있다.

### 2.3 QJL residual correction

MSE quantization만 쓰면 dot product 왜곡이 남기 때문에, TurboQuant는 residual을 한 번 더 다룬다.

$$
r = x - \tilde{x}_{mse}
$$

그리고 residual에 대해 Gaussian projection matrix $S$를 적용하고 sign bit만 저장한다.

$$
\text{sign}(Sr)
$$

이 sign sketch와 residual norm을 함께 써서 attention score를 unbiased하게 보정한다. repo는 이를 `TurboQuantProd`로 구현한다.

핵심 아이디어는 다음과 같다.

- coarse MSE approximation으로 대부분의 signal을 담고
- sign sketch가 residual contribution을 보정하며
- 최종 inner product estimator가 unbiased하도록 scale을 맞춘다.

이 구현은 `turboquant/quantizer.py`의 `TurboQuantProd`에 있다.

### 2.4 Value quantization

value는 key보다 단순하다. `turboquant/kv_cache.py`에서 group-wise min/max 기반 asymmetric quantization을 수행한다.

구현 요약:

1. head_dim을 group_size로 나눈다.
2. group별 `min/max`를 구한다.
3. scale/zero를 만든다.
4. 2-bit 또는 4-bit로 pack한다.

repo README와 코드 코멘트 모두 value 쪽이 품질 병목이라고 인정한다. 2-bit value는 cos similarity가 눈에 띄게 낮아지고, 4-bit가 quality-sensitive path에 더 적합하다고 정리한다.

---

## 3. 구현 데이터 플로우

TurboQuant의 실제 시스템은 “prefill 중 capture → compressed store 적재 → decode에서 hybrid attention” 구조다.

### 3.1 Write path

write path는 `turboquant/capture.py`와 `turboquant/store.py`가 담당한다.

흐름은 다음과 같다.

1. prefill token이 들어오면 KV를 bulk capture한다.
2. 오래된 token은 compressed store로 보낸다.
3. 최근 token은 ring buffer에 exact(bf16/fp16)로 유지한다.
4. decode hot path에서는 가능한 per-token quantization을 피하고, ring이 넘칠 때만 chunk 단위로 flush한다.

이 설계는 중요한 의미가 있다. TurboQuant는 **decode 한 토큰마다 바로 quantize하는 구조가 아니라**, 최근 토큰은 exact로 두고 오래된 token만 chunk 단위로 압축한다. 즉 decode 지연을 줄이기 위한 memory hierarchy가 들어가 있다.

### 3.2 Read path

read path는 `turboquant/score.py`가 담당한다.

decode 시 query가 들어오면 attention은 두 세그먼트로 나뉜다.

1. compressed history
2. recent exact buffer

compressed history는 quantized key/value를 dequantize하거나 score를 직접 계산해서 logits를 만들고, recent buffer는 exact matmul로 계산한다. 최종 attention output은 둘을 합친다.

repo의 현재 구현은 README가 암시하듯 “완전 fused compressed decode”보다는 **hybrid + dequantize fallback** 쪽이 더 현실적으로 쓰이고 있다.

---

## 4. 주요 파일별 역할

### 4.1 `turboquant/rotation.py`

- QR decomposition으로 random orthogonal matrix 생성
- QJL용 Gaussian projection matrix 생성
- rotate forward/backward 유틸 제공

포인트:

- head_dim 64~256 정도에서는 full $d \times d$ rotation matrix를 그냥 들고 가도 크지 않다는 판단이다.
- layer별 seed를 바꿔 matrix를 생성한다.

### 4.2 `turboquant/codebook.py`

- rotated coordinate 분포를 Beta PDF로 모델링
- 그 분포에 대해 continuous Lloyd-Max codebook 계산
- `codebooks/*.json`으로 캐시

포인트:

- 이 repo는 runtime uniform quantizer가 아니라, **오프라인/초기화 시점 codebook 생성 또는 로드**를 전제로 한다.
- MLX 쪽으로 갈 때도 이 방식은 그대로 쓸 수 있다.

### 4.3 `turboquant/quantizer.py`

- `TurboQuantMSE`
- `TurboQuantProd`
- bit packing/unpacking
- attention score estimator

포인트:

- 이 파일이 repo의 핵심 알고리즘 구현이다.
- `TurboQuantProd.attention_score()`는 실질적으로 query와 quantized key 사이 dot product를 복원하는 reference path다.

### 4.4 `turboquant/kv_cache.py`

- value quantization
- unpack/dequantize
- standalone `TurboQuantKVCache` abstraction

포인트:

- 실제 vLLM integration은 이 standalone class를 직접 쓰기보다 `capture/store/score` 기반 시스템을 더 많이 쓴다.
- 하지만 MLX port 관점에서는 이 standalone cache abstraction이 오히려 더 직접적인 참고 대상일 수 있다.

### 4.5 `turboquant/capture.py`

- `RingBuffer`
- `KVCaptureEngine`

포인트:

- ring buffer는 최근 token exact KV를 들고 간다.
- decode token은 ring에 append만 하고, overflow 시에만 compressed store로 flush한다.
- hot path에 quantization을 최대한 넣지 않으려는 설계다.

### 4.6 `turboquant/store.py`

- `CompressedKVStore`
- chunked append
- lazy flatten

포인트:

- append는 chunk 단위
- 여러 chunk는 list로 들고 있다가 read가 필요할 때 flat view를 materialize한다.
- write 시 `_flat`을 invalidate하고, read 때만 concatenate한다.

이건 메모리와 append overhead를 줄이는 단순하지만 실용적인 설계다.

### 4.7 `turboquant/score.py`

- compressed-only attention
- exact-only attention
- hybrid attention

포인트:

- current PyTorch path는 결국 compressed history를 dequantize해서 attention을 계산한다.
- 즉 저장 메모리는 줄지만, compute 측면에서는 “compressed domain에서 완전히 끝내는” path가 아직 약하다.
- repo README의 “hybrid decode dequantizes all history” 제한과 일치한다.

### 4.8 `turboquant/triton_kernels.py`

- fused MSE score kernel
- fused QJL score kernel
- fused decode attention kernel

포인트:

- 이 파일은 repo의 가장 공격적인 최적화 아이디어를 담고 있지만, 실제 hybrid path의 기본 구현은 아직 PyTorch fallback 성격이 더 강하다.
- 즉 kernel이 존재한다고 해서 곧바로 end-to-end path가 완전히 fused 되어 있는 것은 아니다.

### 4.9 `turboquant/integration/vllm.py`

- layer별 config/state 생성
- `do_kv_cache_update()` patch
- `forward()` patch
- `free_kv_cache()`
- `get_stats()`

포인트:

- 이 repo는 vLLM 내부 attention backend를 광범위하게 다시 쓰는 것이 아니라, **patch surface를 얇게 유지하고 capture/store/score로 위임**하려고 한다.
- integration 방식은 깔끔하지만, 여전히 **vLLM 내부 구조와 CUDA backend 전제**를 강하게 가진다.

---

## 5. vLLM 통합 방식 분석

TurboQuant는 vLLM에 네 가지 mode를 둔다.

- `off`
- `capture_only`
- `hybrid`
- `full_tq` (future)

실제로 의미가 있는 것은 `capture_only`와 `hybrid`다.

### 5.1 `capture_only`

- KV는 capture해서 compressed store에 저장
- attention compute는 기존 flash path 사용

이 모드는 baseline 대비 압축 capture만 넣었을 때의 영향이나 메모리 절감 효과를 보기 좋다.

### 5.2 `hybrid`

- decode에서 compressed history + recent exact buffer를 같이 사용
- compressed history가 충분히 길 때만 TQ path 사용
- 아니면 flash 또는 exact path fallback

### 5.3 `no_alloc`

repo는 “zero-allocation”에 가까운 방향도 탐색하지만, README와 코드가 함께 말하는 현실은 다음과 같다.

- prefill cache allocation을 완전히 없앤 건 아님
- paged cache를 init 시점에 여전히 쓰는 구간이 있음
- 진짜 zero-allocation path는 deeper vLLM integration이 필요

즉 현 구현은 메모리 절감 실험과 hybrid decode 실험에는 유용하지만, 완전히 정제된 production backend라기보다 연구 프로토타입에 더 가깝다.

---

## 6. repo가 스스로 인정하는 한계

README와 코드에서 읽히는 실제 한계는 다음과 같다.

1. **full-attention layer만 압축 가능**
2. **value quantization이 품질 병목**
3. **hybrid decode는 여전히 history dequantize cost가 큼**
4. **Triton fused kernel path가 repo의 기본 hot path에 완전히 녹아든 상태는 아님**
5. **dense transformer에서의 이득과 hybrid/MoE 구조에서의 이득은 다름**

이 중 `qwen3_5-mlx` 관점에서 가장 중요한 건 1번과 3번이다.

- 1번: 우리는 linear-attention 계층이 섞여 있다.
- 3번: MLX/Metal에서는 Triton이 없기 때문에 dequantize-heavy path는 더 비싸질 가능성이 높다.

---

## 7. qwen3_5-mlx에 그대로 가져오면 안 되는 이유

TurboQuant를 `qwen3_5-mlx`에 바로 “합친다”는 표현은 구현적으로는 맞지 않는다. 이유는 아래와 같다.

### 7.1 backend 전제가 다르다

TurboQuant current repo는:

- PyTorch tensor
- CUDA device
- Triton kernel
- vLLM attention backend

를 전제로 한다.

반면 `qwen3_5-mlx`는:

- MLX array
- Metal backend
- Apple Silicon memory model
- custom C++ / MLX graph

를 전제로 한다.

즉 가져올 수 있는 것은 **알고리즘, 데이터 구조, capture 정책**이지, 구현을 그대로 복사하는 것은 아니다.

### 7.2 linear-attention layer에는 직접 적용 불가

TurboQuant는 standard full-attention의 KV cache를 줄이기 위한 것이다. 그러나 `qwen3_5-mlx`의 linear-attention 쪽은 KV cache가 아니라:

- conv_state
- recurrent_state

같은 다른 상태를 가진다. 따라서 이 계층에는 TurboQuant의 key/value quantization 논리가 직접 들어가지 않는다.

### 7.3 macOS/Metal에는 Triton이 없다

repo의 long-context decode 실용성을 높이려면 compressed domain scoring을 fused kernel로 처리해야 한다. 그런데 macOS/Metal에는 Triton path가 없다.

즉 MLX port에서 naive path는 아래 둘 중 하나가 된다.

1. dequantize 후 일반 attention
2. MLX/Metal custom kernel을 새로 작성

1번은 메모리는 절약해도 compute가 악화될 수 있고, 2번은 구현 난도가 높다.

### 7.4 라이선스 리스크

repo는 `GPL-3.0`이다. 따라서 소스 코드를 직접 섞어 쓰거나 파생 구현을 배포할 계획이 있다면 라이선스 검토가 필요하다.

실무적으로는 다음처럼 접근하는 편이 안전하다.

- 아이디어와 구조를 참고
- 수학/알고리즘을 독자 구현
- 코드 직접 복사 최소화 또는 회피

---

## 8. qwen3_5-mlx에 적용할 수 있는 부분

그대로 이식은 어렵지만, 아래 조각들은 충분히 재사용 가치가 있다.

### 8.1 full-attention KV만 선택적으로 압축하는 정책

현재 `qwen3_5-mlx`는 full-attention layer와 linear-attention layer가 섞여 있다. 따라서 가장 현실적인 설계는:

1. full-attention KV만 압축 대상으로 삼고
2. linear-attention state는 기존 방식 유지
3. long context에서만 압축 활성화

하는 것이다.

이 정책은 TurboQuant와 모델 구조가 가장 잘 만나는 지점이다.

### 8.2 exact recent buffer + compressed history 구조

ring buffer 아이디어는 MLX 쪽에서도 유용하다.

- 최근 64~256 token은 exact KV
- 더 오래된 history는 compressed form

이렇게 나누면 short-context quality와 decode path 안정성을 유지하기 쉽다.

### 8.3 offline codebook + per-layer rotation matrix

rotation matrix와 Lloyd-Max codebook은 runtime hot path에 부담을 거의 주지 않는 초기화 비용이다. 따라서 MLX/C++ 포팅에서도 비교적 쉽게 가져갈 수 있다.

필요한 것은 다음 정도다.

1. head_dim별 codebook 생성
2. layer별 rotation/QJL matrix 생성
3. startup 시 로드

### 8.4 chunked compressed store

`CompressedKVStore`의 chunk append + lazy flatten 아이디어도 MLX/C++에서 그대로 재설계하기 좋다.

이 방식은:

- append path를 단순하게 유지하고
- read path에서만 materialization cost를 낸다.

현재 `qwen3_5-mlx` full-attention cache 구조를 확장할 때 참고할 만하다.

---

## 9. qwen3_5-mlx에 적용하기 어려운 부분

### 9.1 current hybrid decode scoring path

repo의 `score.py`는 compressed history를 결국 dequantize해서 계산하는 부분이 있다. MLX에서 이걸 그대로 따라가면 메모리는 아끼더라도 decode latency가 나빠질 수 있다.

즉 우리 쪽에서 정말 의미 있는 구현은 아래 둘 중 하나여야 한다.

1. long-context memory capacity 확장 전용 path
2. Metal fused score kernel까지 포함하는 path

### 9.2 vLLM monkey patch 방식

`qwen3_5-mlx`는 vLLM runner가 아니므로 `integration/vllm.py`는 구조 참고용이지 구현 재사용용이 아니다.

실제로 가져와야 할 것은 monkey patch 코드가 아니라 다음 개념이다.

- prefill에서 bulk capture
- decode에서 cheap append
- layer별 state object
- exact recent + compressed history split

### 9.3 fused Triton kernels

repo의 `triton_kernels.py`는 MLX/Metal에 직접 대응되지 않는다. 이 부분은 완전히 별도의 Metal kernel 설계가 필요하다.

---

## 10. qwen3_5-mlx에 대한 현실적 통합 전략

TurboQuant를 `qwen3_5-mlx`에 붙이려면 아래 순서가 현실적이다.

### 10.1 Phase 1: 연구용 reference path

목표:

- full-attention layer에서만 KV 압축 실험
- decode correctness / quality / memory 절감 확인
- 속도는 당장 희생 가능

방법:

1. prefill 완료 후 full-attention KV를 압축
2. recent exact buffer 유지
3. decode에서 compressed history는 CPU/MLX reference path로 복원 후 attention score 계산
4. baseline과 token/logit 비교

이 단계에서는 “속도 개선”보다 “정확성 + memory tradeoff 확인”이 목적이다.

### 10.2 Phase 2: hybrid long-context mode

목표:

- 긴 문맥에서만 TQ-like 압축 path 활성화
- 짧은 문맥에서는 기존 exact KV 유지

방법:

- threshold 기반 activation
- ring buffer 크기 sweep
- value bits 2/4 비교

이 단계는 실전성과 실험 편의성의 균형이 좋다.

### 10.3 Phase 3: Metal kernel 최적화

목표:

- compressed-domain score 계산의 compute overhead 제거

필요한 것:

1. MSE score fused kernel
2. QJL score fused kernel
3. 가능하면 attention merge kernel

이 단계에 들어가기 전까지는 TurboQuant류 기법이 memory-saving feature로는 유의미해도, speed feature로는 오히려 손해일 수 있다.

---

## 11. qwen3_5-mlx 관점의 최종 판단

TurboQuant repo는 “지금 당장 복붙해서 붙일 코드”는 아니다. 하지만 연구 관점에서는 꽤 가치가 있다.

가치가 있는 이유:

1. full-attention KV만 선택 압축하는 전략이 우리 모델 구조와 잘 맞는다.
2. random rotation + Lloyd-Max + residual sign correction 조합은 key dot-product 보존 관점에서 설득력이 있다.
3. recent exact buffer + compressed history 구조는 decode 품질을 지키기 좋은 현실적 설계다.

주의할 점:

1. repo 구현은 CUDA/vLLM/Triton 중심이다.
2. 현재 기본 path는 dequantize cost를 여전히 많이 진다.
3. linear-attention 계층에는 직접 적용되지 않는다.
4. GPL-3.0 라이선스 주의가 필요하다.

따라서 `qwen3_5-mlx`에 대한 가장 현실적인 결론은 이렇다.

- **TurboQuant의 수학과 캐시 설계는 참고할 가치가 높다.**
- **하지만 실제 통합은 MLX/Metal 기준으로 다시 설계해야 한다.**
- **특히 long-context memory extension 기능으로 먼저 검증하고, 속도 개선은 그 다음 단계로 보는 것이 맞다.**

---

## 12. 바로 참고할 파일 목록

TurboQuant에서 우선적으로 다시 볼 파일은 아래다.

- `/Volumes/990pro/Documents/SoC/.repo_cache/TurboQuant/README.md`
- `/Volumes/990pro/Documents/SoC/.repo_cache/TurboQuant/turboquant/quantizer.py`
- `/Volumes/990pro/Documents/SoC/.repo_cache/TurboQuant/turboquant/codebook.py`
- `/Volumes/990pro/Documents/SoC/.repo_cache/TurboQuant/turboquant/rotation.py`
- `/Volumes/990pro/Documents/SoC/.repo_cache/TurboQuant/turboquant/kv_cache.py`
- `/Volumes/990pro/Documents/SoC/.repo_cache/TurboQuant/turboquant/capture.py`
- `/Volumes/990pro/Documents/SoC/.repo_cache/TurboQuant/turboquant/store.py`
- `/Volumes/990pro/Documents/SoC/.repo_cache/TurboQuant/turboquant/score.py`
- `/Volumes/990pro/Documents/SoC/.repo_cache/TurboQuant/turboquant/triton_kernels.py`
- `/Volumes/990pro/Documents/SoC/.repo_cache/TurboQuant/turboquant/integration/vllm.py`

필요하다면 다음 단계 문서는 아래 주제로 이어질 수 있다.

1. `TurboQuant -> MLX 설계안`
2. `full-attention KV 압축 실험 계획`
3. `Metal kernel 후보 연산 분해`