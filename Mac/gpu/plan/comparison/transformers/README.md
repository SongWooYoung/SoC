# transformers Comparison

## Repo Snapshot

- Repository: `huggingface/transformers`
- Local snapshot: `/tmp/soc_compare/transformers`
- Inspected commit: `bc57673`

## 왜 이 레포를 보는가

사용자가 비교 기준으로 실제 PyTorch MPS 성능을 원하고 있고, 현재 우리가 benchmark에 쓰는 PyTorch 경로도 결국 `transformers` 모델 구현을 탄다. 그래서 "Metal 구현이 어떻게 되어 있는가"를 보면, 답은 "우리처럼 직접 Metal kernel runtime을 짜는 구조는 아니다"에 가깝다. 대신:

- Qwen model 구현은 표준 PyTorch `nn.Linear`, `sdpa`, cache abstraction을 사용한다.
- Apple Silicon fast path는 `PyTorch MPS`, `scaled_dot_product_attention`, 그리고 별도 Metal quantization extension을 통해 얻는다.

## 직접 읽은 파일

- `src/transformers/models/qwen3/modeling_qwen3.py`
- `src/transformers/modeling_flash_attention_utils.py`
- `src/transformers/cache_utils.py`
- `src/transformers/quantizers/quantizer_metal.py`
- `src/transformers/integrations/metal_quantization.py`

## 구조 요약

### 1. 기본 Qwen 경로는 고수준 PyTorch다

[`modeling_qwen3.py`](/tmp/soc_compare/transformers/src/transformers/models/qwen3/modeling_qwen3.py) 기준:

- `Qwen3MLP`는 그냥 `down_proj(act(gate_proj(x)) * up_proj(x))`
- attention은 `q_proj/k_proj/v_proj/o_proj` + rotary + cache + attention function
- RoPE, RMSNorm도 PyTorch tensor 연산 위주

즉 기본 모델 구현엔 우리 같은 custom Metal kernel scheduler가 없다.

### 2. attention fast path는 `sdpa`/flash 계층이다

[`modeling_flash_attention_utils.py`](/tmp/soc_compare/transformers/src/transformers/modeling_flash_attention_utils.py) 기준:

- attention mask/padding/varlen 처리를 한 뒤 `flash_fn` 또는 `flash_varlen_fn`으로 넘긴다.
- MPS 경로에서 `cu_seq_lens_k = cu_seq_lens_k.clone()` 같은 호환 처리가 있다.
- 주석에 `metal-flash-sdpa` external kernel 링크가 직접 들어 있다.

핵심:

- `transformers` 자체가 Metal attention kernel을 직접 갖고 있는 게 아니라, PyTorch SDPA 또는 외부 hub kernel integration을 활용한다.

### 3. cache 계층은 매우 강하다

[`cache_utils.py`](/tmp/soc_compare/transformers/src/transformers/cache_utils.py) 기준:

- `DynamicCache`
- `StaticCache`
- `QuantizedCache`
- sliding / chunked / hybrid / offloading

이건 `mlx-lm`과 비슷하게 runtime policy 계층이 매우 두껍다는 뜻이다.

### 4. Metal low-bit path는 별도 quantization integration이다

[`quantizer_metal.py`](/tmp/soc_compare/transformers/src/transformers/quantizers/quantizer_metal.py),
[`metal_quantization.py`](/tmp/soc_compare/transformers/src/transformers/integrations/metal_quantization.py) 기준:

- `MetalLinear`가 `nn.Linear`를 대체한다.
- weight는 packed `uint32` + `scales` + `qbiases`
- forward는 external hub kernel의 `affine_qmm_t(...)`를 호출한다.
- 즉 `y = x @ dequant(weight).T`를 low-bit fused Metal kernel로 처리한다.

이건 매우 중요하다.

현재 PyTorch `transformers`가 Apple Silicon에서 더 빠른 길은:

- dense float matmul을 우리보다 더 잘 짠 custom runtime이 있어서가 아니라
- PyTorch MPS backend와 low-bit Metal kernels를 붙일 수 있기 때문이다.

## 우리 코드와의 직접 비교

### 우리 코드가 직접 하는 것

- [`qwen_attention.mm`](/Volumes/990pro/Documents/SoC/Mac/gpu/src/module/qwen_attention.mm)
- [`qwen_mlp.mm`](/Volumes/990pro/Documents/SoC/Mac/gpu/src/module/qwen_mlp.mm)
- [`matmul_op.mm`](/Volumes/990pro/Documents/SoC/Mac/gpu/src/op/matmul_op.mm)
- [`gpu_kernels.metal`](/Volumes/990pro/Documents/SoC/Mac/gpu/shaders/gpu_kernels.metal)

우리는 직접:

- scheduler
- command buffer 경계
- matmul kernel
- RMSNorm/Softmax/RoPE kernel
- KV blit

를 구현한다.

### transformers가 직접 안 하는 것

`transformers`는 위 항목을 직접 다루지 않는다. 대신:

- model graph는 PyTorch가 실행
- device backend는 MPS가 실행
- attention fast path는 SDPA/flash 계층
- low-bit linear는 external Metal quantization kernel

을 쓴다.

## 결론적으로 배울 것

### 1. PyTorch를 넘기려면 dense float path만으로는 어렵다

현재 우리 병목은 여전히 projection matmul이다. `transformers` 쪽에서 실제 Metal 전용 가속 포인트는 `MetalLinear + affine_qmm_t` 같은 low-bit fused qmm다. 이건 결국:

- weight bandwidth를 크게 줄이고
- dequant + matmul을 fused 하고
- Apple GPU에 맞는 kernel을 쓰는

방향이다.

즉 우리가 PyTorch MPS를 넘기고 싶다면, 다음 큰 축은 단순 threadgroup width 조정이 아니라:

- `float16` 완성
- 그 다음 `4-bit / 8-bit packed weight + fused qmm`

이다.

### 2. runtime policy 계층도 강화해야 한다

`transformers`는 cache abstraction이 두껍다. `mlx-lm`과 같은 결론이다. long context, static cache, quantized cache, sliding window를 다 policy로 푼다.

### 3. attention은 custom eager kernel보다 SDPA류 구조를 다시 봐야 한다

지금 우리 attention은 `score -> softmax -> value` eager 3-op 구조다. `transformers`는 SDPA / flash 계층을 쓰므로, 중장기적으로는 attention도 fused attention path를 다시 생각해야 한다.

## 우리 코드에 대한 바로 적용 가능한 아이디어

### 아이디어 1. low-bit Metal linear 준비

`transformers`의 `MetalLinear`처럼:

- packed weight format
- scale / bias tensor
- fused qmm kernel

을 별도 경로로 설계한다. 이게 PyTorch 수준 이상으로 가는 가장 현실적인 장기 방향이다.

### 아이디어 2. cache policy 계층 확장

`DynamicCache/StaticCache/QuantizedCache`처럼 우리도:

- current append cache
- future rotating cache
- future static/slab cache

를 분리해야 한다.

### 아이디어 3. attention fast path 재설계

현재 `AttentionScore + Softmax + AttentionValue` 3-op eager path를 유지하되, 중기적으로 SDPA-like fused path를 연구 대상으로 올린다.

## 결론

`transformers`를 보면, PyTorch MPS를 이기는 길은 "우리도 똑같이 고수준 Python을 쓴다"가 아니라, "weight bandwidth를 줄이는 Metal low-bit fused linear까지 간다"는 쪽이다. 현재의 decode vec4 matmul 최적화는 필요한 1단계지만, PyTorch를 넘기려면 결국 low-bit fused qmm가 다음 큰 축이다.
