# Phase 0: 의존성 조사

## 목표
구현에 필요한 모든 외부 의존성을 조사하고 `docs/`에 정리한다.
이후 Phase에서 코드를 짤 때 추가 조사 없이 바로 구현에 들어갈 수 있는 상태를 만든다.

## 0a. modeling_qwen3_5.py 의존성 조사

`modeling_qwen3_5.py`의 import 전체와 내부에서 사용하는 transformers 유틸을 한 번에 조사한다.

### 조사할 import 목록

**transformers 내부 유틸** (GitHub에서 fetch 필요):
- `cache_utils` → `Cache`, `DynamicCache`
- `masking_utils` → `create_causal_mask`
- `modeling_rope_utils` → `ROPE_INIT_FUNCTIONS`, `dynamic_rope_update`
- `modeling_utils` → `ALL_ATTENTION_FUNCTIONS`, `PreTrainedModel`
- `modeling_flash_attention_utils` → `FlashAttentionKwargs`
- `modeling_layers` → `GradientCheckpointingLayer`
- `modeling_outputs` → `BaseModelOutputWithPast`, `CausalLMOutputWithPast`, ...
- `activations` → `ACT2FN`
- `generation` → `GenerationMixin`
- `configuration_utils` → `PreTrainedConfig`
- `tokenization_utils_tokenizers` → `TokenizersBackend`

**외부 라이브러리** (인터페이스만 파악):
- `tokenizers` (HuggingFace) → `BPE`, `Regex`, `pre_tokenizers`, `decoders`
- `causal_conv1d` → `causal_conv1d_fn`, `causal_conv1d_update`
- `flash_linear_attention (fla)` → `FusedRMSNormGated`, `chunk_gated_delta_rule`, `fused_recurrent_gated_delta_rule`

**torch 연산** (조사 불필요, 직접 구현):
- `nn.Linear`, `nn.Embedding`, `nn.Conv1d`, `nn.Conv3d`, `nn.LayerNorm`, `nn.Parameter`
- `F.softmax`, `F.silu`, `F.pad`, `F.conv1d`, `F.softplus`
- `torch.matmul`, `torch.rsqrt`, `torch.cat`, `torch.stack`, `torch.split`
- `torch.arange`, `torch.outer`, `torch.triu`, `torch.cumsum`

### 결과물
→ `docs/transformers/` 에 모듈별 정리

### 상태
- [x] cache_utils 조사 → `docs/transformers/cache_utils.md`
- [x] masking_utils 조사 → `docs/transformers/masking_utils.md`
- [x] modeling_rope_utils 조사 → `docs/transformers/rope_utils.md`
- [x] activations (ACT2FN) 조사 → `docs/transformers/activations.md`
- [x] tokenization_utils_tokenizers 조사 → (skip, C++ 추론에서 tokenizer.json 직접 사용)
- [x] GatedDeltaNet fallback 구현 확인 → `docs/transformers/gated_delta_net.md`
- [x] 기타 유틸 정리 → `docs/transformers/misc_utils.md`

## 0b. PyTorch MPS backend 조사

PyTorch가 `device="mps"`일 때 각 연산을 어떻게 Metal에 디스패치하는지 조사한다.

### 조사 대상
PyTorch 소스: `aten/src/ATen/native/mps/`

| 연산 | PyTorch MPS 구현 방식 | 상태 |
|------|----------------------|------|
| Matrix multiply (Linear) | **Dual**: Custom Metal (`matmul_` TILE=16) edge case + MPSGraph `matrixMultiplication` 기본 | ✅ |
| Softmax | MPSGraph `softMaxWithTensor:axis:` | ✅ |
| SiLU / GELU | MPSGraph 분해 (SiLU=`x/(1+exp(-x))`, GELU=erf 또는 tanh variant) | ✅ |
| RMSNorm (rsqrt, mul) | 전용 커널 없음. LayerNorm은 Custom Metal shader. RMSNorm은 elementwise 분해 | ✅ |
| RoPE (cos/sin + apply) | 전용 커널 없음. cos/sin MPSGraph + elementwise BinaryOps 조합 | ✅ |
| Conv1d | MPSGraph `MPSGraphConvolution2DOpDescriptor` (Conv2D 경유) | ✅ |
| Elementwise (add, mul) | MPSGraph (`additionWith...`, `multiplicationWith...`) | ✅ |
| Embedding lookup | MPSGraph `gatherWithUpdatesTensor:indicesTensor:axis:` (index_select 경유) | ✅ |
| Argmax / sampling | MPSGraph `reductionArgMaximumWithTensor:axis:` | ✅ |
| BPE tokenize/detokenize | CPU (GPU 불필요) | ✅ |

### 각 연산에 대해 파악할 것
- `.metal` shader 직접 작성인지 / `MPSGraph` 사용인지 / CPU fallback인지
- 입력 텐서 레이아웃 (NCHW vs NHWC 등)
- quantized 연산 지원 여부

### 결과물
→ `docs/torch/mps_ops.md` 에 정리
