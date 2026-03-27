# Research Questions And Answers

이 문서는 기존 plan을 읽으며 실제 구현 전에 반드시 답을 고정해야 하는 질문을 정리하고, GitHub 및 공식 문서를 조사해 답을 기록한 것이다.

## Q1. Tensor를 그냥 raw pointer 기반으로 갈지, `Storage + Tensor metadata`로 갈지?

질문:

1. backendless pure C++ 목표라면 처음부터 `Tensor` 추상화가 오히려 과한가?
2. 반대로 training 확장을 생각하면 raw pointer만으로는 너무 빨리 막히지 않는가?

조사:

1. PyTorch internals는 tensor를 “data + metadata(size, dtype, stride, storage offset)”로 설명하고, view를 위해 storage와 tensor를 분리한다.
2. ATen/C10 구현은 `Storage`와 `TensorImpl`을 분리하고, `as_strided` 기반 view를 유지한다.
3. `llm.c`는 pure C reference에서 contiguous buffer 위에 논리 shape를 얹는 방식으로 단순성과 성능을 유지한다.

답:

1. Phase 1은 contiguous-first가 맞다.
2. 하지만 인터페이스는 반드시 `Storage + Tensor`로 분리해야 한다.
3. 즉, 구현은 단순하게 시작하되 ABI는 training-ready로 설계한다.

설계 반영:

1. `Storage`는 owning / mapped / external 세 종류를 둔다.
2. `Tensor`는 `dtype`, `shape`, `stride`, `storage`, `offset_bytes`, `contiguous`를 가진다.
3. 실제 kernel은 contiguous 경로만 우선 지원하되, stride 메타데이터는 구조체에 남긴다.

## Q2. KV cache의 baseline layout은 무엇으로 고정할 것인가?

질문:

1. baseline에서도 packed key cache를 도입해야 하는가?
2. 아니면 디버깅 가능한 단순 layout으로 먼저 가야 하는가?

조사:

1. FasterTransformer는 decoder 최적화를 위해 key cache를 `[B, H, Dh/x, L, x]`처럼 packed 형태로 두고 value cache는 `[B, H, L, Dh]`로 둔다.
2. ggml/llama.cpp 계열 예시는 구현 단순성을 위해 view와 contiguous memory를 적극 사용한다.
3. 현재 우리 `Mac/cpu/header/attention.h`의 `KVCache`는 직관적인 layer-major append 모델과 더 잘 맞는다.

답:

1. CPU baseline은 packed KV cache로 시작하지 않는다.
2. baseline canonical layout은 `[layer][batch][kv_head][seq][head_dim]`로 고정한다.
3. 최적화 단계에서만 `PackedKeyCache` 같은 별도 내부 표현을 추가한다.

설계 반영:

1. public runtime 계약은 canonical KV layout만 노출한다.
2. kernel 내부 최적화는 추후 private layout 변환으로 처리한다.
3. decode fast path는 append-only contiguous tail write를 우선 최적화한다.

## Q3. Qwen3 tokenizer를 C++에서 직접 완전 재현해야 하는가?

질문:

1. `tokenizer.json` 전체 스펙을 C++로 읽는 것이 현실적인가?
2. chat template와 thinking mode까지 감안하면 exporter가 더 많이 해줘야 하는가?

조사:

1. `tokenizer_config.json`에는 많은 added token과 복잡한 Jinja chat template가 포함된다.
2. Qwen 공식 문서는 thinking / non-thinking mode를 chat template 수준에서 제어한다.
3. Qwen3-0.6B tokenizer에는 `<|im_start|>`, `<|im_end|>`, `<think>`, `</think>` 같은 중요한 control token이 있다.

확인된 핵심 토큰:

1. `bos_token_id = 151643` with content `<|endoftext|>`
2. `eos_token_id = 151645` with content `<|im_end|>`
3. `<think> = 151667`
4. `</think> = 151668`

답:

1. Phase 1에서 HF fast tokenizer 전체 재구현은 범위를 벗어난다.
2. exporter가 runtime-friendly tokenizer format과 최소 chat-template metadata를 추가로 생성해야 한다.
3. C++ runtime은 “Qwen3 전용 최소 기능 tokenizer + template builder”부터 구현한다.

설계 반영:

1. `LLM_interpreter`가 `tokenizer_runtime.json`을 추가 생성한다.
2. C++는 generic Jinja engine 대신 Qwen3 전용 template builder를 구현한다.
3. thinking / non-thinking은 template preset 두 개로 나눈다.

## Q4. Qwen3-0.6B에서 실제로 고정해야 하는 모델 사실은 무엇인가?

조사로 확인된 내용:

1. `model_type = qwen3`
2. `hidden_size = 1024`
3. `intermediate_size = 3072`
4. `num_hidden_layers = 28`
5. `num_attention_heads = 16`
6. `num_key_value_heads = 8`
7. `head_dim = 128`
8. `rms_norm_eps = 1e-6`
9. `rope_theta = 1000000`
10. `torch_dtype = bfloat16`
11. `tie_word_embeddings = true`
12. `vocab_size = 151936`
13. pretrained context length baseline은 32768 token

답:

1. `Qwen3-0.6B`는 GQA 모델이다. `n_heads != n_kv_heads`를 반드시 반영해야 한다.
2. `lm_head`는 token embedding과 tied contract로 설계해야 한다.
3. Phase 1 기준 RoPE는 `rope_theta=1000000`을 그대로 사용한다.

설계 반영:

1. `Qwen3Config`는 위 필드를 강제 검증한다.
2. attention 모듈은 Q와 KV head 수를 분리한 구현이어야 한다.
3. `lm_head`는 독립 weight가 아니라 tied view를 허용해야 한다.

## Q5. 긴 컨텍스트 대응은 Phase 1에 포함할 것인가?

조사:

1. Qwen 공식 문서는 pretraining context를 32768로 설명한다.
2. YaRN으로 131072까지 확장 가능하다고 하지만, static YaRN은 짧은 문맥에서 성능 저하 가능성이 있다고 명시한다.
3. Transformers와 llama.cpp 모두 long context를 별도 설정으로 취급한다.

답:

1. Phase 1 baseline은 original context 32768에 맞춘다.
2. YaRN은 Phase 1 completion 조건이 아니라 Phase 1.5 또는 Phase 2 항목이다.
3. long-context 옵션은 runtime option으로 두고 기본값은 off다.

설계 반영:

1. `rope_scaling`은 config parser에 남기되 기본 비활성화.
2. validation test도 우선 32k 이내에서 만든다.

## Q6. QKV projection은 separate weight로 유지할지, fused QKV로 export할지?

조사:

1. Hugging Face checkpoint는 보통 `q_proj`, `k_proj`, `v_proj`로 separate tensor를 가진다.
2. FasterTransformer 계열 converter는 inference 최적화를 위해 fused QKV weight를 별도 저장하기도 한다.
3. 우리 exporter는 현재 HF tensor를 직접 `.bin`으로 내보낸다.

답:

1. export contract baseline은 separate Q/K/V를 유지한다.
2. fused QKV는 load-time optimization 또는 optional secondary artifact로 처리한다.
3. 원본 계약을 깨지 않는 것이 디버깅과 parity에 유리하다.

설계 반영:

1. `Qwen3WeightMapping`은 HF tensor name을 그대로 유지한다.
2. `PackedLinearWeight`와 `FusedQKVWeight`는 loader 이후 opt cache로 취급한다.

## Q7. batch dimension은 Phase 1에서 어디까지 지원할 것인가?

조사:

1. `llm.c` 문서는 local inference reference에서 batch=1 decode를 단순화 선택으로 둔다.
2. 하지만 training과 serving 확장을 고려하면 tensor/module/runtime 계층은 batch-aware가 되어야 한다.
3. 현재 우리 prototype도 `batched_matrix`와 batched attention을 이미 일부 가지고 있다.

답:

1. Tensor/ops/module은 batch-aware로 설계한다.
2. generation runtime fast path는 batch=1 decode를 1순위로 최적화한다.
3. batch decode는 기능적으로 지원하되 Phase 1 성능 최적화 타겟은 아니다.

설계 반영:

1. 모든 public shape 표기에서 batch 차원을 생략하지 않는다.
2. runtime benchmark는 batch=1 과 batch>1을 분리한다.

## Q8. CPU 최적화 우선순위는 무엇인가?

조사:

1. `llm.c`는 먼저 단순 CPU loop + OpenMP로 reference를 만들고, 그 뒤 specialization을 쌓는다.
2. ggml은 allocator, packed layout, quantized row layout, backend buffer abstraction을 분리해 최적화를 쌓는다.
3. FasterTransformer는 small-op fusion, KV cache reuse, fused QKV, custom attention kernel을 강조한다.

답:

1. 우리도 최적화 순서를 지켜야 한다.
2. 우선순위는 `correctness -> memory layout -> prepack -> threading -> quantization -> fusion`이다.
3. fused attention 같은 고난도 커널은 baseline correctness 이후다.

설계 반영:

1. Phase 1 완료 기준에 fused kernel은 넣지 않는다.
2. 대신 prepack, fp16/bf16 weight, fp32 accumulate, arena allocator를 필수로 둔다.

## Q9. training-ready는 어디까지 미리 설계해야 하는가?

조사:

1. PyTorch internals는 autograd를 tensor 메타와 분리된 추가 메타로 취급한다.
2. `llm.c`는 inference와 training의 메모리/중간 activation 유지 전략이 크게 다르다고 강조한다.

답:

1. Phase 1에서 autograd 엔진은 구현하지 않는다.
2. 하지만 `Tensor`에 optional grad slot, `Module` parameter registry, op trace hook 자리는 남겨야 한다.
3. inference session과 training session은 처음부터 분리한다.

설계 반영:

1. `train/` 디렉토리는 skeleton만 둔다.
2. inference fast path에 graph bookkeeping을 넣지 않는다.