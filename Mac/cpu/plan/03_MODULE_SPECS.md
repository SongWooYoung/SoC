# Module Specifications

## 1. Export Contract With LLM_interpreter

`LLM_interpreter`는 현재 아래를 만든다.

1. `manifest.json`
2. `weights/*.bin`
3. `tokenizer/*`

런타임은 이 계약을 기준으로 동작해야 한다.

### 1.1 ManifestLoader

역할:

1. `manifest.json` parse
2. `tensors[]`를 name index로 변환
3. `config`에서 model hyperparameter 추출
4. `generation_config`에서 sampling default 추출

필수 API:

```text
Manifest manifest = ManifestLoader::Load(path)
const TensorRecord& manifest.FindTensor(name)
Qwen3Config manifest.ToQwen3Config()
```

검증 항목:

1. dtype 문자열이 지원 범위 안에 있는지
2. shape product와 byte_size가 맞는지
3. duplicate tensor name이 없는지
4. `tie_word_embeddings`, `num_attention_heads`, `num_key_value_heads`, `head_dim` 같은 Qwen3 필수 필드가 존재하는지

### 1.2 WeightLoader

역할:

1. raw `.bin` 파일 read 또는 mmap
2. storage 생성
3. Tensor object 생성

정책:

1. immutable model weight는 기본 mmap 우선
2. mutable activation은 allocator 기반 heap/arena 사용
3. fp16/bf16 raw decode 지원

필수 API:

```text
Tensor WeightLoader::LoadTensor(const TensorRecord& record)
Tensor WeightLoader::LoadByName(const Manifest& manifest, const std::string& name)
```

## 2. Core Tensor Stack

### 2.1 DType

필수 dtype:

1. float32
2. float16
3. bfloat16
4. int32
5. int64
6. uint16 for raw bf16 handling

선택 dtype:

1. int8
2. uint8

### 2.2 Storage

필수 타입:

1. `OwnedStorage`
2. `MappedStorage`
3. `ExternalStorage`

필수 기능:

1. pointer
2. byte size
3. alignment
4. ownership/lifetime policy

### 2.3 Tensor

필수 필드:

1. dtype
2. shape
3. stride
4. storage handle
5. offset bytes
6. contiguous flag

Phase 1 필수 함수:

1. `numel()`
2. `nbytes()`
3. `is_contiguous()`
4. `reshape()`
5. `view()`
6. `data<T>()`

## 3. Kernel and Op Minimum Set

## 3.1 반드시 구현할 Kernel

1. `matmul_f32`
2. `matmul_f16_weight_f32_accum`
3. `matmul_packed_rhs`
4. `embedding_lookup`
5. `rmsnorm_f32_accum`
6. `silu`
7. `softmax_last_dim_f32_accum`
8. `rope_apply_inplace`
9. `argmax`
10. `topk/top-p helper`

## 3.2 반드시 구현할 Op

1. `add`
2. `mul`
3. `matmul`
4. `linear`
5. `embedding`
6. `rmsnorm`
7. `silu`
8. `softmax`
9. `rope`
10. `concat` or head merge helper

## 4. Runtime-Critical Modules

### 4.1 Embedding

입력:

1. token id tensor `[seq]` 또는 `[batch, seq]`

출력:

1. hidden states `[seq, hidden]` 또는 `[batch, seq, hidden]`

### 4.2 Linear

지원 경로:

1. fp32 dense
2. fp16 weight + fp32 accum
3. packed rhs
4. 이후 int8/quantized rhs

### 4.3 RMSNorm

Qwen/Llama 계열에서 필수다. 반드시 fp32 accumulate로 구현한다.

### 4.4 RoPE

필수 config:

1. `rope_theta`
2. `head_dim`
3. `position_offset`

### 4.5 Attention

반드시 지원할 개념:

1. Q heads와 KV heads 분리
2. GQA
3. prefill attention
4. decode attention with KV cache
5. output projection
6. `num_attention_heads = 16`, `num_key_value_heads = 8` 같은 GQA ratio 기반 head broadcast

### 4.6 KVCache

레이어별 관리가 필수다.

권장 layout:

```text
[layer][batch][kv_head][seq][head_dim]
```

이유:

1. layer 단위 접근이 자연스럽다.
2. decode 시 특정 layer만 update한다.
3. kv_head 분리와 GQA 접근이 명확하다.
4. CPU baseline에서 디버깅과 parity 확인이 쉽다.

주의:

1. FasterTransformer 같은 고성능 구현은 key cache를 packed layout으로 두기도 한다.
2. 하지만 그것은 internal optimization 문제이지 public ABI 문제로 올리지 않는다.

필수 함수:

1. `reserve(max_batch, max_seq)`
2. `write(layer, batch, position, K, V)`
3. `read(layer, batch, seq_range)`
4. `clear_sequence(batch)`

### 4.7 Sampler

Phase 1에서도 필요하다.

최소 기능:

1. greedy
2. temperature
3. top-k
4. top-p
5. repetition/presence penalty는 optional but planned

## 5. Tokenizer Plan

사용자 요구상 backendless C++ tokenizer가 필요하다. 선택지는 두 가지다.

1. `tokenizer.json`을 직접 파싱해서 C++에서 처리
2. exporter가 tokenizer를 더 단순한 custom format으로 재-export

권장 방향:

1. Phase 1: exporter에서 `tokenizer_export.json` 또는 `vocab + merges + special token table` 같은 단순 포맷을 추가 생성
2. C++ runtime은 이 단순 포맷을 읽는다.

이유:

1. HF fast tokenizer 전체 스펙을 C++에서 그대로 구현하는 것은 비용이 크다.
2. Qwen3 단일 타겟 모델이라면 exporter 쪽 단순화가 훨씬 현실적이다.

따라서 tokenizer 모듈 구현 계획은 다음과 같다.

1. `LLM_interpreter` 확장
   `tokenizer.json` -> runtime-friendly custom tokenizer export 추가
2. C++ tokenizer loader
3. encode/decode
4. special token handling
5. chat template 적용은 초기에는 Python export 단계에서 assistant/user prompt builder를 정의하거나, C++에 최소 템플릿만 고정 구현

Qwen3-0.6B에서 반드시 고정할 토큰/계약:

1. `eos_token_id = 151645` and content `<|im_end|>`
2. `<think> = 151667`
3. `</think> = 151668`
4. `<|im_start|>` / `<|im_end|>` 채팅 경계 토큰 사용

실행 결정:

1. generic Jinja interpreter는 Phase 1 범위에서 제외
2. `Qwen3TemplateBuilder`가 thinking / non-thinking preset 두 개를 제공

## 6. Qwen3 Weight Mapping

필수 weight class:

1. token embedding
2. final norm
3. lm head
4. per-layer input norm
5. q_proj
6. k_proj
7. v_proj
8. o_proj
9. post-attention norm
10. gate_proj
11. up_proj
12. down_proj

필수 작업:

1. HF tensor names를 그대로 받아오는 mapping table 생성
2. layer index parsing helper 작성
3. missing tensor detect 및 startup validation 수행
4. `tie_word_embeddings = true`일 때 embedding/lm_head alias 처리

## 7. Training-Ready Hooks

지금 당장 autograd를 구현하지 않더라도 아래 hook은 설계에 포함한다.

1. `TensorId` 또는 debug name
2. Module parameter registry
3. op trace option
4. gradient tensor storage slot reserve
5. checkpoint write path planned
