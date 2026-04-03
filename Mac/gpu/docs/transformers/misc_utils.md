# 기타 유틸리티 분석 (C++ 구현 불필요)

## 요약
modeling_qwen3_5.py에서 import하는 나머지 유틸리티들은 대부분 
Python 학습 프레임워크 전용이거나 타입 힌트/문서화 용도이다.
C++/Metal 추론에서는 무시해도 된다.

## Import 목록 및 처분

| Import | 용도 | C++ 필요 | 이유 |
|--------|------|----------|------|
| `GenerationMixin` | `.generate()` 루프 | ❌ | C++에서 자체 generation loop 작성 |
| `PreTrainedModel` | weight loading, config 관리 | ❌ | C++에서 직접 safetensors 로드 |
| `GradientCheckpointingLayer` | training 전용 | ❌ | 추론 불필요 |
| `ALL_ATTENTION_FUNCTIONS` | attention dispatch registry | ❌ | C++에서 직접 호출 |
| `FlashAttentionKwargs` | TypedDict 힌트 | ❌ | 타입 힌트 |
| `LossKwargs` | TypedDict 힌트 | ❌ | training 전용 |
| `ModelOutput` | dataclass 래퍼 | ❌ | C++ struct 사용 |
| `BaseModelOutputWithPast` | return type | ❌ | C++ struct |
| `CausalLMOutputWithPast` | return type | ❌ | C++ struct |
| `Unpack` | kwargs unpacking | ❌ | Python 전용 |
| `TransformersKwargs` | TypedDict 힌트 | ❌ | 타입 힌트 |
| `auto_docstring` | decorator | ❌ | 문서화 전용 |
| `can_return_tuple` | decorator | ❌ | Python 전용 |
| `logging` | 로거 | ❌ | C++ 자체 로깅 사용 |
| `is_torch_flex_available` | 런타임 체크 | ❌ | Metal에서 불필요 |
| `is_torchdynamo_compiling` | 컴파일 체크 | ❌ | PyTorch 전용 |

## C++에서 자체 구현이 필요한 것

### GenerationLoop (GenerationMixin 대체)
```
while (!done):
    logits = model.forward(input_ids, past_kv)
    next_token = sample(logits)  # greedy/top-k/top-p
    input_ids = [next_token]
    if next_token == eos: done = true
```
→ 이미 Mac/cpu 프로젝트에 `generation_session.h`로 구현됨

### Output Structs
```cpp
struct ModelOutput {
    MTL::Buffer* logits;        // [1, vocab_size]
    HybridCache* past_kv;       // KV cache reference
    // hidden_states, attentions → 추론에서 불필요
};
```

### Weight Loading
- safetensors 포맷 직접 파싱 (헤더 JSON + raw tensor bytes)
- 또는 기존 `weight_loader.h` 활용

## `torch.*` 함수 중 Metal 구현 필요한 것

| torch 함수 | 사용 위치 | Metal 매핑 |
|-----------|----------|-----------|
| `F.silu` | GatedDeltaNet conv, MLP | `x * sigmoid(x)` kernel |
| `F.softplus` | GatedDeltaNet gate | `log(1 + exp(x))` kernel |
| `F.sigmoid` | GatedDeltaNet beta | `1 / (1 + exp(-x))` kernel |
| `F.conv1d(groups=C)` | depthwise conv | depthwise conv kernel |
| `F.linear` | 모든 Linear층 | Metal matmul |
| `torch.matmul` | attention score | Metal matmul |
| `torch.softmax` | full attention | Metal softmax |
| `torch.cat` | conv state concat | Metal copy/concat |
| `torch.cumsum` | chunk decay | Metal prefix sum |
| `torch.tril` | causal mask | 상수 마스크 또는 kernel |
| `torch.outer` | delta rule state update | Metal outer product |
| `repeat_interleave` | GQA head expansion | Metal index kernel |
