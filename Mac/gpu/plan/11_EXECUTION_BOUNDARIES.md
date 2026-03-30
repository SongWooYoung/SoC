# Execution Boundaries And Layer Split

## `--layer` Semantics

`Mac/gpu/infer.mm`는 이제 `--layer`를 받아 실행 모드를 세 가지로 나눈다.

1. `--layer 0`
   full CPU. tokenizer 이후 transformer 전체를 CPU runtime이 처리한다.

2. `--layer 1..N-1`
   hybrid. 앞의 `k`개 layer는 GPU가, 나머지 `[k, N)`은 CPU가 처리한다.

3. `--layer -1` 또는 `--layer N`
   full GPU. 전체 layer를 GPU가 처리한다.

여기서 `N`은 manifest `config.num_hidden_layers` 값이다. `k > N`이면 즉시 오류를 낸다.

## CPU Stage / GPU Stage

현재 end-to-end 추론은 완전한 GPU-only가 아니다. 모드별 책임은 다음과 같다.

### Full GPU

CPU:

1. manifest/tokenizer file I/O
2. prompt serialize + tokenizer encode/decode
3. JSON/plain-text output write
4. reduced top-k 결과의 host-side consume

GPU:

1. embedding lookup
2. transformer layers `[0, N)`
3. KV cache update
4. sampler top-k reduction

### Hybrid

CPU:

1. manifest/tokenizer file I/O
2. prompt serialize + tokenizer encode/decode
3. transformer suffix layers `[k, N)`
4. final logits consume + next-token selection
5. JSON/plain-text output write

GPU:

1. embedding lookup
2. transformer prefix layers `[0, k)`
3. prefix KV cache update
4. hidden-state handoff to CPU suffix

### Full CPU

CPU:

1. manifest/tokenizer file I/O
2. prompt serialize + tokenizer encode/decode
3. embedding + transformer layers `[0, N)`
4. sampler
5. JSON/plain-text output write

GPU:

1. 없음

## Remaining CPU Checklist

end-to-end에서 아직 CPU에 남아 있는 구간은 다음 체크리스트로 정리할 수 있다.

1. tokenizer encode/decode
2. chat template serialization
3. manifest/config/token metadata parse
4. tensor file I/O
5. float16/bfloat export weight의 host-side convert/upload
6. hybrid 모드의 suffix transformer layers
7. hybrid 모드의 hidden-state bridge readback
8. full GPU 모드의 reduced top-k 결과 consume
9. plain-text/JSON output formatting and file write

완전한 GPU-only path로 가려면 적어도 `5`, `8`을 줄이고, 별도 tokenizer/device-side sampler 또는 hostless decode strategy가 필요하다.

## Memory Lifetime

추론이 끝난 뒤 모델이 메모리에서 내려가는 별도 unload API는 현재 필수가 아니다. 현재 구현은 RAII로 정리된다.

1. `MetalContext`, `MetalBuffer`는 `unique_ptr`/`shared_ptr` 소멸 시 정리된다.
2. Objective-C Metal object는 ARC가 retain/release를 처리한다.
3. `GenerationContext::Reset()`은 KV cache와 logits buffer를 명시적으로 비운다.
4. `gpu_infer` 프로세스 종료 시 model/context/buffer가 scope 종료와 함께 해제된다.

즉, 지금 상태에서는 “추론 종료 후 모델 메모리 unload를 추가 구현해야만 leak가 막히는 구조”는 아니다. 다만 장수 프로세스에서 model hot-swap을 지원하려면 explicit unload/reload API를 별도로 두는 편이 좋다.