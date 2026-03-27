# Recommended Directory Layout

## 1. Top-Level Target Layout Under Mac/cpu

```text
Mac/cpu/
  CMakeLists.txt or Makefile
  main.cpp
  plan/
  include/
    soc/
      core/
      io/
      ops/
      nn/
      model/
      runtime/
      train/
      tokenizer/
      util/
  src/
    core/
    io/
    ops/
    nn/
    model/
    runtime/
    train/
    tokenizer/
    util/
  tests/
    unit/
    integration/
    model/
  benchmarks/
  assets/
    schemas/
    tokenizer/
```

현재 `header/` 기반 구조는 초기 프로토타입에는 괜찮지만, 이후 확장성과 빌드 시간, 계층 분리를 위해 `include/` + `src/` 구조로 전환하는 것이 맞다.

## 2. Folder-by-Folder Responsibility

### 2.1 `include/soc/core/`

핵심 데이터 구조 계층.

예상 파일:

1. `dtype.h`
2. `shape.h`
3. `storage.h`
4. `allocator.h`
5. `tensor.h`
6. `tensor_view.h`
7. `memory_span.h`

내용:

1. DType enum
2. shape/stride 유틸리티
3. owning storage와 non-owning view
4. aligned allocator
5. Tensor object

### 2.2 `include/soc/io/`

모델과 자산을 읽는 계층.

예상 파일:

1. `json_reader.h`
2. `manifest_loader.h`
3. `weight_loader.h`
4. `checkpoint_index.h`
5. `mmap_reader.h`
6. `tokenizer_runtime_loader.h`

내용:

1. manifest parsing
2. tensor metadata index
3. raw weight file read/mmap
4. config/generation_config extraction
5. tokenizer runtime export parsing

### 2.3 `include/soc/ops/`

Tensor op public interface.

예상 파일:

1. `binary_ops.h`
2. `matmul.h`
3. `norm.h`
4. `activation.h`
5. `softmax.h`
6. `rope.h`
7. `embedding.h`
8. `attention_ops.h`

내용:

1. shape inference
2. dtype rule
3. kernel dispatch

### 2.4 `src/ops/`

op 구현과 실제 kernel 연결.

예상 파일:

1. `matmul.cpp`
2. `matmul_fp16.cpp`
3. `matmul_packed.cpp`
4. `softmax.cpp`
5. `rmsnorm.cpp`
6. `rope.cpp`
7. `embedding.cpp`
8. `attention_ops.cpp`

### 2.5 `include/soc/nn/`

reusable module layer.

예상 파일:

1. `module.h`
2. `parameter.h`
3. `embedding.h`
4. `linear.h`
5. `rmsnorm.h`
6. `attention.h`
7. `mlp.h`
8. `transformer_block.h`

내용:

1. parameter registry
2. submodule tree
3. forward interface

### 2.6 `include/soc/model/`

모델별 graph와 weight mapping.

예상 파일:

1. `qwen3_config.h`
2. `qwen3_mapping.h`
3. `qwen3_attention.h`
4. `qwen3_block.h`
5. `qwen3_causal_lm.h`

내용:

1. HF config 필드 구조체
2. state_dict key to module parameter binding
3. GQA/QKV/MLP 구조 반영

### 2.7 `include/soc/runtime/`

실제 generation runtime.

예상 파일:

1. `generation_context.h`
2. `kv_cache.h`
3. `sampler.h`
4. `prefill_decode.h`
5. `session.h`

내용:

1. prompt prefill
2. decode step
3. cache update
4. temperature/top-p/top-k sampling

### 2.8 `include/soc/tokenizer/`

backendless tokenizer.

예상 파일:

1. `tokenizer.h`
2. `tokenizer_json_loader.h`
3. `bpe.h`
4. `normalizer.h`
5. `pretokenizer.h`
6. `decoder.h`
7. `qwen3_template_builder.h`

내용:

1. `tokenizer.json` 파싱
2. BPE merge/apply
3. special token 처리
4. encode/decode
5. Qwen3 thinking / non-thinking template preset

### 2.9 `include/soc/train/`

Phase 2 이후 training 계층.

예상 파일:

1. `autograd_tensor.h`
2. `graph_node.h`
3. `optimizer.h`
4. `adamw.h`
5. `loss.h`
6. `checkpoint_writer.h`

초기에는 skeleton만 두고 실제 구현은 뒤로 미룬다.

### 2.10 `tests/`

필수 테스트 구분:

1. `unit/`
   tensor, storage, dtype, shape, allocator, matmul, softmax, tokenizer

2. `integration/`
   manifest load, weight load, block forward, kv cache decode

3. `model/`
   Qwen3 layer-by-layer parity check, logits sanity, short prompt generation

## 3. Migration Plan From Current Structure

현재 `header/` 기반 코드에서 새 구조로 가는 방법:

1. 현재 `matrix.h`, `matrix_ops.h`, `activation.h`, `attention.h`는 legacy prototype으로 둔다.
2. 새 runtime은 `include/soc/*`, `src/*`에서 별도 개발한다.
3. legacy kernel 중 재사용 가능한 수치 루프는 새 `ops/`로 옮긴다.
4. old and new runtime을 한동안 병행 유지한다.

추가 마이그레이션 규칙:

1. `LLM_interpreter`가 만든 `manifest.json`과 `weights/*.bin`은 새 `io/` 계층의 유일한 진실 공급원으로 둔다.
2. `tokenizer.json`은 참고 자산으로 보관하되, 새 runtime은 우선 `tokenizer_runtime.json`을 읽는다.
3. 기존 `header/attention.h`의 `KVCache`와 `matrix_ops.h`의 packed matmul은 새 runtime 최적화 구현의 참고 prototype으로 취급한다.

이 방식이 안전하다. 기존 테스트를 깨지 않으면서 새 runtime을 올릴 수 있다.
