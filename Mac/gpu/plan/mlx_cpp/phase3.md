# MLX→C++ Port — Phase 3: End-to-End Model & Generation 구현

## 목표
Qwen3_5Model + LanguageModel을 완성하고, safetensors 가중치 로딩 + greedy generation을 구현한다.

---

## 3.1 Qwen3_5Model (= TextModel)

**Python**: `language.py` Qwen3_5Model

```
가중치:
  embed_tokens: Embedding(vocab_size, hidden_size)
  layers: [DecoderLayer × num_hidden_layers]
  norm: RMSNorm(hidden_size)

인덱스:
  ssm_idx = 0  (첫 GDN 레이어의 cache index)
  fa_idx = full_attention_interval - 1  (첫 Attention 레이어의 cache index)

Forward(inputs, inputs_embeds, mask, cache, position_ids):
  1. h = embed_tokens(inputs) or inputs_embeds
  2. fa_mask = create_attention_mask(h, cache[fa_idx])  → causal attention mask
  3. ssm_mask = create_ssm_mask(h, cache[ssm_idx])     → SSM padding mask
  4. for layer, c in zip(layers, cache):
       mask = ssm_mask if layer.is_linear else fa_mask
       h = layer(h, mask, c, position_ids)
  5. return norm(h)
```

### Mask 생성

```cpp
// create_attention_mask: causal mask for full-attention layers
// shape [B, 1, S, total_S] — upper-triangular -inf
mx::array create_attention_mask(const mx::array& h, const KVCache* cache) {
    int S = h.shape(1);
    int total = cache ? cache->offset + S : S;
    // causal: mask[i][j] = (j > i + offset) ? -inf : 0
    // MLX SDPA supports mask_mode="causal" → may not need explicit mask
}

// create_ssm_mask: padding mask for GDN layers
// shape [B, S] — 1 for valid, 0 for pad
mx::array create_ssm_mask(const mx::array& h, const ArraysCache* cache) {
    // For inference with no padding: all ones
    // With padding: attention_mask from tokenizer
}
```

**C++ 구현**:
```cpp
struct Qwen3_5Model {
    mx::array embed_w;                    // [vocab_size, hidden_size]
    std::vector<DecoderLayer> layers;
    mx::array norm_w;
    float norm_eps;
    int ssm_idx, fa_idx;

    mx::array operator()(const mx::array& input_ids,
                          std::vector<void*>& cache,
                          const mx::array& position_ids) {
        auto h = embedding(embed_w, input_ids); // [B, S, D]

        auto fa_mask = create_attention_mask(h, cache[fa_idx]);
        auto ssm_mask = create_ssm_mask(h, cache[ssm_idx]);

        for (int i = 0; i < layers.size(); i++) {
            auto mask = layers[i].is_linear ? ssm_mask : fa_mask;
            h = layers[i](h, mask, cache[i], position_ids);
        }

        return mx::fast::rms_norm(h, norm_w, norm_eps);
    }
};
```

---

## 3.2 LanguageModel

**Python**: `language.py` LanguageModel

```
가중치:
  model: Qwen3_5Model
  lm_head: Linear(hidden_size, vocab_size) — tie_word_embeddings면 embed_tokens 재사용

Forward:
  1. position_ids 계산 (text-only: arange, vision: get_rope_index)
  2. hidden = model(inputs, cache=cache, position_ids=position_ids)
  3. if tie_word_embeddings: logits = hidden @ embed_tokens.T
     else: logits = lm_head(hidden)
  4. return logits
```

**C++ 구현**:
```cpp
struct LanguageModel {
    Qwen3_5Model model;
    mx::array lm_head_w;     // [vocab_size, hidden_size] or empty if tied
    bool tie_word_embeddings;

    mx::array operator()(const mx::array& input_ids,
                          std::vector<void*>& cache) {
        // Position IDs (text-only: simple arange)
        int offset = get_cache_offset(cache);
        int S = input_ids.shape(1);
        auto pos = mx::arange(offset, offset + S);
        pos = mx::expand_dims(pos, 0); // [1, S]
        // MRoPE: [3, 1, S] — same pos for all 3 components (text-only)
        pos = mx::stack({pos, pos, pos}, 0); // [3, 1, S]

        auto hidden = model(input_ids, cache, pos);

        if (tie_word_embeddings) {
            return mx::matmul(hidden, mx::transpose(model.embed_w));
        }
        return linear(hidden, lm_head_w);
    }
};
```

---

## 3.3 Generation Loop

```cpp
struct GenerationSession {
    LanguageModel& lm;
    std::vector<void*> cache;  // KVCache or ArraysCache per layer
    int eos_token = 248046;    // <|im_end|>

    std::vector<int> generate(const std::vector<int>& prompt_tokens, int max_tokens) {
        // 1. Init cache
        init_cache();

        // 2. Prefill
        auto input = mx::array(prompt_tokens.data(), {1, (int)prompt_tokens.size()}, mx::int32);
        auto logits = lm(input, cache);           // [1, S, vocab]
        mx::eval(logits);                          // Force evaluation

        auto next_token = mx::argmax(logits(0, -1), -1); // greedy
        mx::eval(next_token);
        int token = next_token.item<int>();

        std::vector<int> output = {token};

        // 3. Decode loop
        for (int i = 0; i < max_tokens && token != eos_token; i++) {
            auto input_1 = mx::array(&token, {1, 1}, mx::int32);
            logits = lm(input_1, cache);
            mx::eval(logits);

            next_token = mx::argmax(logits(0, -1), -1);
            mx::eval(next_token);
            token = next_token.item<int>();
            output.push_back(token);
        }

        return output;
    }

    void init_cache() {
        cache.clear();
        for (auto& layer : lm.model.layers) {
            if (layer.is_linear) {
                cache.push_back(new ArraysCache());
            } else {
                cache.push_back(new KVCache());
            }
        }
    }
};
```

### MLX eval() 패턴

MLX는 lazy evaluation → `mx::eval(array)`로 계산 강제 실행.
- Prefill 후: `eval(logits)` → 전체 prefill 그래프 실행
- Decode 각 step: `eval(logits)` → 1 token씩 실행
- 메모리: unified memory (CPU+GPU 공유) → 복사 불필요

---

## 3.4 Tokenizer

py_cpp와 동일한 토크나이저 재사용:
- `tokenizer_runtime.h` → vocab.json + merges.txt 기반 BPE
- Chat template: `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`

---

## 3.5 가중치 로딩

### Safetensors 직접 로드 (MLX API 사용)

```cpp
#include <mlx/io.h>  // or mlx/utils.h

// MLX가 safetensors 로드 기능 내장 (load_safetensors 또는 load)
auto weights = mx::load("model.safetensors");  // map<string, mx::array>

// sanitize 변환 (qwen3_5.py에서):
// 1. conv1d weight transpose 확인
// 2. norm weight += 1.0 offset
for (auto& [key, arr] : weights) {
    if (key.find("norm.weight") != std::string::npos) {
        arr = arr + 1.0f;  // RMSNorm offset
    }
}
```

### 또는 직접 safetensors 파서

py_cpp의 `weight_loader.h`를 확장하여 MLX array로 변환:
```cpp
mx::array from_safetensors(const char* data, size_t offset, size_t nbytes,
                            const std::vector<int>& shape, mx::Dtype dtype) {
    return mx::array(data + offset, shape, dtype);
}
```

---

## 상태
- [ ] 3.1 Qwen3_5Model (embed + layer loop + mask dispatch + final norm)
- [ ] 3.2 LanguageModel (lm_head, position_ids, tie_word_embeddings)
- [ ] 3.3 Generation loop (prefill → decode, eval pattern)
- [ ] 3.4 Tokenizer 연동 (py_cpp 재사용)
- [ ] 3.5 Safetensors 가중치 로더
- [ ] 3.T 통합 테스트 (첫 번째 토큰 일치)
