# Implementation Decisions

이 문서는 조사 결과를 바탕으로 현재 시점에서 확정한 구현 결정을 짧게 정리한다.

## 확정 사항

1. baseline target model은 `Qwen/Qwen3-0.6B`다.
2. export contract baseline은 HF tensor names를 보존한다.
3. runtime tensor core는 `Storage + Tensor` 구조로 간다.
4. Phase 1 kernel은 contiguous-first다.
5. public KV cache layout은 `[layer][batch][kv_head][seq][head_dim]`다.
6. tokenizer는 full HF fast tokenizer 재구현이 아니라 exporter-assisted runtime format으로 간다.
7. Qwen3 template는 generic Jinja engine이 아니라 Qwen3-specific builder를 먼저 구현한다.
8. `tie_word_embeddings = true`를 반영해 embedding/lm_head tying을 지원한다.
9. Q/K/V는 export 시 separate, runtime opt cache에서 fused 가능으로 둔다.
10. batch-aware 설계를 유지하되 decode 최적화는 batch=1 우선이다.
11. long context YaRN은 기본 비활성화다.
12. CPU optimization 순서는 `correctness -> layout -> prepack -> threading -> quantization -> fusion`이다.

## Phase 1 필수 구현

1. `ManifestLoader`
2. `WeightLoader`
3. `TokenizerRuntimeLoader`
4. `Qwen3TemplateBuilder`
5. `Storage`
6. `Tensor`
7. `Linear`, `RMSNorm`, `RoPE`, `Attention`, `MLP`, `Block`, `CausalLM`
8. `KVCache`
9. `GenerationSession`
10. `Sampler`

## Defer 항목

1. generic tokenizer.json interpreter
2. generic Jinja template engine
3. YaRN long-context default-on support
4. packed key cache public ABI
5. fused QKV export as primary contract
6. autograd engine
7. optimizer / checkpoint write-back

## 리스크 메모

1. tokenizer simplification이 충분하지 않으면 exporter 확장이 필요하다.
2. tied embedding/lm_head를 잘못 처리하면 logits parity가 깨진다.
3. GQA head mapping을 잘못 구현하면 attention output이 조용히 틀릴 수 있다.
4. long-context를 너무 일찍 넣으면 baseline correctness 검증이 느려진다.