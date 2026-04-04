mlx-vlm의 코드를 mac mini M4, 32GB (System on Chip + Unified Memory)에 최대한 최적화를 시키면 성능을 얼마나 향상 시킬 수 있을 지 궁금함 

base : Qwen3.5-9B-MLX-8bit
내가 수정할것: qwen3_5_mlx 

0. 의문점 : base 쪽 코드를 그대로 porting 했는데 왜 성능 차이가 발생하는 가?

MLX library: prefill 325.039 ms, decode 85.193 ms/tok, throughput 10.473 tok/s
qwen3_5_mlx가 prefill 831.370 ms, decode 119.557 ms/tok, throughput 7.468 tok/s

     1) 컴파일을 다르게 하나?
                 - 조사 결과:
                     - MLX 본체는 `.repo_cache/mlx/CMakeLists.txt` 기준으로 `CMAKE_CXX_STANDARD=20`, `CMAKE_POSITION_INDEPENDENT_CODE=ON`, `CMAKE_EXPORT_COMPILE_COMMANDS=ON`을 사용한다.
                     - 로컬에 빌드된 MLX는 `.repo_cache/mlx/build/CMakeCache.txt` 기준으로 `CMAKE_BUILD_TYPE=Release`, `BUILD_SHARED_LIBS=OFF`, `MLX_BUILD_METAL=ON`, `MLX_BUILD_CPU=ON`, `MLX_BUILD_SAFETENSORS=ON`, `MLX_BUILD_GGUF=ON`, `MLX_METAL_JIT=OFF` 상태다.
                     - MLX 문서 `docs/src/dev/mlx_in_cpp.rst`는 C++ 사용자 코드도 CMake에서 `find_package(MLX CONFIG REQUIRED)`로 붙이고 `cmake -B build -DCMAKE_BUILD_TYPE=Release`로 빌드하는 방식을 권장한다.
                     - 현재 우리 `Mac/gpu/Makefile`은 custom MLX 바이너리를 `clang++ -std=c++20 -O2 -I../../.repo_cache/mlx ... -L../../.repo_cache/mlx/build -lmlx -framework Metal -framework Foundation -framework Accelerate -framework QuartzCore`로 직접 링크한다.
                 - 코멘트:
                     - 큰 차이는 없다. 가장 중요한 부분은 이미 `libmlx.a` 안에서 Release 모드로 빌드되어 있고, 우리 코드도 `-std=c++20 -O2`로 그 라이브러리를 링크하고 있다.
                     - 즉, “MLX 본체가 특별한 비밀 컴파일 옵션으로 빨라서”라기보다는, 같은 MLX를 붙여 놓고 우리가 상위 런타임 계층을 덜 효율적으로 쓰고 있을 가능성이 더 높다.
                     - 다만 차이가 아예 0은 아니다. 우리는 MLX package config가 주는 `MLX_CXX_FLAGS`나 include/link 설정을 쓰지 않고 수동 링크한다. 또 CMake의 `Release` 기본 최적화와 정확히 같은 플래그 세트를 쓰는 것도 아니다.
                     - 추가로 MLX CI에는 `.github/actions/build-macos/action.yml` 기준 `MinSizeRel + MLX_METAL_JIT=ON` 조합이 보이지만, 현재 로컬 build는 `Release + MLX_METAL_JIT=OFF`라서 이 부분은 “MLX 자체 빌드 실험” 대상으로 따로 볼 수 있다. 다만 현재 성능 차이의 주원인으로 보이진 않는다.
                 - 테스트:
                     - 1차: 현재 `Makefile`에서 custom binary를 `-O0/-O2/-O3/-flto`로 바꿔 20 prompt 재측정한다.
                     - 2차: MLX 문서 방식대로 작은 `CMakeLists.txt`를 만들어 `find_package(MLX CONFIG REQUIRED)` + `CMAKE_BUILD_TYPE=Release`로 같은 바이너리를 다시 빌드해 결과를 비교한다.
                     - 3차: 필요하면 `.repo_cache/mlx`를 별도 build dir에서 `-DCMAKE_BUILD_TYPE=MinSizeRel -DMLX_METAL_JIT=ON`으로 재빌드한 `libmlx`에 custom binary를 다시 링크해 본다.
                     - 판정 기준: compile/link 계열 변경만으로 개선 폭이 5% 이내면 1번은 후순위로 내리고, 10% 이상이면 build-system 차이도 실제 최적화 항목으로 승격한다.

     2) 코드 구현의 문제인가?
         - 조사 결과:
             - base `Qwen3.5-9B-MLX-8bit`는 Python에서 `mlx_vlm.models.qwen3_5.qwen3_5.Model`로 로드되고, text 경로는 `mlx_vlm.models.qwen3_5.language.LanguageModel -> Qwen3_5Model -> Qwen3_5DecoderLayer` 구조다.
             - `mlx.nn.Module` 쪽에는 PyTorch의 `register_forward_hook` 같은 공개 hook API가 없다. `mlx_vlm`에도 stage timing을 바로 뽑아주는 내장 trace hook은 보이지 않았다.
             - 대신 base는 Python 소스 계층이 열려 있어서, `LanguageModel.__call__`, `Qwen3_5Model.__call__`, `Qwen3_5DecoderLayer.__call__`, `Qwen3_5Attention.__call__`, `Qwen3_5GatedDeltaNet.__call__`, `Qwen3_5MLP.__call__`, `KVCache.update_and_fetch()`를 monkey patch 또는 wrapper로 감싸서 stage별 계측을 넣는 것은 가능하다.
             - 다만 base의 실제 runner인 `/opt/homebrew/lib/python3.11/site-packages/mlx_vlm/generate.py`는 `prefill_step_size`, `wired_limit`, `mx.clear_cache()`, `mx.async_eval(next_y)`를 사용한다. 즉 단순히 함수 입출력 시간만 재면 lazy/asynchronous scheduling 시간과 실제 device 실행 시간이 섞인다.
             - MLX 자체는 `mx.metal.start_capture()` / `mx.metal.stop_capture()`로 GPU trace를 저장하는 기능을 제공한다. 이것은 layer semantic hook의 대체재는 아니지만, Python/C++ 양쪽에서 동일하게 kernel-level 상관관계를 확인하는 보조 수단으로 쓸 수 있다.
         - 코멘트:
             - 결론적으로 “base에서도 같은 방식의 비교가 가능한가?”에 대한 답은 가능하다. 다만 built-in hook을 켜는 방식이 아니라, 우리가 별도 trace harness를 만들어 Python model 계층을 감싸는 방식이어야 한다.
             - 중요한 점은 benchmark와 trace를 분리해야 한다는 것이다. trace 모드에서 stage 경계마다 `mx.eval(...)` 또는 `mx.synchronize()`를 넣으면 원래의 async/chunked 실행이 깨지므로, 이 숫자는 병목 위치를 찾는 진단용 숫자이지 최종 tok/s 숫자가 아니다.
             - apples-to-apples 비교를 하려면 1차는 `stream_generate()`를 직접 비교하지 말고, base도 custom도 동일한 입력 토큰과 동일한 cache schedule로 `language_model(...)` forward를 직접 호출하는 “model-core trace”를 만든다. 이후 2차로 official runner의 `async_eval/prefill chunking/wired_limit`가 주는 이득을 따로 분리해서 본다.
             - 실제 4-prompt trace 결과를 보면 custom의 평균은 `prefill 1844.759 ms`, `decode 159.884 ms/tok`, `5.018 tok/s`였고 base는 `prefill 514.607 ms`, `decode 115.162 ms/tok`, `7.670 tok/s`였다. 즉 병목 위치를 찾는 진단 trace에서도 gap은 그대로 재현된다.
             - stage 기준으로는 prefill/decode 모두 `mlp`, `linear_cache_update`, `lm_head`가 가장 크게 벌어졌다. 특히 decode에서는 `linear_cache_update`가 base 대비 `+997.365 ms` 누적 sync delta로 가장 컸다.
             - 반대로 `rope`는 이번 trace에서 주원인으로 보이지 않았다. prefill delta는 작고 decode에서는 오히려 custom 쪽 `sync_ms`가 base보다 작게 나왔다.
             - 또 하나 중요한 점은 custom `linear_attention`의 `dispatch_ms` 자체가 비정상적으로 크다는 것이다. 평균 기준 prefill dispatch는 base `2.032 ms` 대비 custom `325.507 ms`, decode dispatch는 base `43.622 ms` 대비 custom `1020.052 ms`였다. 즉 device sync만 느린 게 아니라 host-side graph construction / operator composition 비용도 같이 커져 있다.
             - arrays-style conv cache update 실험을 해 보니 full trace는 오히려 `prefill 2290.630 ms`, `decode 160.046 ms/tok`, `4.978 tok/s`로 baseline custom보다 좋아지지 않았다. 즉 단순 `conv_state` tail 처리 자체는 주원인이 아니었다.
             - 그 다음 `gated_delta` follow-up을 보면, custom C++에 upstream의 `@mx.compile`에 대응하는 `compiled_ops` 경로를 넣었을 때 full trace 평균이 `prefill 466.022 ms`, `decode 151.419 ms/tok`, `6.031 tok/s`까지 좋아졌다. 특히 `linear_attention` prefill dispatch는 `325.507 -> 162.022 ms (-50.2%)`, prefill sync는 `216.420 -> 57.751 ms (-73.3%)`로 크게 줄었다.
             - isolated custom `gated_delta` microbench에서도 `compiled_ops`는 ops 대비 prefill sync `14.664 -> 9.619 ms (-34.4%)`, decode sync `0.474 -> 0.405 ms (-14.6%)`였다. 즉 compile 부재는 실제 병목이었다.
             - 하지만 upstream Python `gated_delta_update`를 `use_kernel=False/True`로 직접 나누어 재보면 prefill sync `9.795 -> 0.729 ms (-92.6%)`, decode sync `0.473 -> 0.333 ms (-29.6%)`로 kernel 경로 이득이 더 크다. 즉 compile 부재만이 아니라 upstream의 Metal kernel recurrent path 부재가 더 강한 원인이다.
             - 따라서 2번의 현재 결론은 이렇다. `linear_attention` anomaly의 핵심은 `linear_cache_update` 자체보다 `gated_delta_update` 구현 차이이며, 그 안에서도 `@mx.compile`/graph reuse 부재가 1차 원인, Metal kernel recurrent update 부재가 더 큰 2차이자 최종 원인 후보다.
         - 테스트 / 실험 설계:
             - 1차: base trace harness를 Python으로 만든다. 새 스크립트에서 official model을 로드한 뒤, 위 메서드들을 wrapper로 감싸 `prefill` 1회와 `decode` N step 동안 stage별 누적 시간을 JSON으로 저장한다.
             - 1차 측정 단위는 두 종류로 나눈다. `dispatch_ms`는 wrapper 안의 순수 Python/graph scheduling 시간, `sync_ms`는 stage 출력 또는 cache state에 대해 `mx.eval(...)` 후 `mx.synchronize()`까지 포함한 시간이다. 해석은 `sync_ms`를 우선으로 하고 `dispatch_ms`는 host overhead 참고치로만 본다.
             - 1차 stage 이름은 최소한 `input_embeddings`, `position_ids`, `rope`, `full_attention`, `linear_attention`, `mlp`, `final_norm`, `lm_head`, `kv_cache_update`, `linear_cache_update`, `sampler_sync(argmax/item)`로 맞춘다. custom C++ 쪽 trace도 같은 schema로 맞춘다.
             - 1차 실행 조건은 prompt 4개만 쓴다. short 2개 + long 2개, `max_new_tokens=32`, temperature 0, 같은 tokenizer/prompt template 사용. 목적은 병목 위치 확인이지 throughput 재측정이 아니다.
             - 2차: model-core trace를 base vs custom에 대해 동일 입력으로 돌린다. output은 prompt별 총합뿐 아니라 `stage -> {calls, dispatch_ms, sync_ms}` 구조의 JSON/CSV로 남긴다.
             - 3차: runner overhead trace를 별도로 만든다. base는 `stream_generate()` 경로에서 `get_input_embeddings`, chunked prefill loop, `_step`, `mx.async_eval`, first-token sync를 감싸고, custom은 현재 runner에 같은 구간 타이머를 넣는다. 이 실험은 “구현체 차이”와 “실행기 차이”를 분리하기 위한 것이다.
             - 4차: 필요하면 representative prompt 1개에 대해 base와 custom 모두 `mx.metal.start_capture()`로 GPU capture를 떠서, trace에서 가장 큰 구간이 실제로 어떤 Metal kernel 묶음에 대응하는지 확인한다. semantic trace와 GPU trace가 같은 병목을 가리키는지 교차검증한다.
             - 산출물은 세 개로 고정한다. `base_stage_trace.json`, `custom_stage_trace.json`, `stage_trace_diff.md`. 마지막 문서에는 prefill/decode를 분리해서 stage별 비중, 절대 ms 차이, 배수 차이를 정리한다.
             - 판정 기준: 특정 stage의 `sync_ms`가 base 대비 custom에서 15% 이상 크게 반복되면 실제 최적화 타깃으로 올린다. 반대로 model-core trace에서는 차이가 작고 runner trace에서만 차이가 크면, 그 항목은 model 구현보다 `async/chunking/cache management` 문제로 분류한다.

     3) 추가적인 의심사항

         3-1) KV cache 증가 방식이 base와 다르다.
         - 근거: 우리 구현의 `models/qwen3_5_mlx/language.h`에서 `KVCache::update_and_fetch()`는 decode 때마다 `mx::concatenate`로 key/value를 계속 붙인다. 반면 base가 쓰는 `/opt/homebrew/lib/python3.11/site-packages/mlx_lm/models/cache.py`의 `KVCache`는 256 token step으로 미리 버퍼를 잡고 in-place write 후 slice만 반환한다.
         - 코멘트: decode 성능 저하의 가장 강한 후보. 현재 decode delta가 `+34.364 ms/tok`인데, attention layer cache reallocation이 그 차이를 상당 부분 설명할 수 있다.
         - 테스트: custom `KVCache`를 없애고 upstream 스타일의 step-based preallocated cache를 그대로 이식해 A/B 테스트한다. short prompt 10개만 먼저 돌려도 decode 개선 여부를 빠르게 볼 수 있다.

         3-2) linear attention용 conv/recurrence cache도 매 step 재할당 성향이 있다.
         - 근거: `GatedDeltaNet::forward()`에서 `conv_state`를 `concatenate + take_last_tokens`로 갱신한다. upstream `/opt/homebrew/lib/python3.11/site-packages/mlx_lm/models/lfm2.py`의 `ShortConv`는 `ArraysCache`와 `advance()`를 이용해 상태를 유지하고 필요한 구간만 갱신한다.
         - 코멘트: full attention만 느린 게 아니라 linear attention 경로도 decode 누적 비용을 키울 가능성이 높다.
         - 테스트: linear layer만 따로 남긴 micro-benchmark를 만든다. 1 token decode를 512회 반복하면서 `conv_state` 갱신 시간을 분리 측정한다. 이후 upstream `ArraysCache`/`ShortConv` 방식으로 바꿔서 동일 측정값을 비교한다.

         3-3) RoPE 구현이 CPU loop 중심이라 prefill 손해가 클 수 있다.
         - 근거: `models/qwen3_5_mlx/language.h`의 `RotaryEmbedding::operator()`는 `mx::eval` 후 `data<int>()`, `data<float>()`를 꺼내 C++ loop로 `merged`, `emb` 벡터를 만들고 다시 `mx::array`로 복원한다. base `/opt/homebrew/lib/python3.11/site-packages/mlx_lm/models/lfm2.py`는 `nn.RoPE`를 바로 사용한다.
         - 코멘트: prefill delta `+506.331 ms`를 설명하는 유력한 후보다. 특히 prompt 길이가 길수록 이 비용은 바로 커진다.
         - 테스트: `nn.RoPE` 또는 완전 vectorized `mx` 연산으로 대체한 브랜치를 만든 뒤 long prompt 10개 prefill만 측정한다. stage timer에서 rope 구간이 크게 줄어드는지 본다.

         3-4) text-only 경로인데도 position id를 매번 새로 크게 만든다.
         - 근거: `LanguageModel::forward()`는 text-only에서도 `make_position_ids()`로 `[3, B, S]` 배열을 새로 만들고, attention 경로마다 이를 넘긴다. base LFM2는 `rope(offset=cache.offset)` 식으로 offset 기반 처리를 한다.
         - 코멘트: 수식은 맞더라도 구현이 지나치게 무겁다. Qwen3.5의 mRoPE를 유지하더라도 text-only에서는 세 축이 동일하므로 더 싼 경로를 둘 수 있다.
         - 테스트: text-only 전용 fast path를 추가해 3개 축이 같은 position id를 재사용하거나 offset 기반 RoPE를 적용한다. prefill/1-token decode 둘 다 비교한다.

         3-5) generation loop가 매 step 강제 sync를 건다.
         - 근거: `GenerationSession::generate()`와 `test/test_mlx_quantized_output_eval.cpp` 모두 `mx::eval(logits) -> argmax -> mx::eval(current) -> item<int>()` 패턴으로 매 token 동기화한다. base `/opt/homebrew/lib/python3.11/site-packages/mlx_vlm/generate.py`는 `mx.async_eval(next_y)`로 다음 step과 평가를 겹치게 한다.
         - 코멘트: 같은 연산량이어도 host/device sync 패턴만으로 tok/s가 크게 떨어질 수 있다.
         - 테스트: custom runner에 async path를 추가한다. `next token` 계산을 미리 스케줄하고, `item()` 호출 직전까지만 sync 하도록 바꾼 뒤 decode tok/s를 비교한다.

         3-6) base는 prefill을 운영 레벨에서 chunking한다.
         - 근거: `/opt/homebrew/lib/python3.11/site-packages/mlx_vlm/generate.py`는 `prefill_step_size`, `wired_limit`, `mx.clear_cache()`를 사용해 긴 prefill을 나눠 처리한다. 우리 runner는 prompt 전체를 한 번에 넣는다.
         - 코멘트: 현재 prompt가 그렇게 길지 않아도 long prompt에서 working set이 커지면 prefill latency가 늘어난다. Mac mini M4 32GB에서 unified memory pressure를 낮추는 효과가 있을 수 있다.
         - 테스트: custom runner에도 `prefill_step_size=256/512/1024/2048` 옵션을 추가하고 long prompt 10개만 따로 sweep 한다. throughput보다 prefill ms와 peak memory 변화를 같이 기록한다.

         3-7) tied embedding / lm_head 경로가 upstream module과 다르다.
         - 근거: base LFM2는 `embed_tokens.as_linear(out)`를 사용한다. 우리 쪽은 `mlx_helpers::embedding()`에서 quantized row를 `dequantize`하고, lm_head는 generic `quantized_matmul`을 호출한다.
         - 코멘트: 기능적으로는 맞더라도 embedding gather와 lm_head projection이 upstream quantized embedding module보다 덜 최적화됐을 가능성이 있다.
         - 테스트: 별도 micro-benchmark로 `embedding only`, `lm_head only` 시간을 뽑는다. prompt token 길이와 vocab projection 크기를 고정해 base 모듈과 custom helper를 직접 비교한다.

         3-8) 현재 benchmark에 correctness noise가 섞여 있다.
         - 근거: 비교 결과상 generated token은 `16/20` 일치인데 output text는 `10/20`만 일치했다. 일부는 실제 divergence지만 일부는 custom 쪽 출력 끝에 `<|im_end|>`가 남아서 생긴 mismatch다.
         - 코멘트: correctness mismatch가 있는 상태의 속도 비교는 해석을 조심해야 한다. 불필요한 special token 생성/후처리 차이가 decode step 수와 stop condition을 흔들 수 있다.
         - 테스트: stop token 처리와 postprocess를 먼저 base와 동일하게 맞춘다. 그 다음 “완전히 동일한 출력만 모은 subset”과 “전체 prompt”를 분리해서 성능을 다시 비교한다.

         3-9) 수동 weight mapping/loader는 correctness와 perf 둘 다 흔들 수 있다.
         - 근거: `test/test_mlx_quantized_output_eval.cpp`는 shard를 직접 합치고 tensor 이름을 수동 매핑한다. conv weight reorder와 quantized param 조립도 수동이다. base는 model class의 `sanitize()`와 module update 경로를 사용한다.
         - 코멘트: 이건 steady-state 성능 자체보다는 correctness risk에 가깝지만, 잘못된 layout이면 특정 layer에서 extra transpose/cast가 숨어 있을 수 있다.
         - 테스트: layer별 output을 base와 custom에서 같은 prompt에 대해 dump해서 cosine similarity / max abs diff를 비교한다. 처음 divergence가 나는 layer를 찾으면 그 지점부터 layout과 dtype을 본다.

         3-10) measurement 방식이 완전히 apples-to-apples가 아닐 수 있다.
         - 근거: base는 `mlx_vlm.generate.stream_generate()`의 stream/async/wired-limit 운영 아래에서 측정되고, custom은 단일 blocking loop에서 측정된다.
         - 코멘트: “모델 구현” 차이와 “runner 운영” 차이가 섞여 있다. 최적화 계획에서는 둘을 분리해야 한다.
         - 테스트: runner-only benchmark를 만든다. 동일한 `qwen3_5_mlx` model forward를 두 가지 실행기(blocking vs async/chunked)로 돌려서 runner overhead를 먼저 분리한다.

     4) 현재 우선순위 정리
         - 1순위: `gated_delta_update`의 Metal kernel / fused recurrent path 이식 또는 동등 최적화
         - 2순위: `KVCache`, generation sync 패턴
         - 3순위: `RoPE`, position id 생성, prefill chunking, lm_head/embedding quantized path, loader 정합성