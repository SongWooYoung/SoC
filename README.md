# SoC Runtime Workspace

이 저장소는 Qwen3 계열 모델을 외부 텐서 백엔드 없이 순수 C++ CPU runtime으로 실행하기 위한 실험용 workspace다.

현재 기준으로 가장 정리된 경로는 `Mac/cpu`, `Mac/gpu`, `LLM_interpreter` 조합이다.

GPU 경로는 이제 real export bundle 기준 CPU/GPU sequence parity regression과 성능 보고 경로까지 포함한다. Metal runtime 설계와 결과 문서는 [Mac/gpu/plan/README.md](/Users/song-ganghui/Documents/SoC/Mac/gpu/plan/README.md)에서 시작할 수 있다.

## Workspace Layout

1. `LLM_interpreter`
   Hugging Face snapshot을 runtime manifest, tokenizer runtime, binary weights로 export하는 Python tooling
2. `Mac/cpu`
   export bundle을 읽어 prompt tokenize -> prefill/decode -> one-step generation까지 수행하는 pure C++ CPU runtime
3. `models/raw/qwen3-0.6b`
   실제 `Qwen/Qwen3-0.6B` raw snapshot
4. `models/cpp/qwen3-0.6b`
   C++ runtime이 읽는 실제 exported bundle
5. `Mac/gpu`
   Apple Metal 기반 GPU runtime, real-bundle regression, sweep automation, report artifacts

## Validated Path

현재 실제로 검증된 기본 경로는 아래와 같다.

1. Python exporter로 real bundle 생성
2. C++ runtime이 `manifest.json`과 `tokenizer_runtime.json`을 읽어 startup validation 수행
3. runtime tokenizer가 ByteLevel BPE의 byte-to-unicode, decoder metadata, `continuing_subword_prefix`를 반영해 encode/decode 수행
4. real Qwen3 bundle 기준 integration golden 두 경로가 고정

검증된 golden targets:

1. `cd /Users/song-ganghui/Documents/SoC/Mac/cpu && make integration-qwen3-golden`
   raw prompt 기준 golden
2. `cd /Users/song-ganghui/Documents/SoC/Mac/cpu && make integration-qwen3-chat-golden`
   chat template 기준 golden
3. `cd /Users/song-ganghui/Documents/SoC/Mac/cpu && make integration-qwen3-goldens`
   위 두 golden을 연속 실행하는 aggregate target
4. `cd /Users/song-ganghui/Documents/SoC/Mac/cpu && make regression`
   clean build + core tests + two goldens를 한 번에 실행하는 상위 aggregate target

CI나 로컬 자동화에서 재사용할 수 있도록 `Mac/cpu/Makefile`은 다음 단계 target도 제공한다.

1. `make build-tests`
   core test binary build only
2. `make build-integration`
   HF integration test binary build only
3. `make check-core`
   core tests execute
4. `make check-goldens`
   two real-bundle golden paths execute

## Quickstart

### 1. Python environment

Python tooling은 workspace 가상환경을 사용한다.

설치:

```bash
cd /Users/song-ganghui/Documents/SoC/LLM_interpreter
/Users/song-ganghui/Documents/SoC/.venv/bin/python -m pip install --upgrade pip
/Users/song-ganghui/Documents/SoC/.venv/bin/python -m pip install -r requirements.txt
```

```bash
cd /Users/song-ganghui/Documents/SoC/LLM_interpreter
/Users/song-ganghui/Documents/SoC/.venv/bin/python -m unittest test_convert_py_to_cpp.py test_export_test_bundle.py
```

### 2. Export a real Qwen3 bundle

세부 exporter 옵션과 사용 예시는 [LLM_interpreter/README.md](/Users/song-ganghui/Documents/SoC/LLM_interpreter/README.md)에 정리되어 있다.

```bash
cd /Users/song-ganghui/Documents/SoC/LLM_interpreter
/Users/song-ganghui/Documents/SoC/.venv/bin/python model_downloader.py \
  --model-id Qwen/Qwen3-0.6B \
  --download-dir /Users/song-ganghui/Documents/SoC/models/raw/qwen3-0.6b \
  --export-dir /Users/song-ganghui/Documents/SoC/models/cpp/qwen3-0.6b \
  --dtype float16
```

이미 snapshot이 있다면 exporter만 다시 실행해도 된다.

```bash
cd /Users/song-ganghui/Documents/SoC/LLM_interpreter
/Users/song-ganghui/Documents/SoC/.venv/bin/python convert_py_to_cpp.py \
  --model-dir /Users/song-ganghui/Documents/SoC/models/raw/qwen3-0.6b \
  --output-dir /Users/song-ganghui/Documents/SoC/models/cpp/qwen3-0.6b \
  --model-id Qwen/Qwen3-0.6B \
  --dtype float16
```

### 3. Build and run the CPU runtime

```bash
cd /Users/song-ganghui/Documents/SoC/Mac/cpu
make main
./build/bin/main --manifest /Users/song-ganghui/Documents/SoC/models/cpp/qwen3-0.6b/manifest.json --prompt 'Hello, world!' --max-new-tokens 1
```

chat template 경로 확인:

```bash
cd /Users/song-ganghui/Documents/SoC/Mac/cpu
./build/bin/main --manifest /Users/song-ganghui/Documents/SoC/models/cpp/qwen3-0.6b/manifest.json --prompt 'Hello, world!' --apply-chat-template --disable-thinking --max-new-tokens 1
```

### 4. Run validated tests

core test run only:

```bash
cd /Users/song-ganghui/Documents/SoC/Mac/cpu
make check-core
```

real-bundle golden regression:

```bash
cd /Users/song-ganghui/Documents/SoC/Mac/cpu
make integration-qwen3-goldens
```

full aggregate regression:

```bash
cd /Users/song-ganghui/Documents/SoC/Mac/cpu
make regression
```

### 5. GPU regression and sweep

Metal GPU 경로는 real bundle 기준으로 CPU first-token parity, multi-token sequence parity, `GenerationContext` parity, stage timing, Metal GPU timestamp 기반 GPU active ratio, temporary arena working-set ratio를 함께 기록한다.

```bash
cd /Users/song-ganghui/Documents/SoC
make regression
```

GPU regression만 따로 실행:

```bash
cd /Users/song-ganghui/Documents/SoC/Mac/gpu
make real-bundle-regression
```

prompt-length sweep summary 생성:

```bash
cd /Users/song-ganghui/Documents/SoC
make gpu-sweep
```

기본 report artifact:

1. `Mac/gpu/build/reports/test_real_bundle_regression_report.md`
2. `Mac/gpu/build/reports/test_real_bundle_sweep_summary.md`
3. `Mac/gpu/build/reports/sweep_cases/*.md`

bootstrap binary만 확인하려면 기존 경로도 그대로 사용할 수 있다.

```bash
cd /Users/song-ganghui/Documents/SoC/Mac/gpu
make regression
./build/bin/gpu_bootstrap
```

offline Metal Toolchain이 설치된 환경이라면 metallib도 미리 만들 수 있다.

```bash
cd /Users/song-ganghui/Documents/SoC/Mac/gpu
make build-shaders
```

## Notes

1. `Mac/cpu/Makefile`은 이제 `-MMD -MP` dependency tracking을 사용한다. 이전 `test_nn_modules` 종료 시 segfault는 stale incremental build 가능성이 가장 높았고, clean rebuild 기준으로는 재현되지 않았다.
2. Python export tooling의 CLI 옵션과 파일 산출물은 [LLM_interpreter/README.md](/Users/song-ganghui/Documents/SoC/LLM_interpreter/README.md)에 정리되어 있다.
3. 더 상세한 설계 문서는 [Mac/cpu/plan/README.md](/Users/song-ganghui/Documents/SoC/Mac/cpu/plan/README.md)부터 시작하면 된다.
4. 현재 문서 기준의 golden 값은 실제 `Qwen/Qwen3-0.6B` export bundle에서 측정한 값이다.
5. Metal GPU 경로 설계와 real-bundle 결과 문서는 [Mac/gpu/plan/README.md](/Users/song-ganghui/Documents/SoC/Mac/gpu/plan/README.md)에 정리되어 있다.
6. `Mac/gpu`는 offline Metal Toolchain이 없어도 runtime shader source compile fallback으로 regression을 수행할 수 있고, offline toolchain이 있으면 `make build-shaders`로 metallib를 미리 만들 수 있다.