# LLM Interpreter Export Guide

`LLM_interpreter`는 Hugging Face snapshot을 `Mac/cpu` runtime이 바로 읽을 수 있는 C++ bundle로 변환하는 Python tooling이다.

현재 문서 기준의 주 진입점은 두 개다.

1. `model_downloader.py`
   snapshot download + optional export를 한 번에 수행
2. `convert_py_to_cpp.py`
   이미 내려받은 snapshot을 C++ bundle로 다시 export

## Environment

workspace 가상환경을 기준으로 실행한다.

설치:

```bash
cd /Users/song-ganghui/Documents/SoC/LLM_interpreter
/Users/song-ganghui/Documents/SoC/.venv/bin/python -m pip install --upgrade pip
/Users/song-ganghui/Documents/SoC/.venv/bin/python -m pip install -r requirements.txt
```

현재 `requirements.txt`는 download/export에 필요한 최소 패키지 네 개를 담고 있다.

1. `huggingface_hub`
   snapshot download
2. `safetensors`
   shard read
3. `transformers`
   config/tokenizer loading
4. `torch`
   tensor dtype conversion and CPU materialization

```bash
cd /Users/song-ganghui/Documents/SoC/LLM_interpreter
/Users/song-ganghui/Documents/SoC/.venv/bin/python -m unittest test_convert_py_to_cpp.py test_export_test_bundle.py
```

export나 download 전에 필요한 패키지가 없으면 `LLM_interpreter/requirements.txt` 기준으로 설치되어 있어야 한다.

Apple Silicon 환경에서 `torch` wheel 설치 시간이 길 수 있으므로, CI에서는 Python environment setup 단계와 export/test 단계를 분리하는 편이 낫다.

## Command 1: model_downloader.py

snapshot을 내려받고, 기본적으로 곧바로 C++ export까지 수행한다.

```bash
cd /Users/song-ganghui/Documents/SoC/LLM_interpreter
/Users/song-ganghui/Documents/SoC/.venv/bin/python model_downloader.py \
  --model-id Qwen/Qwen3-0.6B \
  --download-dir /Users/song-ganghui/Documents/SoC/models/raw/qwen3-0.6b \
  --export-dir /Users/song-ganghui/Documents/SoC/models/cpp/qwen3-0.6b \
  --dtype float16
```

### Options

1. `--model-id`
   Hugging Face repo id. 기본값은 `Qwen/Qwen3-0.6B`.
2. `--download-dir`
   raw snapshot 저장 경로. 기본값은 `models/raw/qwen3-0.6b`.
3. `--revision`
   branch, tag, commit hash를 고정할 때 사용.
4. `--hf-token`
   명시적 HF token. 없으면 `HUGGINGFACE_HUB_TOKEN` 또는 `HF_TOKEN` 환경 변수를 사용.
5. `--export-dir`
   exported C++ bundle 출력 경로. 기본값은 `models/cpp/qwen3-0.6b`.
6. `--skip-export`
   snapshot만 받고 export는 생략.
7. `--dtype`
   export 시 floating weight dtype. 허용값은 `native`, `float32`, `float16`.

### Output

이 명령은 보통 아래 산출물을 남긴다.

1. `download_metadata.json`
   실제 snapshot download 메타데이터
2. raw Hugging Face files
   `config.json`, `tokenizer.json`, `*.safetensors` 등
3. export bundle
   `manifest.json`, `weights/*.bin`, `tokenizer/tokenizer_runtime.json` 등

## Command 2: convert_py_to_cpp.py

이미 있는 snapshot에서 export만 다시 수행한다.

```bash
cd /Users/song-ganghui/Documents/SoC/LLM_interpreter
/Users/song-ganghui/Documents/SoC/.venv/bin/python convert_py_to_cpp.py \
  --model-dir /Users/song-ganghui/Documents/SoC/models/raw/qwen3-0.6b \
  --output-dir /Users/song-ganghui/Documents/SoC/models/cpp/qwen3-0.6b \
  --model-id Qwen/Qwen3-0.6B \
  --dtype float16
```

### Options

1. `--model-dir`
   downloaded Hugging Face snapshot directory. 필수.
2. `--output-dir`
   exported bundle 출력 경로. 필수.
3. `--model-id`
   manifest에 기록할 optional model id.
4. `--dtype`
   floating tensor export dtype. 허용값은 `native`, `float32`, `float16`.

### Export Contract

현재 exporter는 아래 정보를 C++ runtime이 읽는 형태로 고정한다.

1. `manifest.json`
   tensor inventory, config path, tokenizer runtime path, generation config 경로
2. `weights/*.bin`
   tensor별 raw binary payload
3. `tokenizer/tokenizer_runtime.json`
   vocab, added tokens, chat template, BPE metadata, pre-tokenizer metadata, decoder metadata, byte-to-unicode 매핑
4. copied tokenizer assets
   `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `generation_config.json`, `config.json` 등

## Validated Qwen3 Flow

실제 검증된 기준 경로는 아래다.

1. `model_downloader.py --model-id Qwen/Qwen3-0.6B --dtype float16`
2. `Mac/cpu`에서 `make check-goldens`
3. 필요 시 전체는 `cd /Users/song-ganghui/Documents/SoC/Mac/cpu && make regression`

## Notes

1. tokenizer fidelity를 위해 exporter는 `tokenizer.json`의 backend state를 읽어 ByteLevel pre-tokenizer, decoder, byte-to-unicode mapping까지 runtime manifest에 포함한다.
2. tokenizer `vocab_size`는 단순 HF property가 아니라 실제 max token id 범위를 기준으로 계산된다.
3. runtime 쪽 설계와 regression 명령은 workspace 루트의 [README.md](/Users/song-ganghui/Documents/SoC/README.md)와 [Mac/cpu/plan/README.md](/Users/song-ganghui/Documents/SoC/Mac/cpu/plan/README.md)를 함께 보면 된다.