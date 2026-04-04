# Phase 2: tokenization.h

## 목표
`Qwen3_5Tokenizer`를 C++로 구현하여 BPE 인코딩/디코딩을 수행한다.

## 원본
- `tokenization_qwen3_5.py` → `Qwen3_5Tokenizer(TokenizersBackend)`
- 내부적으로 HuggingFace `tokenizers` 라이브러리 사용 (Rust 기반)

## 구현 항목

### BPE 코어
- vocab (token → id) 딕셔너리 로드
- merges 파일 파싱 및 merge 순서 테이블 구축
- byte-level BPE 인코딩
- 디코딩 (id → token → string)

### Pre-tokenization
- regex 기반 split (`PRETOKENIZE_REGEX`)
- byte-level pre-tokenizer
- NFC normalization

### Special tokens
- `<|endoftext|>` (unk, eos, pad)
- `bos_token` = None
- 추가 special tokens (vision 관련은 Phase 1 범위에서 제외)

### 파일 로드
- `vocab.json` 또는 `tokenizer.json` 에서 vocab + merges 로드
- HuggingFace tokenizer 포맷 호환

## 결과물
- `models/qwen3_5/tokenization.h`

## 상태
- [x] vocab/merges 로더
- [x] BPE encode
- [x] BPE decode
- [x] pre-tokenization (regex split)
- [x] 테스트 (Python tokenizer output과 token-by-token 비교) — 23 tests passed
