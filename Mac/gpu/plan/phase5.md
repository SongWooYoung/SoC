# Phase 5: 최적화 실험 (EXP-1 ~ EXP-7)

## 목표
Phase 4에서 확정한 baseline 대비 성능을 개선한다.
각 실험은 독립적으로 진행하며, baseline 대비 성능 차이를 측정하여 채택 여부를 결정한다.

## 실험 관리
- 각 실험은 독립 브랜치 또는 컴파일 플래그로 관리
- 실험 결과는 이 문서에 기록

## 실험 목록

### EXP-1: 연산별 실행 위치
**가설**: 일부 연산을 CPU에서 실행하면 GPU↔CPU 전송 비용을 줄일 수 있다.

| 연산 | baseline 위치 | 실험 위치 | 결과 |
|------|--------------|-----------|------|
| Embedding lookup | ? | CPU | |
| Argmax / sampling | ? | CPU | |
| RMSNorm | ? | GPU | |
| ... | | | |

### EXP-2: matmul shader
**가설**: Apple MPS 프레임워크 대신 직접 tiled `.metal` shader를 작성하면 더 빠를 수 있다.
- baseline: PyTorch MPS 방식
- 실험: 직접 tiled matmul (threadgroup memory 활용)
- 비교: GFLOPS, prefill_ms, decode_ms

### EXP-3: fused kernels
**가설**: 연속 실행되는 작은 연산들을 하나의 Metal 커널로 합치면 launch overhead가 줄어든다.

후보 fusion:
- RMSNorm + residual add
- RoPE apply + Q@K matmul
- SiLU(gate_proj) * up_proj
- softmax + attention score * V

### EXP-4: 메모리 배치 전략
**가설**: unified memory를 활용한 zero-copy가 명시적 GPU upload보다 나을 수 있다.

| 전략 | 설명 |
|------|------|
| A | weight 전체를 MTLBuffer(GPU)에 상주 |
| B | layer 단위로 CPU↔GPU swap |
| C | MTLBuffer(shared mode) — unified memory zero-copy |

### EXP-5: KV cache 전략
**가설**: KV cache를 GPU에 상주시키면 decode 속도가 빨라진다.

| 전략 | 설명 |
|------|------|
| A | GPU 상주 (pre-allocate max seq len) |
| B | CPU 보관, 매 step GPU 전송 |
| C | ring buffer (고정 크기, oldest 덮어쓰기) |

### EXP-6: quantization on Metal
**가설**: q4/q8 weight를 Metal shader 내에서 dequant하면 메모리 대역폭이 줄어 throughput이 증가한다.
- baseline: fp16
- 실험: q8_0, q4_0 dequant in shader
- 비교: throughput, 품질 (perplexity 또는 output 일치도)

### EXP-7: GatedDeltaNet 커스텀 커널
**가설**: `chunk_gated_delta_rule` 전체를 단일 Metal 커널로 구현하면 여러 작은 커널의 launch overhead를 제거할 수 있다.
- baseline: PyTorch fallback 방식 (여러 연산 순차 실행)
- 실험: 전체 chunk loop을 하나의 compute shader로
- 특이점: recurrent state 업데이트가 sequential → parallelism 제한

## 결과 기록 템플릿

| 실험 | prefill_ms | decode_ms | wall_ms | tok/s | vs baseline |
|------|-----------|-----------|---------|-------|-------------|
| baseline | | | | | — |
| EXP-1 | | | | | |
| EXP-2 | | | | | |
| ... | | | | | |

## 상태
- [ ] EXP-1
- [ ] EXP-2
- [ ] EXP-3
- [ ] EXP-4
- [ ] EXP-5
- [ ] EXP-6
- [ ] EXP-7
