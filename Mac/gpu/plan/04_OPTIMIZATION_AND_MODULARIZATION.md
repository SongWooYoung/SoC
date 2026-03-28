# Optimization And Modularization Plan

## 1. Why Modularization Matters More On GPU

CPU에서는 느려도 correctness path를 유지한 뒤 kernel을 갈아끼울 수 있다. GPU에서는 다음 이유로 구조가 더 중요하다.

1. kernel launch / command buffer 비용이 크다.
2. host-device sync가 잘못 들어가면 전체 이득이 사라진다.
3. KV cache와 activation lifetime이 메모리 사용량을 바로 터뜨린다.
4. pipeline state, buffer layout, threadgroup size가 서로 얽혀 있다.

즉, `최적화와 기능을 분리하는 모듈화`가 아니라, `최적화를 가능하게 만드는 모듈화`가 필요하다.

## 2. Optimization Axes

### 2.1 Weight Layout Prepack

목표:

1. matmul memory access locality 개선
2. decode의 small-batch GEMM path 최적화
3. pipeline specialization 단순화

원칙:

1. export format은 portable 하게 유지
2. GPU startup 시 prepack copy 허용
3. prepack metadata는 runtime private contract로 둔다

### 2.2 Kernel Fusion

우선 후보:

1. residual add + RMSNorm
2. q/k/v projection 후 reshape 경로 일부
3. SiLU + gate multiply
4. attention softmax/value reduce path 일부

원칙:

1. baseline unfused path를 먼저 둔다.
2. fused path는 op layer에서 선택한다.
3. fused path가 module public API를 바꾸면 안 된다.

### 2.3 Prefill vs Decode Split

prefill:

1. throughput 우선
2. 긴 sequence parallelism 활용
3. larger tile, more occupancy 방향

decode:

1. latency 우선
2. sequence length 증가에 따른 KV read 비용이 핵심
3. single-token specialized kernels 필요

이 둘을 같은 kernel family로 강제하면 둘 다 손해 본다.

### 2.4 KV Cache Layout

핵심 질문:

1. layer-major vs token-major
2. packed head layout
3. paged cache 필요 여부
4. append/update 비용과 read 비용의 균형

초기 원칙:

1. public API는 canonical logical layout
2. private implementation은 packed 가능
3. prefill append와 decode read 모두 측정해 결정

### 2.5 Command Scheduling

최적화 포인트:

1. op별 command buffer를 쪼개지 않기
2. readback이나 CPU sampling이 sync point가 되지 않게 하기
3. 필요하면 logits 일부만 CPU로 읽기
4. pipeline state와 temporary buffer를 재사용하기

### 2.6 Memory Residency

규칙:

1. weight는 가능하면 `private`
2. upload staging은 `shared`
3. logits/sampling 경로는 최소 readback
4. debug readback은 명시적 slow path

## 3. Modularization Rules

### 3.1 What Must Stay Stable

다음 경계는 최적화가 들어와도 유지해야 한다.

1. asset loader
2. tensor descriptor
3. module forward interface
4. runtime prompt/generation interface

### 3.2 What May Be Swapped Internally

다음은 private optimization으로 바뀔 수 있다.

1. weight packed layout
2. kernel tile size
3. fusion 여부
4. command batch 경계
5. KV cache packed layout

### 3.3 Anti-Patterns

금지:

1. module 내부에서 직접 shader 이름 문자열을 고정하는 것
2. runtime layer가 buffer offset 계산을 직접 하는 것
3. debug readback이 normal path에 섞이는 것
4. prefill/decode policy가 분리되지 않는 것

## 4. Phase 1 Performance Strategy

1. correctness-first baseline kernels 작성
2. CPU reference와 numerical compare 경로 확보
3. matmul, RMSNorm, RoPE, attention의 실제 병목 측정
4. decode latency에 직접 영향 주는 fused path부터 최적화

우선순위:

1. small-batch decode matmul
2. KV cache read path
3. attention softmax/value reduce
4. residual + norm fusion

## 5. Success Condition For Optimization Work

최적화 작업이 성공한 것으로 보려면 다음을 만족해야 한다.

1. public API가 더럽혀지지 않는다.
2. debug/correctness path가 유지된다.
3. CPU reference와 error budget 비교가 가능하다.
4. pipeline cache / allocator / scheduler가 재사용 가능하다.

## 6. Initial Apple GPU Performance Hypotheses

Phase 0과 Phase 1 초기에 바로 검증해야 할 가설은 아래와 같다.

1. Apple Silicon에서는 discrete GPU 스타일의 사고보다 unified memory cost model이 더 중요하다.
	의미: host staging copy 자체보다 command scheduling과 resource hazard가 더 큰 병목일 수 있다.

2. decode latency에서는 large GEMM peak FLOPS보다 small-batch dispatch overhead와 KV read pattern이 더 지배적이다.
	의미: matmul kernel 하나만 빠르게 만드는 것으로는 충분하지 않을 수 있다.

3. `private` weight buffer는 장기적으로 유리하되, Phase 0 계측 없이 무조건 고정하면 startup cost를 과소평가할 수 있다.
	의미: upload-once runtime인지, repeated load/dev loop인지에 따라 정책이 달라질 수 있다.

4. threadgroup memory를 많이 쓰는 fused attention보다, first baseline에서는 occupancy가 높은 단순 kernel family가 더 빠를 가능성이 높다.
	의미: Apple GPU에서 threadgroup memory pressure와 register pressure를 먼저 측정해야 한다.

5. logits 전체 readback보다 top-k reduced readback이 decode tail latency를 더 잘 줄일 가능성이 높다.
	의미: sampling을 언제 GPU로 올릴지 판단하려면 readback byte size와 sync cost를 같이 재야 한다.

6. function constant specialization은 decode hot path에만 제한적으로 써야 한다.
	의미: head dim, causal flag, prefill/decode mode 정도만 early specialization 후보로 보고 pipeline explosion을 피해야 한다.