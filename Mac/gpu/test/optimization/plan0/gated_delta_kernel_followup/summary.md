# Gated Delta Kernel Follow-Up

- Generated at: 2026-04-05T00:36:33
- Base trace: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan_0/operator_hook_test/base_stage_trace.json
- Baseline custom trace: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan_0/operator_hook_test/custom_stage_trace.json
- Compiled-ops custom trace: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/gated_delta_kernel_followup/compiled_ops_stage_trace.json
- Custom gated delta microbench: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/gated_delta_kernel_followup/custom_gated_delta_microbench.json
- Upstream gated delta kernel bench: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan0/gated_delta_kernel_followup/upstream_gated_delta_kernel_bench.json

## Experiment 1: Full Model With Compiled Gated Delta Ops

| Backend | Avg Prefill ms | Avg Decode ms/tok | Avg Wall ms | Avg Tok/s |
|--------|---------------:|------------------:|------------:|----------:|
| base | 514.607 | 115.162 | 4200.318 | 7.670 |
| custom ops | 1844.759 | 159.884 | 6961.741 | 5.018 |
| custom compiled_ops | 466.022 | 151.419 | 5311.723 | 6.031 |

- Prefill linear_attention dispatch: ops 325.507 ms -> compiled_ops 162.022 ms (-50.2%).
- Decode linear_attention dispatch: ops 1020.052 ms -> compiled_ops 957.233 ms (-6.2%).
- Prefill linear_attention sync: ops 216.420 ms -> compiled_ops 57.751 ms (-73.3%).
- Decode linear_attention sync: ops 680.499 ms -> compiled_ops 616.324 ms (-9.4%).

## Experiment 2: Custom C++ Gated Delta Microbenchmark

| Shape | Ops sync ms | Compiled ops sync ms | Delta |
|------|------------:|---------------------:|------:|
| prefill | 14.664 | 9.619 | -34.4% |
| decode | 0.474 | 0.405 | -14.6% |

## Experiment 3: Upstream Python Gated Delta With And Without Kernel

| Shape | Upstream ops sync ms | Upstream kernel sync ms | Delta |
|------|---------------------:|------------------------:|------:|
| prefill | 9.795 | 0.729 | -92.6% |
| decode | 0.473 | 0.333 | -29.6% |

## Interpretation

Custom C++ `compiled_ops` directly tests the part we were missing from upstream `@mx.compile` usage. If this reduces isolated gated-delta cost but does not close the full-model gap, then compile alone is only a partial explanation.
The upstream Python benchmark separates `use_kernel=False` and `use_kernel=True`. If the kernel path is much faster than upstream ops, that means the missing Metal-kernel recurrent update is the stronger root-cause candidate than cache-tail handling.
The upstream kernel delta is larger than the custom compiled-ops delta on prefill-sized shapes. That points to the absent kernelized recurrent update as the bigger remaining gap after adding compile-level optimization.
Full-model trace improvement confirms that `gated_delta` implementation details do affect end-to-end runtime, not just the isolated microbenchmark.
