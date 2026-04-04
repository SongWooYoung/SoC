# Plan 0 Stage Trace Summary

- Generated at: 2026-04-05T00:04:52
- Base trace: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan_0/base_stage_trace.json
- Custom trace: /Volumes/990pro/Documents/SoC/Mac/gpu/test/optimization/plan_0/custom_stage_trace.json
- Notes: `full_attention` includes nested `rope` and `kv_cache_update`, and `linear_attention` includes nested `linear_cache_update`. Stage deltas are therefore interpreted per-stage, not summed into a single exclusive total.

## Overall Metrics

| Backend | Rows | Avg Prefill ms | Avg Decode ms/tok | Avg Wall ms | Avg Tok/s |
|--------|-----:|---------------:|------------------:|------------:|----------:|
| base | 4 | 514.607 | 115.162 | 4200.318 | 7.670 |
| custom | 4 | 1844.759 | 159.884 | 6961.741 | 5.018 |

## High-Level Read

Custom prefill is slower mainly in: mlp, linear_cache_update, lm_head.
Custom decode is slower mainly in: linear_cache_update, mlp, lm_head.
Custom linear_attention also shows a large dispatch-side anomaly: prefill dispatch 325.507 ms vs base 2.032 ms, decode dispatch 1020.052 ms vs base 43.622 ms.

## Prefill Stage Delta

| Stage | Base sync ms | Custom sync ms | Delta ms | Ratio |
|------|-------------:|---------------:|---------:|------:|
| mlp | 289.822 | 820.399 | 530.577 | 2.831x |
| linear_cache_update | 0.000 | 320.026 | 320.026 | n/a |
| lm_head | 39.308 | 341.235 | 301.927 | 8.681x |
| linear_attention | 119.438 | 216.420 | 96.982 | 1.812x |
| full_attention | 27.367 | 72.673 | 45.306 | 2.655x |
| input_embeddings | 19.799 | 45.360 | 25.561 | 2.291x |
| kv_cache_update | 9.822 | 16.765 | 6.943 | 1.707x |
| rope | 3.029 | 4.492 | 1.463 | 1.483x |
| sampler_sync(argmax/item) | 0.518 | 0.681 | 0.163 | 1.315x |
| final_norm | 0.277 | 0.306 | 0.029 | 1.105x |
| position_ids | 1.358 | 0.029 | -1.329 | 0.021x |

## Decode Stage Delta

| Stage | Base sync ms | Custom sync ms | Delta ms | Ratio |
|------|-------------:|---------------:|---------:|------:|
| linear_cache_update | 0.000 | 997.365 | 997.365 | n/a |
| mlp | 1951.633 | 2472.593 | 520.960 | 1.267x |
| lm_head | 351.627 | 434.422 | 82.795 | 1.235x |
| full_attention | 226.909 | 278.430 | 51.521 | 1.227x |
| input_embeddings | 7.213 | 30.436 | 23.223 | 4.220x |
| kv_cache_update | 103.386 | 113.889 | 10.503 | 1.102x |
| final_norm | 6.889 | 7.160 | 0.271 | 1.039x |
| sampler_sync(argmax/item) | 10.628 | 9.448 | -1.180 | 0.889x |
| position_ids | 7.717 | 0.618 | -7.099 | 0.080x |
| rope | 68.560 | 55.652 | -12.908 | 0.812x |
| linear_attention | 859.887 | 680.499 | -179.388 | 0.791x |

## Mean Stage Stats

### Base Prefill

{
  "final_norm": {
    "calls": 1.0,
    "dispatch_ms": 0.007,
    "sync_ms": 0.277
  },
  "full_attention": {
    "calls": 8.0,
    "dispatch_ms": 13.804,
    "sync_ms": 27.367
  },
  "input_embeddings": {
    "calls": 1.0,
    "dispatch_ms": 0.013,
    "sync_ms": 19.799
  },
  "kv_cache_update": {
    "calls": 8.0,
    "dispatch_ms": 0.095,
    "sync_ms": 9.822
  },
  "linear_attention": {
    "calls": 24.0,
    "dispatch_ms": 2.032,
    "sync_ms": 119.438
  },
  "linear_cache_update": {
    "calls": 48.0,
    "dispatch_ms": 0.011,
    "sync_ms": 0.0
  },
  "lm_head": {
    "calls": 1.0,
    "dispatch_ms": 0.005,
    "sync_ms": 39.308
  },
  "mlp": {
    "calls": 32.0,
    "dispatch_ms": 0.331,
    "sync_ms": 289.822
  },
  "position_ids": {
    "calls": 1.0,
    "dispatch_ms": 0.024,
    "sync_ms": 1.358
  },
  "rope": {
    "calls": 8.0,
    "dispatch_ms": 0.263,
    "sync_ms": 3.029
  },
  "sampler_sync(argmax/item)": {
    "calls": 1.0,
    "dispatch_ms": 0.004,
    "sync_ms": 0.518
  }
}

### Custom Prefill

{
  "final_norm": {
    "calls": 1.0,
    "dispatch_ms": 0.001,
    "sync_ms": 0.306
  },
  "full_attention": {
    "calls": 8.0,
    "dispatch_ms": 22.385,
    "sync_ms": 72.673
  },
  "input_embeddings": {
    "calls": 1.0,
    "dispatch_ms": 0.026,
    "sync_ms": 45.36
  },
  "kv_cache_update": {
    "calls": 8.0,
    "dispatch_ms": 0.002,
    "sync_ms": 16.765
  },
  "linear_attention": {
    "calls": 24.0,
    "dispatch_ms": 325.507,
    "sync_ms": 216.42
  },
  "linear_cache_update": {
    "calls": 48.0,
    "dispatch_ms": 0.035,
    "sync_ms": 320.026
  },
  "lm_head": {
    "calls": 1.0,
    "dispatch_ms": 0.001,
    "sync_ms": 341.235
  },
  "mlp": {
    "calls": 32.0,
    "dispatch_ms": 0.095,
    "sync_ms": 820.399
  },
  "position_ids": {
    "calls": 1.0,
    "dispatch_ms": 0.009,
    "sync_ms": 0.029
  },
  "rope": {
    "calls": 8.0,
    "dispatch_ms": 0.826,
    "sync_ms": 4.492
  },
  "sampler_sync(argmax/item)": {
    "calls": 1.0,
    "dispatch_ms": 0.007,
    "sync_ms": 0.681
  }
}

### Base Decode

{
  "final_norm": {
    "calls": 31.0,
    "dispatch_ms": 0.105,
    "sync_ms": 6.889
  },
  "full_attention": {
    "calls": 248.0,
    "dispatch_ms": 194.766,
    "sync_ms": 226.909
  },
  "input_embeddings": {
    "calls": 31.0,
    "dispatch_ms": 0.321,
    "sync_ms": 7.213
  },
  "kv_cache_update": {
    "calls": 248.0,
    "dispatch_ms": 1.984,
    "sync_ms": 103.386
  },
  "linear_attention": {
    "calls": 744.0,
    "dispatch_ms": 43.622,
    "sync_ms": 859.887
  },
  "linear_cache_update": {
    "calls": 1488.0,
    "dispatch_ms": 0.421,
    "sync_ms": 0.0
  },
  "lm_head": {
    "calls": 31.0,
    "dispatch_ms": 0.131,
    "sync_ms": 351.627
  },
  "mlp": {
    "calls": 992.0,
    "dispatch_ms": 6.689,
    "sync_ms": 1951.633
  },
  "position_ids": {
    "calls": 31.0,
    "dispatch_ms": 0.541,
    "sync_ms": 7.717
  },
  "rope": {
    "calls": 248.0,
    "dispatch_ms": 6.418,
    "sync_ms": 68.56
  },
  "sampler_sync(argmax/item)": {
    "calls": 31.0,
    "dispatch_ms": 0.069,
    "sync_ms": 10.628
  }
}

### Custom Decode

{
  "final_norm": {
    "calls": 31.0,
    "dispatch_ms": 0.016,
    "sync_ms": 7.16
  },
  "full_attention": {
    "calls": 248.0,
    "dispatch_ms": 174.752,
    "sync_ms": 278.43
  },
  "input_embeddings": {
    "calls": 31.0,
    "dispatch_ms": 0.189,
    "sync_ms": 30.436
  },
  "kv_cache_update": {
    "calls": 248.0,
    "dispatch_ms": 0.139,
    "sync_ms": 113.889
  },
  "linear_attention": {
    "calls": 744.0,
    "dispatch_ms": 1020.052,
    "sync_ms": 680.499
  },
  "linear_cache_update": {
    "calls": 1488.0,
    "dispatch_ms": 0.518,
    "sync_ms": 997.365
  },
  "lm_head": {
    "calls": 31.0,
    "dispatch_ms": 0.027,
    "sync_ms": 434.422
  },
  "mlp": {
    "calls": 992.0,
    "dispatch_ms": 2.131,
    "sync_ms": 2472.593
  },
  "position_ids": {
    "calls": 31.0,
    "dispatch_ms": 0.024,
    "sync_ms": 0.618
  },
  "rope": {
    "calls": 248.0,
    "dispatch_ms": 0.545,
    "sync_ms": 55.652
  },
  "sampler_sync(argmax/item)": {
    "calls": 31.0,
    "dispatch_ms": 0.062,
    "sync_ms": 9.448
  }
}
