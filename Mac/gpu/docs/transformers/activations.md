# activations.py 분석

## 원본 위치
`transformers/activations.py`

## import 사용처
```python
from ...activations import ACT2FN
```

## ACT2FN 매핑 (완전)

```python
ACT2CLS = {
    "gelu":          GELUActivation,
    "gelu_10":       (ClippedGELUActivation, {"min": -10, "max": 10}),
    "gelu_fast":     FastGELUActivation,
    "gelu_new":      NewGELUActivation,
    "gelu_python":   (GELUActivation, {"use_gelu_python": True}),
    "gelu_pytorch_tanh": GELUTanh,
    "gelu_python_tanh":  (GELUTanh, {"use_gelu_tanh_python": True}),
    "gelu_accurate":  AccurateGELUActivation,
    "hardswish":      nn.Hardswish,
    "laplace":        LaplaceActivation,
    "leaky_relu":     nn.LeakyReLU,
    "linear":         LinearActivation,
    "mish":           MishActivation,
    "quick_gelu":     QuickGELUActivation,
    "relu":           nn.ReLU,
    "relu2":          ReLUSquaredActivation,
    "relu6":          nn.ReLU6,
    "sigmoid":        nn.Sigmoid,
    "silu":           SiLUActivation,      # = nn.functional.silu
    "swish":          nn.SiLU,
    "tanh":           nn.Tanh,
    "prelu":          nn.PReLU,
    "xielu":          XIELUActivation,
}
ACT2FN = ClassInstantier(ACT2CLS)
```

`ClassInstantier`는 `OrderedDict` 서브클래스로, `__getitem__` 시 자동 인스턴스화.

## Qwen3.5에서 사용하는 활성화 함수

### Text model
- `config.hidden_act = "silu"` (Qwen 시리즈 기본값)
- 사용 위치: `Qwen3_5MLP`, `Qwen3_5GatedDeltaNet`

### Vision model
- 보통 `"quick_gelu"` 또는 `"gelu"` (Qwen VL 시리즈 기본)
- config에서 확인 필요

### SiLU 구현
```python
silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
```

### QuickGELU 구현 (Vision에서 사용 가능)
```python
quick_gelu(x) = x * sigmoid(1.702 * x)
```

## C++ Metal 구현

### SiLU (Metal shader)
```metal
float silu(float x) {
    return x / (1.0f + exp(-x));
    // 또는: return x * (1.0f / (1.0f + exp(-x)));
}
```
- `metal::fast::exp()`로 성능 최적화 가능
- half precision: `half silu(half x)` 동일 패턴

### QuickGELU (필요 시)
```metal
float quick_gelu(float x) {
    return x * (1.0f / (1.0f + exp(-1.702f * x)));
}
```

### GELU (필요 시)
```metal
float gelu(float x) {
    return 0.5f * x * (1.0f + erf(x * M_SQRT1_2_F));
    // 또는 tanh 근사: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
}
```

### 구현하지 않을 것
- ClippedGELU, Mish, Laplace, ReLUSquared, xIELU 등 미사용 활성화
- `ClassInstantier` 레지스트리 패턴 (직접 함수 호출로 대체)
