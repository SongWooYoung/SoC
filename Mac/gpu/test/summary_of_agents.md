Codex 5.4 - llama.cpp & SoC

핵심부터 말하면, **`llama.cpp`의 Metal 경로는 “ggml backend registry + device abstraction + layer offload” 기반**이고, **당신의 `SongWooYoung/SoC`의 `Mac/gpu`는 “직접 설계한 Metal runtime + 명시적 버퍼/텐서/커맨드 스트림” 기반**입니다.  
즉 `llama.cpp`는 **범용 백엔드 프레임워크 위에 Metal을 꽂아 모델 레이어를 GPU에 분산/오프로딩**하고, 당신의 repo는 **모델 실행 전체를 Metal-native runtime으로 직접 조립**하는 쪽에 더 가깝습니다.

또 하나 먼저 말하면, `llama.cpp` 쪽은 `AGENTS.md`를 확인했고, 그 지침에 맞게 **코드를 대신 작성하지 않고 구조 분석과 개선 방향 제안 중심**으로 답하겠습니다.  
그리고 코드 검색 결과는 도구 제한 때문에 **일부만 포함될 수 있습니다**. 더 보려면 GitHub 코드 검색 UI에서 확인하세요:  
- `llama.cpp` 검색: `https://github.com/search?q=repo%3Aggml-org%2Fllama.cpp+metal&type=code`  
- `SoC` 검색: `https://github.com/search?q=repo%3ASongWooYoung%2FSoC+path%3AMac%2Fgpu&type=code`

---

## 1. `llama.cpp`에서 Metal GPU에 모델을 “어떻게 올리는지”

요약하면 흐름은 이렇습니다.

1. **Metal backend/device를 registry에 등록**
2. `llama_model_load_from_file_impl()`가 **사용 가능한 GPU device 목록을 수집**
3. `n_gpu_layers`, `devices`, `main_gpu` 같은 파라미터로 **어떤 레이어를 GPU에 둘지 결정**
4. 텐서가 할당될 buffer type/device를 선택해 **일부 또는 전체 weight를 Metal 쪽 메모리에 배치**
5. 실행 시 Metal backend가 **tensor async set/get, graph compute, event sync**를 담당

### 1-1. Metal backend 자체 생성

`ggml_backend_metal_init()`는 registry에서 첫 Metal device를 가져오고, 그 device로 Metal context를 생성한 뒤 backend 객체를 구성합니다.

```c++ name=ggml/src/ggml-metal/ggml-metal.cpp url=https://github.com/ggml-org/llama.cpp/blob/88458164c77509d2022e45f71aaf97040667abe2/ggml/src/ggml-metal/ggml-metal.cpp#L583-L605
ggml_backend_t ggml_backend_metal_init(void) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_metal_reg(), 0);
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    if (ctx == NULL) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return NULL;
    }

    ggml_backend_t backend = (ggml_backend_t) malloc(sizeof(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    return backend;
}
```

여기서 중요한 점은 `llama.cpp`가 Metal을 그냥 직접 부르는 게 아니라, **ggml backend interface 뒤에 감춘다**는 점입니다.

### 1-2. Metal device registry

Metal backend는 registry를 통해 여러 device를 노출할 수 있게 되어 있습니다. `GGML_METAL_DEVICES` 환경변수도 읽습니다.

```c++ name=ggml/src/ggml-metal/ggml-metal.cpp url=https://github.com/ggml-org/llama.cpp/blob/88458164c77509d2022e45f71aaf97040667abe2/ggml/src/ggml-metal/ggml-metal.cpp#L900-L929
ggml_backend_reg_t ggml_backend_metal_reg(void) {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);

        const char * env = getenv("GGML_METAL_DEVICES");
        if (env) {
            g_devices = atoi(env);
        }

        static std::vector<ggml_backend_device_ptr> devs;

        if (!initialized) {
            static ggml_backend_metal_reg_ptr reg_ctx(ggml_backend_metal_reg_init());

            for (int i = 0; i < g_devices; ++i) {
                auto * dev = ggml_backend_metal_device_init(&reg, i);
                devs.emplace_back(dev);

                reg_ctx->devices.push_back(dev);
            }

            reg = {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_metal_reg_i,
                /* .context     = */ reg_ctx.get(),
            };
        }
```

즉 `llama.cpp`에서 Metal은 **“특정 앱 전용 GPU runtime”이 아니라 ggml의 pluggable GPU backend”**입니다.

### 1-3. 모델 로딩 시 GPU device 수집

`llama_model_load_from_file_impl()`는 전체 backend device를 순회하면서 GPU들을 수집합니다.

```c++ name=src/llama.cpp url=https://github.com/ggml-org/llama.cpp/blob/88458164c77509d2022e45f71aaf97040667abe2/src/llama.cpp#L949-L962
for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
    ggml_backend_dev_t dev = ggml_backend_dev_get(i);
    switch (ggml_backend_dev_type(dev)) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:
        case GGML_BACKEND_DEVICE_TYPE_ACCEL:
            // skip CPU backends since they are handled separately
            break;

        case GGML_BACKEND_DEVICE_TYPE_GPU: {
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            if (ggml_backend_reg_name(reg) == std::string("RPC")) {
                rpc_servers.push_back(dev);
            } else {
                // check if there is already a GPU with the same device id
```

그리고 선택된 device 정보를 로그�� 남깁니다.

```c++ name=src/llama.cpp url=https://github.com/ggml-org/llama.cpp/blob/88458164c77509d2022e45f71aaf97040667abe2/src/llama.cpp#L1012-L1028
for (auto * dev : model->devices) {
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    LLAMA_LOG_INFO("%s: using device %s (%s) (%s) - %zu MiB free\n", __func__,
            ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
            props.device_id ? props.device_id : "unknown id",
            props.memory_free/1024/1024);
}
```

### 1-4. “모델을 GPU에 올린다”의 실제 의미

`llama.cpp`에서 “모델을 Metal GPU에 올린다”는 보통 **모든 ���서를 통째로 Metal-only 메모리로 밀어넣는다**기보다,  
**레이어별/버퍼타입별로 GPU buffer를 선택해서 offload**하는 뜻에 가깝습니다.

이를 뒷받침하는 구조:

- `llama_model`이 `devices` 목록을 들고 있음  
- `n_gpu_layers()`, `dev_layer(int il)`, `select_buft(int il)` 같은 API가 있음

```c++ name=src/llama-model.h url=https://github.com/ggml-org/llama.cpp/blob/88458164c77509d2022e45f71aaf97040667abe2/src/llama-model.h#L524-L605
std::vector<ggml_backend_dev_t> devices;

uint32_t n_gpu_layers() const;
llama_split_mode split_mode() const;

ggml_backend_dev_t dev_layer(int il) const;
ggml_backend_dev_t dev_output() const;

ggml_backend_buffer_type_t select_buft(int il) const;
```

즉 설계상 **레이어마다 어느 backend buffer에 놓을지 고르는 구조**입니다. `-ngl` / `n_gpu_layers`가 바로 이 오프로딩 깊이를 결정하는 대표 파라미터입니다. 예제 코드도 모델 파라미터에 `n_gpu_layers`와 `devices`를 넘깁니다.

```c++ name=examples/diffusion/diffusion-cli.cpp url=https://github.com/ggml-org/llama.cpp/blob/88458164c77509d2022e45f71aaf97040667abe2/examples/diffusion/diffusion-cli.cpp#L542-L570
llama_model_params model_params = llama_model_default_params();
model_params.n_gpu_layers       = params.n_gpu_layers;
model_params.devices            = params.devices.data();
model_params.use_mmap           = params.use_mmap;
model_params.use_direct_io      = params.use_direct_io;
model_params.use_mlock          = params.use_mlock;
model_params.check_tensors      = params.check_tensors;

llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);
```

### 1-5. Metal에서 텐서 업로드/다운로드

Metal backend interface는 tensor async set/get와 graph compute를 제공합니다.

```c++ name=ggml/src/ggml-metal/ggml-metal.cpp url=https://github.com/ggml-org/llama.cpp/blob/88458164c77509d2022e45f71aaf97040667abe2/ggml/src/ggml-metal/ggml-metal.cpp#L553-L573
static ggml_backend_i ggml_backend_metal_i = {
    /* .get_name                = */ ggml_backend_metal_name,
    /* .free                    = */ ggml_backend_metal_free,
    /* .set_tensor_async        = */ ggml_backend_metal_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_metal_get_tensor_async,
    /* .cpy_tensor_async        = */ ggml_backend_metal_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_metal_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_metal_graph_compute,
    /* .event_record            = */ ggml_backend_metal_event_record,
```

set/get는 내부 Metal context로 위임됩니다.

```c++ name=ggml/src/ggml-metal/ggml-metal.cpp url=https://github.com/ggml-org/llama.cpp/blob/88458164c77509d2022e45f71aaf97040667abe2/ggml/src/ggml-metal/ggml-metal.cpp#L494-L504
static void ggml_backend_metal_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_tensor_async(ctx, tensor, data, offset, size);
}
```

실제 ggml 공통층에서도 `ggml_backend_tensor_set[_async]()`를 통해 buffer interface를 타게 됩니다.

```c++ name=ggml/src/ggml-backend.cpp url=https://github.com/ggml-org/llama.cpp/blob/88458164c77509d2022e45f71aaf97040667abe2/ggml/src/ggml-backend.cpp#L231-L311
void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ...
    if (backend->iface.set_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } else {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}
```

### 1-6. Metal은 host-visible 경로도 활용

`llama.cpp`에서 Metal은 CPU와 함께 **host-visible 특성**을 적극 활용하는 모습이 보입니다. 예를 들어 `tools/mtmd/clip.cpp`는 **CPU와 Metal backend는 텐서로 직접 읽어들일 수 있다**고 명시합니다.

```c++ name=tools/mtmd/clip.cpp url=https://github.com/ggml-org/llama.cpp/blob/88458164c77509d2022e45f71aaf97040667abe2/tools/mtmd/clip.cpp#L2061-L2075
if (ggml_backend_buft_is_host(buft)) {
    // for the CPU and Metal backend, we can read directly into the tensor
    fin.read(reinterpret_cast<char *>(cur->data), num_bytes);
} else {
    // read into a temporary buffer first, then copy to device memory
    read_buf.resize(num_bytes);
    fin.read(reinterpret_cast<char *>(read_buf.data()), num_bytes);
    ggml_backend_tensor_set(cur, read_buf.data(), 0, num_bytes);
}
```

이건 Apple Silicon의 unified memory 성격을 잘 활용한 흔적입니다.  
즉 `llama.cpp`는 Metal에서 **“host staging → device copy”를 무조건 강제하지 않고**, 가능한 경우 **직접 매핑된 영역에 읽기**를 허용합니다.

---

## 2. `SongWooYoung/SoC`의 `Mac/gpu` 구현은 어떤 방식인가

당신의 repo는 `llama.cpp`와 달리 **ggml 같은 범용 그래프 런타임 위에 얹은 backend**가 아닙니다.  
오히려 **Metal runtime을 직접 관리하는 독립 추론 엔진**에 가깝습니다.

### 2-1. 설계 방향 자체가 direct Metal

문서에 아주 명확히 적혀 있습니다.

```markdown name=Mac/gpu/plan/07_IMPLEMENTATION_DECISIONS.md url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/plan/07_IMPLEMENTATION_DECISIONS.md#L5-L15
1. `Mac/gpu`는 Metal direct compute runtime으로 설계한다.
   이유: user requirement가 명시적으로 Metal 기반이며, GPU plan의 핵심 가치도 low-level control에 있다.

3. MPSGraph는 Phase 1의 기본 경로가 아니다.
   이유: 구조를 직접 제어하기 어렵고 backendless/low-level runtime 목표와 맞지 않는다.
```

즉 `SoC`는 처음부터 **backend abstraction보다 low-level control**을 우선합니다.

### 2-2. MetalContext 중심 런타임

`MetalContext`는 device/library/queue, profiling, command buffer finalize/deferred commit까지 직접 노출합니다.

```c++ name=Mac/gpu/include/metal/metal_context.h url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/include/metal/metal_context.h#L1-L73
class MetalContext {
public:
    static std::unique_ptr<MetalContext> CreateDefault(const std::string& metallib_path,
                                                       const std::string& shader_source_path,
                                                       std::string* error_message);

    ~MetalContext();

    const MetalDeviceInfo& GetDeviceInfo() const;
    const void* GetNativeDevice() const;
    const void* GetNativeCommandQueue() const;
    const void* GetNativeLibrary() const;
    void ResetProfiling() const;
    MetalProfilingSnapshot GetProfilingSnapshot() const;
    bool FinalizeCommandBuffer(const void* command_buffer_handle,
                               const std::string& error_prefix,
                               const char* profile_label,
                               std::size_t encoder_count,
                               std::string* error_message) const;
    bool CommitCommandBufferDeferred(const void* command_buffer_handle,
                                     const std::string& error_prefix,
                                     const char* profile_label,
                                     std::size_t encoder_count,
                                     std::string* error_message) const;
    bool DrainPendingCommandBuffers(std::string* error_message) const;
```

이건 `llama.cpp`보다 훨씬 더 **앱 레벨에서 GPU 실행을 명시적으로 제어**하는 구조입니다.

### 2-3. 버퍼 abstraction이 명시적

`MetalBuffer`는 shared/private/private-initialized를 구분합니다.

```c++ name=Mac/gpu/include/buffer/metal_buffer.h url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/include/buffer/metal_buffer.h#L1-L46
class MetalBuffer {
public:
    static std::shared_ptr<MetalBuffer> CreateShared(const MetalContext& context,
                                                     std::size_t size_bytes,
                                                     const std::string& label,
                                                     std::string* error_message);
    static std::shared_ptr<MetalBuffer> CreatePrivate(const MetalContext& context,
                                                      std::size_t size_bytes,
                                                      const std::string& label,
                                                      std::string* error_message);
    static std::shared_ptr<MetalBuffer> CreatePrivateInitialized(const MetalContext& context,
                                                                 const void* source,
                                                                 std::size_t size_bytes,
                                                                 const std::string& label,
                                                                 std::string* error_message);
```

이 구조는 장점이 큽니다. `llama.cpp`보다 **storage mode 전략을 더 세밀하게 설계할 여지**가 있습니다.

### 2-4. DeviceTensor + BufferArena

당신의 코드는 텐서를 `DeviceTensor(buffer, offset, desc)`로 다루고, 임시 메모리는 arena로 재사용합니다.

```c++ name=Mac/gpu/src/tensor/device_tensor.cpp url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/src/tensor/device_tensor.cpp#L1-L24
DeviceTensor::DeviceTensor(std::shared_ptr<MetalBuffer> buffer, std::size_t byte_offset, TensorDesc desc)
    : buffer_(std::move(buffer)), byte_offset_(byte_offset), desc_(std::move(desc)) {}
```

```c++ name=Mac/gpu/include/buffer/buffer_arena.h url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/include/buffer/buffer_arena.h#L1-L82
class BufferArena {
public:
    static std::unique_ptr<BufferArena> CreateShared(const MetalContext& context,
                                                     std::size_t capacity_bytes,
                                                     const std::string& label,
                                                     std::string* error_message);
...
    bool Allocate(std::size_t size_bytes,
                  std::size_t alignment_bytes,
                  BufferArenaSlice* slice,
                  std::string* error_message);
```

이건 `llama.cpp`의 일반화된 ggml tensor/buffer보다 더 **추론 엔진 전용 메모리 계획**을 하기 좋은 형태입니다.

### 2-5. CommandStream으로 command buffer batching

당신 repo는 `CommandStream`으로 여러 op dispatch를 하나의 command buffer에 모을 수 있게 했습니다.

```c++ name=Mac/gpu/include/metal/command_stream.h url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/include/metal/command_stream.h#L1-L81
/// CommandStream allows batching multiple GPU compute dispatches into a single
/// MTLCommandBuffer. Instead of each Op creating, committing, and waiting on its
/// own command buffer, Ops encode into the stream's shared command buffer.
/// Only when Flush() is called does the command buffer commit and wait.
```

하지만 같은 파일에 이미 중요한 경고가 있습니다.

```c++ name=Mac/gpu/include/metal/command_stream.h url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/include/metal/command_stream.h#L1-L81
/// Important:
/// On the Mac mini M4 32GB validation machine, batching an entire decode step or
/// multi-layer range into one giant command buffer has correlated with
/// WindowServer-level GPU faults. Keep this primitive behind explicit opt-in and
/// prefer bounded scopes such as a single layer.
```

즉 지금 구조는 batching capability는 있지만, **안전한 batching granularity를 아직 완전히 정복한 상태는 아님**을 의미합니다.

### 2-6. 모델 자산 로딩은 자체 manifest 기반

당신 repo는 모델 파일을 gguf backend abstraction으로 읽는 대신, manifest + tensor file 로딩을 직접 합니다.

```c++ name=Mac/gpu/include/asset/runtime_assets.h url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/include/asset/runtime_assets.h#L1-L57
class TensorFileLoader {
public:
    static bool LoadBytes(const TensorRecord& tensor_record,
                          std::vector<char>* bytes,
                          std::string* error_message);
    static DeviceTensor LoadDeviceTensor(const MetalContext& context,
                                         const TensorRecord& tensor_record,
                                         std::string* error_message);
};
```

즉 `SoC`는 **모델 포맷 해석부터 디바이스 텐서 배치까지 앱이 직접 관리**하는 쪽입니다.

---

## 3. 두 구현의 가장 큰 차이

## A. 추상화 레벨 차이

### `llama.cpp`
- ggml backend/device/buffer/tensor/graph abstraction 위에서 Metal 사용
- Metal은 여러 backend 중 하나
- 모델 로딩 시 **backend-independent 방식**으로 device 선택/offload
- 목표: portability + backend 교체 가능성

### `SoC`
- Metal runtime을 직접 사용
- `MetalContext`, `MetalBuffer`, `DeviceTensor`, `CommandStream`이 앱 핵심
- op 단위가 Metal compute 파이프라인과 거의 1:1 대응
- 목표: Apple GPU에 맞춘 저수준 최적화

**정리:**  
`llama.cpp`는 **범용 엔진 속 Metal backend**,  
`SoC`는 **Metal 전용 추론 엔진**입니다.

---

## B. 모델 “올리기” 방식 차이

### `llama.cpp`
- 레이어/버퍼 단위 offload
- `n_gpu_layers`, `devices`, `main_gpu` 등으로 배치 결정
- 일부는 CPU, 일부는 GPU에 둘 수 있음
- host-visible Metal buffer 경로를 활용해 직접 읽기도 함

### `SoC`
- manifest에서 직접 `DeviceTensor`를 생성
- 모델 자산 로딩과 GPU 배치를 직접 통제
- `CreatePrivateInitialized` 같은 API로 **초기 로딩 정책**을 더 정교하게 설계 가능
- 현재 구조상 “전체 모델을 GPU resident하게 유지”하는 방향으로 더 자연스러움

**정리:**  
`llama.cpp`는 **hybrid/offload형**,  
`SoC`는 **full residency 지향 direct placement형**입니다.

---

## C. 실행 스케줄링 차이

### `llama.cpp`
- ggml graph compute 중심
- backend는 `graph_compute`, `event_record`, `event_wait` 제공
- 스케줄링/오프로딩 정책이 프레임워크 안에 있음

```c++ name=ggml/src/ggml-metal/ggml-metal.cpp url=https://github.com/ggml-org/llama.cpp/blob/88458164c77509d2022e45f71aaf97040667abe2/ggml/src/ggml-metal/ggml-metal.cpp#L527-L545
static enum ggml_status ggml_backend_metal_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    return ggml_metal_graph_compute(ctx, cgraph);
}
```

### `SoC`
- op별 Metal encode/dispatch를 직접 관리
- `CommandStream`으로 batching granularity를 앱이 결정
- profiling snapshot도 앱 레벨에서 직접 가짐

이건 강력하지만, 동시에 **스케줄링 품질을 전부 직접 책임져야 한다**는 뜻입니다.

---

## D. profiling / optimization feedback loop 차이

`SoC`는 profiling entry 구조가 매우 명시적입니다.

```c++ name=Mac/gpu/include/metal/metal_context.h url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/include/metal/metal_context.h#L1-L73
struct MetalProfilingEntry {
    std::string label;
    double gpu_ms = 0.0;
    double wait_ms = 0.0;
    std::size_t command_buffer_count = 0;
    std::size_t encoder_count = 0;
};
```

그리고 벤치마크 리포트도 `gpu_ms_avg`, `wait_ms_avg`, `command_buffer_count_avg`, `encoder_count_avg`로 쪼개고 있습니다.

```python name=Mac/gpu/tools/benchmark_full_gpu_vs_pytorch.py url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/tools/benchmark_full_gpu_vs_pytorch.py#L230-L240
"| Label | gpu_ms_avg | wait_ms_avg | command_buffer_count_avg | encoder_count_avg |",
```

이건 `llama.cpp`보다 **최적화 루프를 돌리기 더 좋은 기반**입니다.  
반대로 말하면 아직 최적화가 덜 되어 있어도 **개선 여지가 훨씬 많다**는 뜻입니다.

---

## 4. 지금 `SoC`에서 보이는 GPU 사용상의 특징과 한계

코드와 문서 기준으로 보면 현재 `SoC`는 다음 성격이 강합니다.

### 강점
- direct Metal runtime
- private/shared buffer 구분 가능
- BufferArena로 임시 메모리 재사용 가능
- CommandStream batching 가능
- profiling 체계가 잘 잡혀 있음
- 실제로 CPU 대비 큰 속도 향상 측정됨

```markdown name=Mac/gpu/plan/09_TEST_RESULTS.md url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/plan/09_TEST_RESULTS.md#L47-L72
4. CPU wall: `29408.264 ms`
5. GPU context wall: `1810.261 ms`
6. CPU:GPU wall ratio: `16.245x`
```

### 아직 보이는 제약
1. **샘플링 일부는 CPU fallback**
2. command buffer batching이 공격적일 경우 안정성 이슈
3. 최적화 전략이 아직 phase-based/building stage 느낌
4. `llama.cpp`처럼 레이어 오프로딩/메모리 placement 정책이 체계적으로 일반화되어 있지는 않음

예를 들어 sampler는 여전히 CPU fallback 경로가 있습니다.

```c++ name=Mac/gpu/src/runtime/sampler.cpp url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/src/runtime/sampler.cpp#L133-L155
if (should_use_cpu_sampler) {
    return SampleFromLogitsCpuFallback(logits,
                                       row_index,
                                       top_k,
                                       token_id,
                                       top_logits,
                                       top_token_ids,
                                       error_message);
}
```

이건 decode 단계에서 **GPU→CPU readback + CPU sampling + 다시 GPU로 상태 반영** 같은 비용으로 이어질 가능성이 큽니다.

---

## 5. GPU 사용을 최적화하려면 `SoC`에서 어떤 방향으로 개선해야 하나

여기부터가 가장 중요합니다.  
내 판단으로는 우선순위가 다음과 같습니다.

---

## 우선순위 1: **decode hot path에서 CPU fallback 제거**

가장 먼저 줄여야 할 건 **GPU ↔ CPU 왕복**입니다.

지금 sampler에 CPU fallback이 존재하는데, autoregressive decode에서는 매 토큰마다 이 비용이 누적됩니다.  
`llama.cpp`가 강한 이유 중 하나는 모델 실행 경로를 backend 내부로 최대한 밀어 넣어 **불필요한 host roundtrip을 줄이기 때문**입니다.

### 개선 방향
- top-k / top-p / argmax / sampling RNG까지 가능한 한 GPU에 유지
- token 선택 결과만 최소 크기로 CPU에 가져오기
- 더 나아가 다음 step 입력 준비도 GPU에 최대한 남기기

### 왜 중요한가
prefill보다 decode는 작은 연산이 많아서, **연산 자체보다 dispatch / sync / readback 오버헤드**가 더 크게 작용합니다.  
따라서 샘플링 CPU fallback 제거는 체감 성능 개선 폭이 큽니다.

---

## 우선순위 2: **buffer storage mode 정책을 텐서 종류별로 재설계**

`SoC`는 `CreateShared`, `CreatePrivate`, `CreatePrivateInitialized`를 이미 가지고 있습니다. 이건 매우 좋은 출발점입니다.

```c++ name=Mac/gpu/include/buffer/metal_buffer.h url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/include/buffer/metal_buffer.h#L1-L46
static std::shared_ptr<MetalBuffer> CreateShared(...)
static std::shared_ptr<MetalBuffer> CreatePrivate(...)
static std::shared_ptr<MetalBuffer> CreatePrivateInitialized(...)
```

### 권장 정책
- **정적 weight**: 가능한 한 `Private` + 초기 업로드 1회
- **KV cache**: access pattern을 보고 `Private` 우선 검토
- **작고 자주 읽는 control data / metadata / small staging**: `Shared`
- **임시 workspace**: arena 기반 재사용, 가능하면 `Private` workspace와 `Shared` staging 분리

### `llama.cpp`와의 차이에서 얻을 수 있는 인사이트
`llama.cpp`는 Metal에서 host-visible 경로를 활용하지만, 그건 범용 백엔드 특성상 실용성을 택한 측면이 큽니다.  
당신 repo는 direct runtime이므로, 더 공격적으로 **weight residency 최적화**를 할 수 있습니다.

### 주의
Apple unified memory라고 해서 무조건 shared가 최고는 아닙니다.  
**GPU가 반복적으로 읽는 대형 weight**는 private가 유리한 경우가 많고, shared는 편하지만 bandwidth/캐시 측면에서 손해일 수 있습니다.

---

## 우선순위 3: **command buffer batching granularity를 자동 튜닝 가능하게 만들기**

지금 `CommandStream`은 좋은 도구지만, 문서상으로는 **너무 큰 batch가 GPU fault와 상관**이 있다고 되어 있습니다.

```c++ name=Mac/gpu/include/metal/command_stream.h url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/include/metal/command_stream.h#L1-L81
/// ... batching an entire decode step or multi-layer range into one giant command buffer ...
/// ... correlated with WindowServer-level GPU faults ...
```

### 개선 방향
고정 정책 대신 **적응형 batching**을 넣는 게 좋습니다.

예:
- prefill: layer 여러 개를 묶기
- decode: 1 layer 또는 작은 op cluster만 묶기
- attention / matmul / norm / rope 조합 중 fault-free 검증된 묶음만 허용
- profiling 결과 기반으로 “safe max encoder count”, “safe max GPU ms per command buffer” 제한

### 추천 메트릭
이미 있는 profiling에 아래 판단값을 붙이면 좋습니다.
- command buffer당 GPU time
- command buffer당 encoder 수
- flush 전후 wait 비율
- fault/retry 빈도
- 레이어별 variance

---

## 우선순위 4: **op fusion을 더 적극적으로**

`SoC`는 op abstraction이 잘 분리되어 있는데, 성능만 보면 이것이 오히려 dispatch 수 증가로 이어질 수 있습니다.

현재 구조상 `LinearOp`, `RmsNormOp`, `SoftmaxOp`, `ElementwiseMulOp`, `RopeOp` 등이 분리돼 있습니다.  
이는 유지보수에는 좋지만 decode에서는 **작은 커널 여러 번 실행**으로 이어질 수 있습니다.

### 추천 fusion 후보
- RMSNorm + scale
- Linear + bias + SiLU
- QKV projection 후 reshape/transpose 일부
- attention score scaling + masking + softmax
- residual add + activation

### 왜 필요한가
Apple GPU에서는 특히 작은 dispatch가 많으면
- encoder 생성
- pipeline state switch
- command buffer finalize/wait
- 메모리 왕복  
이 겹쳐서 손실이 큽니다.

`llama.cpp`는 ggml graph 수준에서 어느 정도 이를 흡수하지만, 당신 repo는 direct runtime이라 **이 최적화를 직접 구현할 여지가 큼**니다.

---

## 우선순위 5: **pipeline cache를 “shape-aware”하게 강화**

지금 인터페이스상 `PipelineCache`를 ops가 받는 구조는 좋습니다.

```c++ name=Mac/gpu/include/op/linear_op.h url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/include/op/linear_op.h#L1-L39
static bool Run(const MetalContext& context,
                PipelineCache* pipeline_cache,
                ...
```

하지만 실제 최적화를 위해서는 단순 함수명 캐시보다 더 나아가야 합니다.

### 제안
- `(kernel_name, dtype, tile size, seq bucket, head dim, causal/non-causal)` 단위 캐시
- 런타임 첫 실행 시 auto-select
- decode와 prefill에 서로 다른 pipeline variant 사용
- Apple GPU family별 specialization

또한 bootstrap report에 이미 `supports_simdgroup_matrix`, `thread_execution_width`, `recommended_max_working_set_size` 등을 수집하고 있습니다.

```c++ name=Mac/gpu/include/runtime/bootstrap_runtime.h url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/include/runtime/bootstrap_runtime.h#L1-L30
struct BootstrapReport {
    std::string device_name;
    bool is_apple_silicon_gpu = false;
    bool has_unified_memory = false;
    bool supports_simdgroup_matrix = false;
    std::uint64_t recommended_max_working_set_size = 0;
    std::uint32_t max_threads_per_threadgroup = 0;
    std::uint32_t thread_execution_width = 0;
```

이 정보를 실제 kernel variant 선택에 연결하면 좋습니다.

---

## 우선순위 6: **모델 로딩 시 persistent residency plan 도입**

`llama.cpp`는 모델 로딩 시 device 목록과 buffer type 선택이 구조화돼 있습니다.  
당신 repo도 이제는 단순히 tensor를 device에 올리는 수준을 넘어, **모델 전체 residency plan**을 가져가야 합니다.

### 제안할 분류
- always resident weights
- layer-local scratch
- KV cache
- per-token temporary tensors
- CPU-side metadata only
- lazy upload 가능 자산

### 좋은 결과
- 메모리 사용량 예측 가능
- OOM/fault 줄어듦
- command buffer 내 리소스 생명주기 단순화
- prefill/decode 별 메모리 정책 분리 가능

---

## 우선순위 7: **KV cache 접근 패턴 재설계**

LLM decode에서는 KV cache가 핵심 병목 중 하나입니다.  
현재 제공된 조각만으로 KV 구현을 단정할 수는 없지만, `CommandStream`에 blit encoder가 따로 있는 점을 보면 copy/update가 꽤 중요해 보입니다.

```c++ name=Mac/gpu/include/metal/command_stream.h url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/include/metal/command_stream.h#L1-L81
/// Create a new blit command encoder from the current command buffer.
/// Used for KV cache copy operations.
const void* BeginBlitEncoder();
```

### 개선 아이디어
- append-only KV layout
- layer-major / head-major / token-major 중 실제 access에 맞는 재배열
- decode 한 토큰 처리 시 필요한 범위만 coalesced read 되게 설계
- sliding window / paged KV 가능성 검토
- KV copy를 blit보다 compute gather/scatter로 바꾸는 게 유리한지 측정

`llama.cpp`와 차이는, 당신은 이 구조를 직접 바꿀 수 있다는 점입니다.

---

## 우선순위 8: **benchmark를 op-level에서 “residency-level”로 올리기**

이미 benchmark/report 체계는 좋습니다.  
이제는 다음 질문에 답할 수 있어야 합니다.

- 어느 op가 느린가?
- 어느 op가 느린 게 아니라, **어느 메모리 전이가 느린가?**
- wait_ms가 큰 이유가 GPU compute 때문인가, CPU sync 때문인가?
- command buffer count가 많아서 느린가, 각 buffer가 너무 커서 느린가?

### 추가하면 좋은 리포트
- bytes uploaded / downloaded per token
- private/shared buffer bytes by class(weight/KV/temp)
- per-token readback bytes
- per-layer command buffer 개수
- pipeline switch count
- fallback path count

---

## 6. 한 문장으로 비교하면

- **`llama.cpp` Metal**: “범용 ggml backend 체계 안에서 모델 레이어를 Metal device에 오프로딩하고 그래프를 실행하는 방식”
- **`SoC` Mac/gpu**: “Metal을 직접 제어하는 전용 런타임으로 모델 자산, 버퍼, 텐서, 커맨드 버퍼를 앱이 직접 관리하는 방식”

그래서 `SoC`의 장점은 **최적화 자유도**이고, 단점은 **스케줄링/메모리 정책을 전부 직접 잘해야 한다는 점**입니다.

---

## 7. 당신 repo에 대한 실전 개선 우선순위 제안

가장 현실적인 순서로 정리하면:

1. **CPU sampler fallback 제거**
2. **weight/KV/temp의 storage mode 정책 분리**
3. **adaptive command buffer batching**
4. **decode hot path op fusion**
5. **shape/device-aware pipeline cache**
6. **persistent residency plan 도입**
7. **KV cache layout/profile 정교화**
8. **profiling에 memory transfer/sync 원인 분석 추가**

---

## 8. 내 판단: 지금 가장 큰 차이를 만드는 3개

성능 효과 기준으로 고르면 이 3개가 제일 큽니다.

### 1) GPU sampling 완전 경로화
decode latency에 직접적.

### 2) weight/private residency 최적화
대형 모델에서 bandwidth와 안정성 개선.

### 3) bounded batching + fusion
dispatch/wait overhead를 줄이면서 fault risk도 관리 가능.


——————————————
Claude Sonnet 4.6 - llama.cpp 분석

이제 두 리포지토리의 핵심 코드를 충분히 분석했습니다. 상세한 비교 분석을 제공할 수 있습니다.

---

# llama.cpp Metal GPU 구현 vs SongWooYoung/SoC Mac/gpu 구현 분석

---

## 1. llama.cpp의 Metal GPU 모델 탑재 방식 (상세 분석)

### 1-1. 초기화 및 디바이스 설정 (`ggml-metal-device.m`)

llama.cpp의 Metal GPU 구현은 다층 레이어 구조로 되어 있습니다.

**Device 레이어 (`ggml_metal_device_init`)**
```
MTLCreateSystemDefaultDevice()
  → newCommandQueue (전역 단일 큐, 모든 백엔드가 공유)
  → 디바이스 capability 감지:
      - has_simdgroup_reduction (Apple7 이상)
      - has_simdgroup_mm (행렬 곱 전용 SIMD 유닛)
      - has_bfloat (Metal3 이상)
      - has_tensor (Metal4/M5+, MetalPerformancePrimitives API)
      - use_residency_sets (macOS 15+, 물리 메모리 고정)
      - use_shared_buffers (Unified Memory일 경우 MTLResourceStorageModeShared)
```

**Library 레이어 (`ggml_metal_library_init`)**
```
우선순위:
  1. GGML_METAL_EMBED_LIBRARY: 바이너리에 임베딩된 .metallib (배포용)
  2. bundle에서 default.metallib 탐색 (앱 번들)
  3. 실행파일 위치의 default.metallib
  4. 없으면 ggml-metal.metal 소스 → 런타임 컴파일
     (BF16, Tensor API 지원 여부를 preprocessorMacros로 조건 컴파일)
```

**모델 웨이트 업로드 (`ggml-backend.cpp` + GGML 텐서 시스템)**
- 모델의 각 레이어 텐서는 `ggml_backend_buffer` 단위로 GPU에 할당됨
- Unified Memory 환경: `MTLResourceStorageModeShared` → CPU/GPU 공유 메모리, 별도 복사 없음
- 비-Unified (eGPU 등): `MTLResourceStorageModePrivate` → `MTLBlitCommandEncoder`를 통한 비동기 복사
- **Residency Set (macOS 15+)**: 모델 웨이트를 OS 메모리 압박 상황에서도 쫓겨나지 않도록 물리 메모리에 고정, 백그라운드 스레드가 30초마다 `requestResidency` 호출

### 1-2. 그래프 실행 (`ggml_metal_graph_compute`)

```
ggml_cgraph (연산 노드 DAG)
  → 노드를 n_cb+1 개의 Command Buffer로 분할
  → 메인 스레드: 첫 n_nodes_0 개 인코딩 (경험적으로 max(64, 10% of nodes))
  → dispatch_apply(n_cb, d_queue, encode_async):
      각 스레드가 병렬로 MTLComputeCommandEncoder에 커널 디스패치
  → cmd_buf.commit() → 비동기 GPU 실행
  → ggml_metal_synchronize()에서 waitUntilCompleted
```

**Op Fusion**: `GGML_METAL_FUSION_DISABLE`가 없으면 활성화됨. 연속된 노드 (예: RMSNorm+Mul, ROPE 등)를 단일 인코더 패스로 합쳐 Command Buffer 오버헤드 감소.

**Concurrency Dispatch**: `MTLDispatchTypeConcurrent` 인코더 사용으로 독립 커널들의 GPU 내부 병렬 실행.

### 1-3. 웨이트 저장 형식

- 기본적으로 GGUF 양자화 포맷 (Q4_0, Q4_K, Q8_0 등) 을 GPU에서 직접 디��타이즈하면서 연산
- `.metal` 셰이더 내 `kernel_mul_mv_q4_0_f32` 류 커널이 양자화 웨이트를 읽으면서 float16/bfloat16으로 계산
- **웨이트는 한 번만 GPU에 올리고, 추론 내내 GPU에 상주** (Residency Set)

---

## 2. SongWooYoung/SoC Mac/gpu의 GPU 사용 방식

### 2-1. 초기화 (`metal_context.mm` - `MetalContext::CreateDefault`)

```cpp
MTLCreateSystemDefaultDevice()
  → newCommandQueue (단일 큐)
  → .metallib 파일 탐색:
      1. 지정 경로의 .metallib 파일 (빌드 결과물)
      2. 없으면 gpu_kernels.metal 소스 런타임 컴파일
  → bootstrap_copy 커널로 파이프라인 초기화
  → 디바이스 capability 감지: has_unified_memory, supports_simdgroup_matrix
```

### 2-2. 모델 웨이트 업로드 (`qwen_model_loader.cpp`)

```cpp
// float16 → float32 CPU 변환 후 GPU 업로드 (기본 경로)
HalfToFloat()  // CPU에서 소프트웨어 변환
  → MetalBuffer::CreatePrivateInitialized()
      → MTLResourceStorageModePrivate 버퍼 생성
      → BlitCommandEncoder로 Shared→Private 복사
      → waitUntilCompleted (동기 대기)

// 선택적 f16 보존 (환경변수 SOC_GPU_ENABLE_EXPERIMENTAL_F16_WEIGHTS=1)
MetalBuffer::CreatePrivateInitialized() with f16 bytes directly
```

모델 로딩 흐름:
- embed_tokens_weight (vocab_size × hidden_size)
- final_norm_weight
- 각 레이어별: q/k/v/o_proj, gate/up/down_proj, q_norm/k_norm, layernorm → 모두 **개별 Private 버퍼**로 GPU 상주
- Q4 양자화 가중치도 지원 (`SOC_GPU_ENABLE_EXPERIMENTAL_Q4_DECODE` 환경변수)

### 2-3. 추론 실행 (`qwen_causal_lm.cpp`)

```
토큰 ID (CPU vector<int>)
  → GPU 버퍼 업로드 (Shared 버퍼, 매 스텝마다)
  → EmbeddingOp: embed_tokens_weight GPU lookup
  → RunBlockRange: 레이어별 순차 실행
      - RmsNormOp → LinearOp (QKV) → Attention → LinearOp (O) → MLP
      - 레이어당 CommandBuffer 1개 (기본 kOff 모드)
      - kLayer 모드: 레이어당 1 CommandBuffer (Flush)
      - kFullRange 모드: 전체 레인지 1 CommandBuffer (실험적)
  → KV Cache: GPU Private 버퍼로 상주
  → 결과 logits 읽기: GPU→CPU readback (waitUntilCompleted)
  → CPU Sampler로 next token 선택
```

---

## 3. 핵심 차이점 비교

| 항목 | llama.cpp (Metal) | SongWooYoung/SoC (Mac/gpu) |
|---|---|---|
| **웨이트 저장 형식** | GGUF 양자화 (Q4_K 등), GPU에서 직접 처리 | float32 기본 (f16은 실험적 환경변수), Private 버퍼 |
| **Shared Memory 활용** | Unified Memory 시 Shared 모드로 복사 생략 | **항상 Private** 버퍼 사용 (복사 발생) |
| **Command Buffer 전략** | n_cb+1개로 분할, `dispatch_apply` 병렬 인코딩 | **레이어당 1~전체 1개** (실험적), 기본은 op마다 개별 |
| **Op Fusion** | 지원 (환경변수로 제어) | 미지원 |
| **Concurrency Dispatch** | `MTLDispatchTypeConcurrent` 활용 | 미사용 |
| **Residency Set** | macOS 15+에서 활성화, 웨이트 메모리 고정 | 미구현 |
| **f16 웨이트** | 네이티브 f16/bf16 셰이더 커널 | CPU에서 f32로 변환 후 업로드 (���본) |
| **그래프 최적화** | `ggml_graph_optimize` (CSE, in-place 등) | 미구현 |
| **KV Cache** | GPU Private 버퍼 상주 | GPU Private 버퍼 상주 (동일) |
| **동기화 모델** | 비동기 commit + 나중에 drain | **동기 waitUntilCompleted** (FinalizeCommandBuffer) |
| **Hybrid 실행** | 지원 (GPU 레이어 수 동적 지정) | 지원 (`--layer` 옵션) |
| **프로파일링** | MTLCaptureManager GPU trace | GPUStartTime/GPUEndTime 기반 ms 측정 |

---

## 4. GPU 사용 최적화를 위한 개선 방향

### 🔴 우선순위 높음

#### 4-1. Unified Memory를 활용한 Shared 버퍼 도입
**현재**: 모든 웨이트를 `MTLResourceStorageModePrivate`로 업로드 → Shared→Private 복사 필요  
**개선**: Apple Silicon은 CPU/GPU가 메모리를 공유하므로 `MTLResourceStorageModeShared` 사용 시 복사 비용 제거

```cpp
// metal_context.mm 또는 metal_buffer.cpp에서
if (device_info.has_unified_memory) {
    buffer = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
    memcpy([buffer contents], data, size);
} else {
    // 기존 Private + Blit 방식 유지
}
```

> llama.cpp의 `use_shared_buffers = has_unified_memory` 패턴을 그대로 적용 가능

#### 4-2. f16 웨이트 네이티브 처리
**현재**: `HalfToFloat()` CPU 변환 → float32로 GPU에 올림 (2배 메모리, 변환 시간)  
**개선**: f16 그대��� GPU에 업로드, 셰이더에서 `half` 타입으로 직접 읽기

`SOC_GPU_ENABLE_EXPERIMENTAL_F16_WEIGHTS=1`을 기본값으로 변경하고, matmul 커널들이 `device const half*` 입력을 받도록 확장하면 됩니다. (현재 `matmul_f32_f16rhs_*` 커널들이 이미 f16 RHS를 지원하고 있음)

#### 4-3. Command Buffer 배칭 (레이어 스트리밍)
**현재**: `kOff` 모드가 기본 — op마다 또는 레이어마다 별도 Command Buffer, `waitUntilCompleted` 동기 호출  
**개선**: `SOC_GPU_ENABLE_EXPERIMENTAL_COMMAND_STREAM=full`을 기본화하거나, 최��한 decode 스텝에서 `kFullRange`를 기본으로 변경

```
현재 decode 1 step:
  [embedding CB] wait → [layer0 CB] wait → [layer1 CB] wait → ... → [lmhead CB] wait
  → N번의 CPU-GPU 동기화

개선 후:
  [embedding + all_layers + lmhead 단일 CB] 1번 commit + 1번 wait
```

---

### 🟡 우선순위 중간

#### 4-4. 동기화 모델 비동기화
**현재**: `FinalizeCommandBuffer`가 `commit` 직후 `waitUntilCompleted` 호출 — GPU idle 대기시간 발생  
**개선**: `CommitCommandBufferDeferred` + `DrainPendingCommandBuffers` 패턴을 decode 루프에 적용, CPU가 다음 토큰 준비와 GPU 실행을 오버랩

```cpp
// 현재: commit → wait (GPU idle 시간 발생)
// 개선: 여러 CB를 commit → 마지막에 일괄 drain
context.CommitCommandBufferDeferred(...);  // 비동기
// ... 다른 작업 ...
context.DrainPendingCommandBuffers(...);   // 일괄 대기
```

#### 4-5. Residency Set 도입 (macOS 15+)
모델 웨이트는 추론 내내 GPU에 상주해야 하는데, macOS 메모리 압박 시 swapout될 수 있습니다. llama.cpp처럼 `MTLResidencySet`을 사용해 웨이트 버퍼를 물리 메모리에 고정하면 첫 토큰 지연(TTFT) 스파이크를 예방할 수 있습니다.

#### 4-6. Decode 단계 토큰 ID 버퍼 재사용
**현재**: decode 매 스텝마다 `UploadGpuTokenIds`에서 새 Shared 버퍼 할당  
**개선**: 1-element 재사용 가능한 Shared 버퍼를 미리 생성, 매 스텝 `memcpy`만 수행

```cpp
// infer.mm의 RunHybridInference / RunFullGpuInference에서
// decode_tensor를 GenerationContext에 미리 할당해 재사용
```

---

### 🟢 장기 개선

#### 4-7. Op Fusion
llama.cpp처럼 `RMSNorm+Mul`, `Add+GELU` 등의 연속 연산을 단일 Metal 커널로 합치면 중간 텐서 메모리 write/read 라운드트립이 제거됩니다. `gpu_kernels.metal`에 fused 커널 추가 + `qwen_causal_lm.cpp`의 block 실행 경로에 fusion 조건 추가.

#### 4-8. `MTLDispatchTypeConcurrent` 활용
독립적인 연산 (QKV projection 3개는 입력이 동일) 을 Concurrent encoder로 디스패치하면 GPU의 여러 compute unit을 동시에 활용 가능.

#### 4-9. Q4 기본 활성화
현재 Q4 양자화 경로가 환경변수 플래그로만 활성화됩니다. `SOC_GPU_ENABLE_EXPERIMENTAL_Q4_DECODE=1`를 기본화하면 웨이트 메모리를 ~4배 절감하고 메모리 대역폭 병목을 크게 완화합니다 (decode 단계는 memory-bandwidth-bound).

---

## 요약 표

| 개선 항목 | 예상 효과 | 난이도 |
|---|---|---|
| Shared 버퍼 (Unified Memory) | 모델 로딩 속도 ↑, 메모리 절감 | 낮음 |
| f16 웨이트 기본화 | 메모리 50% 절감, 로딩 속도 ↑ | 낮음 |
| Command Buffer 배칭 | decode TPS ↑↑ | 중간 |
| 비동기 동기화 | GPU 활용률 ↑ | 중간 |
| Residency Set | TTFT 안정성 ↑ | 낮음 |
| Decode 버퍼 재사용 | 소량 오버헤드 제거 | 낮음 |
| Op Fusion | decode/prefill TPS ↑ | 높음 |
| Concurrent Dispatch | GPU 병렬성 ↑ | 중간 |
| Q4 기본 활성화 | decode TPS ↑↑ (BW 절감) | 낮음 |

가장 즉각적인 효과를 기대할 수 있는 것은 **Command Buffer 배칭 기본화** + **Shared 버퍼 도입** + **f16 웨이트 기본화**의 조합입니다.

————————————
GPT-5.2 - flash-moe


아래가 결론입니다: **flash-moe(`metal_infer`)는 “가중치를 Metal 공유 메모리 버퍼(=CPU/GPU가 같은 메모리)”에 직접 `pread`로 채워 넣어 즉시 GPU 커널이 읽게 하는 방식**으로 “모델을 GPU에 올립니다”. 반면 **@SongWooYoung/SoC (`Mac/gpu`)는 “(대부분) CPU에서 로드한 weight bytes를 `MTLResourceStorageModePrivate`(GPU 전용) 버퍼로 업로드(blit)해서 GPU에 상주시킨 뒤, `DeviceTensor/MetalBuffer` 추상화 + `CommandStream/PipelineCache`로 실행**하는 구조입니다. 즉, 핵심 차이는 **shared(호스트 가시) 중심 vs private(디바이스 전용) 중심**, 그리고 **로드/업로드/실행 경로의 분리 수준**입니다.

---

## 1) flash-moe `metal_infer`에서 “Metal GPU에 모델을 올리는 방식” 분석

`metal_infer/main.m`과 `metal_infer/infer.m`을 보면, Metal 컨텍스트는 대략 이렇게 셋업됩니다.

- `MTLCreateSystemDefaultDevice()`로 디바이스 생성
- `newCommandQueue` 생성
- `metallib` 있으면 로드, 없으면 `.metal` 소스 런타임 컴파일 후 `MTLComputePipelineState` 생성  
  (즉, **pipeline은 런타임/오프라인 둘 다 지원**)  
  ─ `metal_infer/main.m`의 `metal_init()` 흐름

그리고 “모델(가중치)”은 크게 두 종류로 다뤄집니다.

### (A) non-expert weights: mmap + (일부는) Metal buffer로 연결
`infer.m` 헤더 주석에 “Non-expert weights loaded from model_weights.bin (mmap'd at startup)”라고 명시돼 있고, 실제로 `open_weights()`에서 `model_weights.bin`을 `mmap`으로 매핑합니다. (`madvise(MADV_SEQUENTIAL)`도 사용)  
이 구조는 **CPU 주소 공간에서 weight가 즉시 접근 가능**해져서, 필요한 텐서를 포인터로 즉시 얻는 방식(`get_tensor_ptr`)을 만듭니다.

또 `MetalCtx`에 `wf_buf`(mmap'd weight file as a Metal buffer)가 들어가도록 설계되어 있어서, *일부 경로에서는* 이 big blob을 Metal에서 읽게 하는 형태도 염두에 둔 구조입니다. (`infer.m`의 `MetalCtx` 필드 설명)

### (B) MoE expert weights: “디스크 → MTLBuffer(contents)로 pread → GPU 커널이 읽음”
핵심은 expert 쪽입니다. `main.m`에서 아래 최적화가 아주 명확합니다.

- 기본(느린) 버전: expert를 9개 컴포넌트로 쪼개서 **9번 `pread` + 9개의 Metal buffer alloc**
- 최적화 버전: expert 전체를 **1번 `pread(EXPERT_SIZE)`로** 읽고, 커널에서는 **buffer offset**으로 gate/up/down의 W/S/B를 가리킴  
  (`run_expert_forward_fast()` 설명과 구현)

특히 “GPU에 올리는” 동작이 사실상 다음 한 줄로 요약됩니다:

- `pread(packed_fd, [expert_buf contents], EXPERT_SIZE, expert_offset);`  
  (`metal_buf_shared`로 만든 `MTLResourceStorageModeShared` 버퍼의 CPU 포인터에 바로 디스크 데이터를 채움)

즉 flash-moe에서는 “업로드 커맨드(blit)”가 아니라, **shared buffer를 이용해 ‘CPU가 채우면 곧바로 GPU가 읽는’** 방식이 기본입니다. (Apple Silicon의 unified memory 특성을 적극 활용)

추가로 `run_moe_forward_fused()`는 더 공격적인데:

- K개 expert에 대해 **pthread로 병렬 `pread`**를 돌려서 shared buffer K개를 채우고
- **단 1개의 command buffer**에 phase별 encoder를 여러 번 열어서 (gate/up → swiglu → down → blit+weighted_sum)
- commit+wait를 한 번만 합니다.  
  (`run_moe_forward_fused()`)

그리고 `infer.m` 상단 주석에 “CMD3 deferred commit” 같은 방식으로 **다음 레이어와 겹치게(큐 직렬성 이용)** 만드는 파이프라인링까지 서술돼 있습니다. (레이어간 CPU/GPU 동기화를 줄이려는 설계)

---

## 2) @SongWooYoung/SoC `Mac/gpu`의 GPU 사용 방식(구현)과 차이

SoC는 문서(`Mac/gpu/plan/07_IMPLEMENTATION_DECISIONS.md`)에서 “Metal direct compute runtime”을 명시하고, MPSGraph는 기본 경로가 아니라고 못 박습니다. 그리고 실제 구현도:

- `MetalContext`가 metallib 로드/소스 컴파일 fallback을 지원 (`Mac/gpu/src/metal/metal_context.mm`)
- `MetalBuffer`가 **Shared / Private / PrivateInitialized**를 명확히 분리 (`Mac/gpu/src/buffer/metal_buffer.mm`)
- `CommandStream`으로 command buffer/encoder life-cycle을 관리 (`Mac/gpu/src/metal/command_stream.mm`, 검색 결과)
- `PipelineCache` + function_constant 기반 specialization(예: matmul flags)을 염두에 둔 구조 (`gpu_kernels.metal`, `matmul_op.h` 등 검색 결과)

가장 중요한 “모델을 GPU에 올리는 방식” 차이는 여기입니다:

### (1) weight 상주 방식: flash-moe는 shared “직접 채움”, SoC는 private “업로드(블릿)”
SoC의 `MetalBuffer::CreatePrivateInitialized()`는 전형적인 GPU 업로드 패턴입니다.

1) staging shared buffer 생성  
2) CPU memcpy로 staging에 채움  
3) blit encoder로 private buffer에 copy  
4) command buffer commit+wait (`FinalizeCommandBuffer`)  

즉 SoC는 **GPU에서 빠르게 읽도록 private에 상주시킨다**가 기본이고, flash-moe는 **호스트가시(shared)에 둔 채로도 성능을 뽑는 방향**(특히 I/O 스트리밍 MoE)에 가깝습니다.

### (2) 실행/동기화 모델: SoC는 “추상화 + 측정/드레인”, flash-moe는 “핫패스 특화 파이프라인”
SoC의 `MetalContext`는
- `FinalizeCommandBuffer()`에서 GPUStart/EndTime으로 gpu_ms 측정
- `CommitCommandBufferDeferred()` / `DrainPendingCommandBuffers()`로 deferred 커밋도 지원

즉 **런타임 레벨에서 deferred/drain을 공용 기능으로 제공**합니다. 반면 flash-moe는 특정 모델 구조(MoE, 레이어 3 cmd buffers) 기준으로 “이렇게 하면 빨라진다”를 코드에 강하게 박아둔 스타일입니다.

### (3) “GPU를 쓰는 범위”: SoC는 end-to-end GPU 경로를 점점 넓히는 중, flash-moe는 CPU/GPU 혼합이 명시적
SoC는 `QwenCausalLM`, `QwenBlock`, `QwenMLP`, `MatMulOp`, `AffineQmmOp`, `KVCache` 등으로 **모듈 단위로 GPU 오퍼레이터를 확장**하고 있고,
Sampler는 성능 이슈 때문에 기본은 CPU fallback을 두는 정책이 코드/문서로 정리돼 있습니다. (`Mac/gpu/src/runtime/sampler.cpp`, `Mac/gpu/test/errors/02_performance_regressions.md` 검색 결과)

flash-moe도 CPU/GPU 혼합이지만, 그 이유/구조가 “SSD 스트리밍 + topK + expert pread” 중심으로 굉장히 구체적입니다(특히 MoE).

---

## 3) GPU 사용 최적화를 위해 SoC에서 개선해야 할 방향 (flash-moe에서 배울 점 중심)

아래는 **지금 SoC의 설계(Private buffer 상주, CommandStream, PipelineCache)와 충돌하지 않으면서**, flash-moe가 강하게 밀어붙인 최적화 포인트를 SoC에 맞게 “일반화”한 개선 방향입니다.

### A. weight/activation의 “storage mode 정책”을 더 공격적으로 분리
지금 SoC는 `MetalBuffer`가 shared/private를 제공하지만, **어떤 텐서는 shared가 유리하고 어떤 텐서는 private가 유리한지 정책 레이어가 더 강화되면 좋습니다.**

- **장기 상주 weight(큰 matmul rhs, KV cache 등)**: private 권장 (대역폭/캐시 관점)
- **자주 CPU에서 읽거나(샘플링), 디버그/검증용 readback이 잦은 텐서**: shared 유지
- **decode 경로의 작은 scratch**: shared로 두고 “CPU readback 비용 최소화” 선택지도 남김

flash-moe가 MoE expert를 shared로 둔 이유는 “disk→GPU”를 중간 복사 없이 연결하려는 의도라서, SoC도 만약 **asset을 파일로부터 스트리밍**할 계획이 있으면,
- “staging(shared)→private 업로드”를 항상 강제하지 말고
- **shared에서 바로 커널이 읽는 스트리밍 경로**(옵션)를 만들어두는 게 확장성에 좋습니다.

### B. “expert/큰 weight 조각” 로딩에서: N개의 작은 read/alloc을 없애고 “1 read + offset view”로 바꾸기
flash-moe `run_expert_forward_fast()`의 핵심은:
- **한 덩어리로 읽고**
- **offset으로 view를 나눠서** 커널에 바인딩

SoC도 (특히 quant weight: qweight/scales/qbiases 같은 3종 세트가 반복되는 경우)
- 파일 포맷을 바꾸거나(또는 pack 단계 추가),
- 로더에서 “연속 배치 + offset” 구조로 바꾸면,
  - 파일 I/O 호출 수 감소
  - Metal buffer 객체 수 감소
  - 바인딩 횟수 감소
가 동시에 일어납니다.

### C. command buffer/encoder batching을 “정책화”하기
flash-moe는 “K experts를 **단일 command buffer**에 phase별 encoder로 몰아넣기”로 CPU 오버헤드를 크게 줄였습니다.

SoC도 이미 `CommandStream`이 있고 `QwenBlock`에서 flush 전략을 바꾸는 코드가 보이므로(예: layer batch vs full range batch),
다음 개선이 유효합니다.

- prefill에서는 **큰 배치(토큰 수 많음)** → 더 큰 command buffer batching이 유리
- decode에서는 **작은 matmul이 많음** → batching + specialized kernels(예: vec4, decode-tiled)이 핵심
- “어느 op 경계에서 flush할지”를 하드코딩 말고
  - token count, hidden size, device info(thread_execution_width 등)
  - profiling snapshot(gpu_ms vs wait_ms)
  기반으로 자동 튜닝/휴리스틱화

### D. Sampler(top-k) 경로는 “GPU로 계속 미는 게 정답이 아닐 수 있음”을 전제로 최적화
SoC 문서/코드에서 이미 “GPU top-k가 느려서 CPU fallback이 더 낫다”는 관찰이 있습니다. (검색 결과에 포함)

여기서 개선 방향은 두 갈래입니다.

1) **CPU fallback을 ‘최적’으로 만들기**  
   - logits를 shared에 두고 readback 비용 최소화
   - partial sort/top-k를 SIMD/Accelerate(vDSP) 활용
2) **GPU로 간다면 알고리즘을 바꾸기**  
   - single-row scalar scan 형태를 버리고
   - 병렬 reduction + block-wise top-k, 또는 sampling을 바꾸는 접근(예: top-p 기반의 병렬 prefix sum 등)

flash-moe도 top-k/softmax는 CPU에서 하는 부분이 명시돼 있고, “GPU에서 다 한다”가 목적이 아니라 **전체 토큰 latency 최소화**가 목적입니다. SoC도 같은 관점으로 “decode 병목이 샘플링이면, 샘플링은 CPU가 낫다” 같은 실용적인 선택을 더 적극 반영하는 게 좋습니다.

---

## 4) 내가 추가로 확인하면 더 정확해지는 것 (SoC 쪽)
지금 도구 검색 결과가 **최대 10개로 제한**돼서(불완전할 수 있음), “weight 로딩이 실제로 private에 얼마나 올려져 있는지(모델 전체 로더 경로)”는 `Mac/gpu/src/model/qwen_model_loader.cpp`의 로딩 함수들(`LoadFloatingWeight`, `LoadRawWeight`) 구현까지 보면 확정적으로 말할 수 있습니다.  
GitHub 검색 UI에서 더 보려면 아래 쿼리로 확인하면 됩니다.

- `repo:SongWooYoung/SoC LoadFloatingWeight`
- `repo:SongWooYoung/SoC LoadRawWeight`


———————————————

Codex-5.4 pytorch 분석

핵심부터 말하면, **PyTorch의 macOS GPU 경로는 “모델을 MPS/Metal 디바이스로 옮기면 백엔드가 연산 단위로 GPU 커널 실행을 관리하는 구조”**이고, **당신의 `SongWooYoung/SoC`는 “직접 Metal runtime·buffer·pipeline·scheduler를 설계해서 모델을 GPU 친화 포맷으로 올리고 실행하는 구조”**입니다.  
즉 PyTorch는 **범용 프레임워크형 GPU 사용**, SoC는 **전용 inference runtime형 GPU 사용**에 가깝습니다.

아래에 차이와 개선 방향을 정리합니다.

---

## 1. PyTorch에서 Metal/Mac GPU에 모델을 “어떻게 올리는가”

엄밀히 말하면 최근 macOS PyTorch의 주력 GPU 백엔드는 **MPS(Metal Performance Shaders / Metal 기반 백엔드)**이고, 코드베이스에는 별도로 **mobile-oriented Metal backend** 흔적도 있습니다.  
당신이 물은 “metal 구현” 관점에서 보면 PyTorch에는 두 층이 보입니다.

### A. 저수준 Metal backend 등록 구조
PyTorch에는 Metal 인터페이스 등록점이 있습니다.

```c++ name=aten/src/ATen/metal/Context.h url=https://github.com/pytorch/pytorch/blob/d44c4eebc4f6ccd932c13af376fecca72584c530/aten/src/ATen/metal/Context.h#L1-L32
#ifndef MetalContext_h
#define MetalContext_h

#include <atomic>

#include <ATen/Tensor.h>

namespace at::metal {

struct MetalInterface {
  virtual ~MetalInterface() = default;
  virtual bool is_metal_available() const = 0;
  virtual at::Tensor& metal_copy_(at::Tensor& self, const at::Tensor& src)
      const = 0;
};

extern std::atomic<const MetalInterface*> g_metal_impl_registry;
...
```

그리고 실제 구현체는 Metal availability 확인과 copy를 제공합니다.

```objective-c++ name=aten/src/ATen/native/metal/MetalAten.mm url=https://github.com/pytorch/pytorch/blob/d44c4eebc4f6ccd932c13af376fecca72584c530/aten/src/ATen/native/metal/MetalAten.mm#L100-L128
struct MetalImpl : public at::metal::MetalInterface {
  bool is_metal_available() const override {
#if defined(USE_PYTORCH_METAL)
    return [[MetalContext sharedInstance] available];
#else
    return false;
#endif
  }
  at::Tensor& metal_copy_(at::Tensor& input, const at::Tensor& src)
      const override {
    TORCH_CHECK(
        is_metal_available(), "Metal is not available on the current device");
    return native::metal::metal_copy_impl_(input, src);
  }
};
```

이 의미는:
- PyTorch 내부에는 **Metal device abstraction**이 있고
- tensor copy를 통해 **CPU tensor → Metal tensor** 이동이 가능하며
- 연산은 등록된 backend dispatch를 통해 실행됩니다.

### B. 모델 자체는 고수준 API로 device 이동
사용자 관점에서는 보통 `model.to("mps")`, `tensor.to("mps")` ��은 방식으로 모델과 텐서를 GPU로 올립니다.  
당신 repo의 benchmark도 바로 그 경로를 사용합니다.

```python name=Mac/benchmark_pytorch.py url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/benchmark_pytorch.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
...
```

즉 PyTorch에서 “모델을 GPU에 올린다”는 것은 개념적으로:

1. 파라미터/버퍼를 GPU device tensor로 변환
2. 각 op를 해당 backend(MPS/Metal)가 처리
3. 필요하면 runtime이 kernel 선택, 메모리 관리, dispatch를 담당

하는 방식입니다.

### C. PyTorch의 구형/모바일 Metal 경로는 “모델 변환 + prepack + rewrite” 성격이 강함
모바일 Metal 최적화 패스도 있습니다.

```c++ name=torch/csrc/jit/passes/metal_rewrite.cpp url=https://github.com/pytorch/pytorch/blob/d44c4eebc4f6ccd932c13af376fecca72584c530/torch/csrc/jit/passes/metal_rewrite.cpp#L248-L265
script::Module metalOptimizeForMobile(
    const script::Module& m,
    const std::vector<std::string>& preserved_methods) {
  auto cloned_module = m.clone();
  cloned_module.eval();
  cloned_module = FoldConvBatchNorm(cloned_module);
  metalInsertPrePackedOps(cloned_module);
  cloned_module = freeze_module(cloned_module, preserved_methods);
  metalFusePrePackedConvWithClamp(cloned_module);
  metalFoldPrePackingOps(cloned_module);
  removeDropout(cloned_module);
  metalRemoveMutation(cloned_module);
  metalRunCanonicalOptimizations(cloned_module);
  cloned_module.register_attribute(
      "optimized_for_metal", BoolType::get(), true);
  return cloned_module;
}
```

이건 중요한 시사점이 있습니다. PyTorch도 Metal에서 성능을 내기 위해:
- 단순 device 이동만 하는 것이 아니라
- **prepack**
- **graph rewrite**
- **fusion**
- **mutation 제거**
를 합니다.

이건 당신 repo가 앞으로 가야 할 방향과도 매우 유사합니다.

---

## 2. SoC의 Mac/gpu 구현은 무엇이 다른가

당신의 repo는 PyTorch처럼 범용 backend를 쓰지 않고, **Metal direct compute runtime**을 직접 구현하는 방향이 명확합니다.

```markdown name=Mac/gpu/plan/07_IMPLEMENTATION_DECISIONS.md url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/plan/07_IMPLEMENTATION_DECISIONS.md#L5-L15
# Implementation Decisions

## Confirmed Now

1. `Mac/gpu`는 Metal direct compute runtime으로 설계한다.
   이유: user requirement가 명시적으로 Metal 기반이며, GPU plan의 핵심 가치도 low-level control에 있다.
...
3. MPSGraph는 Phase 1의 기본 경로가 아니다.
```

또 전체 mission도 매우 분명합니다.

```markdown name=Mac/gpu/plan/README.md url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/plan/README.md#L11-L21
## Mission

이 프로젝트의 최종 목적은 다음 두 가지를 동시에 만족하는 것이다.

1. Apple Silicon의 GPU를 사용해 Metal 기반으로 LLM inference를 수행한다.
2. 추후 training 또는 최소한 backward-capable graph로 확장 가능한 구조를 처음부터 설계한다.
...
1. host-device 메모리 이동 최소화
2. Metal command scheduling과 kernel specialization
3. KV cache, attention, matmul, normalization의 fused path 설계
4. weight layout prepack과 pipeline state cache 전략
```

즉 SoC는:
- **device / buffer / tensor / kernel / op / module / scheduler / runtime**를 직접 나누고
- 실행용 CLI도 따로 두고
- 모델 로더가 weight를 직접 읽고
- hybrid/full-gpu/full-cpu 실행 모드를 직접 관리합니다.

예를 들면 infer 경로에서 GPU 업로드를 직접 합니다.

```objective-c++ name=Mac/gpu/infer.mm url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/infer.mm#L251-L275
soc::gpu::DeviceTensor UploadGpuTokenIds(const soc::gpu::MetalContext& context,
                                         const std::vector<int>& token_ids,
                                         const std::string& label,
                                         std::string* error_message) {
    auto token_buffer = soc::gpu::MetalBuffer::CreateShared(context,
                                                            token_ids.size() * sizeof(std::int32_t),
                                                            label,
                                                            error_message);
    if (token_buffer == nullptr) {
        return {};
    }
...
```

모델 weight 로딩도 runtime 쪽에서 직접 합니다.

```c++ name=Mac/gpu/src/model/qwen_model_loader.cpp url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/src/model/qwen_model_loader.cpp#L318-L323
const std::string k_proj_name = LayerTensorName(layer_index, "self_attn.k_proj.weight");
const std::string v_proj_name = LayerTensorName(layer_index, "self_attn.v_proj.weight");
const std::string o_proj_name = LayerTensorName(layer_index, "self_attn.o_proj.weight");
const std::string gate_proj_name = LayerTensorName(layer_index, "mlp.gate_proj.weight");
const std::string up_proj_name = LayerTensorName(layer_index, "mlp.up_proj.weight");
const std::string down_proj_name = LayerTensorName(layer_index, "mlp.down_proj.weight");
```

또 샘플링조차 GPU/CPU fallback 경계를 직접 관리합니다.

```c++ name=Mac/gpu/src/runtime/sampler.cpp url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/src/runtime/sampler.cpp#L133-L155
bool Sampler::SampleFromLogits(...){
    ...
    if (should_use_cpu_sampler) {
        return SampleFromLogitsCpuFallback(...);
    }
    ...
}
```

---

## 3. PyTorch vs SoC: “GPU에 올리는 방식”의 본질적 차이

## A. 추상화 레벨 차이

### PyTorch
- 모델 정의는 `nn.Module`
- 파라미터 이동은 `.to(device)`
- op dispatch는 프레임워크가 담당
- backend는 MPS/Metal이 내부적으로 처리
- 사용자는 “모델을 device에 올린다”는 선언만 함

### SoC
- 모델 weight 로드, buffer 생성, tensor descriptor, kernel dispatch를 직접 관리
- GPU memory layout도 직접 정함
- 어느 연산을 GPU/CPU로 보낼지 직접 정함
- pipeline cache, command buffer scheduling도 직접 설계

**정리:**  
PyTorch는 **framework-managed placement**,  
SoC는 **runtime-managed placement**입니다.

---

## B. 메모리 이동 정책 차이

당신의 설계 문서는 host-device 이동 최소화를 가장 우선 과제로 둡니다.

```markdown name=Mac/gpu/plan/README.md url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/plan/README.md#L11-L21
1. host-device 메모리 이동 최소화
2. Metal command scheduling과 kernel specialization
3. KV cache, attention, matmul, normalization의 fused path 설계
4. weight layout prepack과 pipeline state cache 전략
```

PyTorch도 물론 device tensor를 유지하지만, 범용 프레임워크 특성상:
- eager op 단위 경계
- 일반화된 tensor semantics
- autograd/shape/dtype/device consistency
를 유지해야 해서, 전용 runtime만큼 공격적으로 “layout과 lifetime을 고정”하기 어렵습니다.

SoC는 반대로 inference 전용이므로:
- KV cache layout 고정
- weight prepack 고정
- command sequence 고정
- 디코드 루프 재사용
을 더 강하게 할 수 있습니다.

즉 **잠재적으로는 SoC가 더 최적화 여지가 큽니다.**  
하지만 지금은 그 최적화 포인트들이 아직 “계획/초기 구현 단계”에 더 가까워 보입니다.

---

## C. 커널 전략 차이

당신 repo 문서가 이미 아주 정확히 인식하고 있습니다.

```markdown name=Mac/gpu/plan/comparison/transformers/README.md url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/plan/comparison/transformers/README.md#L108-L124
현재 우리 병목은 여전히 projection matmul이다.
...
- weight bandwidth를 크게 줄이고
- dequant + matmul을 fused 하고
- Apple GPU에 맞는 kernel을 쓰는

방향이다.

즉 우리가 PyTorch MPS를 넘기고 싶다면, 다음 큰 축은 단순 threadgroup width 조정이 아니라:

- `float16` 완성
- 그 다음 `4-bit / 8-bit packed weight + fused qmm`
```

이 대목이 핵심입니다.

### PyTorch 측
- 고수준 op들을 backend가 처리
- 경우에 따라 fused path, SDPA, specialized kernel 사용
- 최근 경로는 MPS/Inductor/AOTI 등과도 연결됨

### SoC 측
- custom kernel이 핵심 경쟁력
- 그런데 현재 병목이 dense projection matmul이라면
- 단순히 “Metal로 돌린다”만으로는 PyTorch보다 빠르기 어려움

즉 SoC가 이기려면:
1. **전용 weight layout**
2. **fused dequant + matmul**
3. **decode 특화 kernel**
4. **KV cache friendly layout**
이 필수입니다.

---

## D. fallback 경계 차이

PyTorch는 backend 미지원 op가 나오면 다른 경로나 CPU fallback 가능성이 있고, 일반적으로 그 복잡성이 프레임워크 내부에 숨겨져 있습니다.

SoC는 fallback 경계를 당신이 직접 드러내고 관리합니다:
- full-cpu
- hybrid
- full-gpu
- sampler cpu fallback

```objective-c++ name=Mac/gpu/infer.mm url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/infer.mm#L251-L263
if (requested_layers == 0) {
    plan.mode = "full-cpu";
} else if (requested_layers == plan.model_layer_count) {
    plan.mode = "full-gpu";
} else {
    plan.mode = "hybrid";
}
```

이건 장점과 단점이 동시에 있습니다.

- 장점: 병목과 데이터 이동을 직접 볼 수 있음
- 단점: hybrid 구간에서 **CPU↔GPU 왕복**이 조금만 생겨도 성능이 쉽게 무너짐

---

## 4. 현재 시점에서 보이는 SoC의 구조적 한계

직접적으로 말하면, 지금 문서와 코드 조각 기준으로 SoC는 **아키텍처 방향은 매우 좋지만, GPU 사용 최적화는 아직 “foundation + baseline” 단계**로 보입니다.

특히 아래가 보입니다.

### 1. Shared buffer 업로드가 기본 경로로 보임
`UploadGpuTokenIds()`에서 `MetalBuffer::CreateShared`를 사용합니다.

```objective-c++ name=Mac/gpu/infer.mm url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/infer.mm#L263-L275
auto token_buffer = soc::gpu::MetalBuffer::CreateShared(context,
                                                        token_ids.size() * sizeof(std::int32_t),
                                                        label,
                                                        error_message);
```

Apple Silicon unified memory라 shared가 나쁜 것은 아니지만,  
모든 리소스를 shared 중심으로 가져가면:
- GPU 전용 접근 locality
- write/read ownership
- resource usage hint
최적화가 제한될 수 있습니다.

### 2. CPU fallback가 아직 남아 있음
샘플러의 CPU fallback은 초기 단계에선 합리적이지만, 디코드 루프에서는 토큰당 작은 CPU 개입도 누적 비용이 큽니다.

### 3. 모델 로더는 weight name 기반 개별 적재 중심
이 방식은 correctness에는 좋지만, **GPU-friendly prepacked blob**로 재구성하는 계층이 더 중요합니다.

### 4. runtime 계획상 prepack / pipeline cache / fused path는 아직 진행 중
문서상 roadmap에는 분명히 있지만, 그것이 바로 아직 성능 차이의 원인일 가능성이 큽니다.

---

## 5. GPU 사용을 최적화하기 위해 SoC에�� 개선해야 할 방향

중요도 순으로 말하겠습니다.

## 1) “모델을 GPU에 올린다”를 단순 buffer upload가 아니라 **GPU 실행 포맷으로의 영구 prepack**으로 바꿔야 함

PyTorch의 mobile metal rewrite도 prepack을 핵심으로 둡니다.  
당신 repo도 여기를 가장 강하게 밀어야 합니다.

### 해야 할 것
- `q_proj/k_proj/v_proj/o_proj/gate/up/down` weight를
  - GPU kernel tile에 맞�� 레이아웃으로 변환
  - decode/pre-fill 각각에 맞는 packed format 제공
- 로딩 시 1회 prepack 후, 이후 실행 루프에서는 원본 weight를 안 만지게 설계
- manifest에 “runtime private prepack metadata” 추가

당신 문서도 이미 이 방향을 암시합니다.

```markdown name=Mac/gpu/plan/05_QWEN3_IMPLEMENTATION_ROADMAP.md url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/plan/05_QWEN3_IMPLEMENTATION_ROADMAP.md#L3-L33
4. GPU prepack metadata를 runtime private로 둘지 확정
```

**가장 큰 성능 개선 가능성은 여기 있습니다.**

---

## 2) float32 중심 경로에서 **float16 중심 경로**로 빨리 이동해야 함

당신 문서가 이미 정답을 적어놨습니다.

```markdown name=Mac/gpu/plan/comparison/transformers/README.md url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/plan/comparison/transformers/README.md#L108-L124
- `float16` 완성
- 그 다음 `4-bit / 8-bit packed weight + fused qmm`
```

Apple GPU에서 LLM inference 병목은 대개:
- memory bandwidth
- projection matmul
입니다.

그래서 float16 전환 효과가 큽니다:
- weight bandwidth 절감
- cache 적중 향상
- threadgroup 내 register/shared footprint 감소
- 더 많은 in-flight work 가능

우선순위:
1. embedding 제외 주요 linear 경로 fp16
2. RMSNorm / RoPE / attention intermediate fp16 또는 mixed precision
3. logits/probability 일부만 fp32 유지

---

## 3) **dequant + matmul fused path**를 별도 2단계 목표가 아니라 핵심 목표로 격상해야 함

PyTorch와의 비교 문서에서도 인정했듯, dense float path만으로는 어렵습니다.  
LLM inference에서는 **weight bandwidth가 절대적 병목**인 ��우가 많습니다.

### 제안
- 8-bit packed linear 먼저
- 그 다음 4-bit grouped quant
- dequant buffer를 따로 만들지 말고, kernel 내부에서 바로 matmul accumulate
- prefill용 kernel과 decode용 kernel을 분리

왜 분리해야 하냐면:
- prefill은 sequence 길이가 길고
- decode는 batch/token 길이가 매우 작아
- 최적 타일/스레드 구성이 다르기 ���문입니다.

---

## 4) CPU fallback 제거 순서를 정해서 decode loop를 **완전 GPU resident**로 만들어야 함

현재 sampler fallback이 보입니다. 디코드 단계는 토큰마다 반복되므로:
- logits → topk → sample
- KV update
- next token embedding lookup
이 경로 중 CPU 왕복이 있으면 throughput이 크게 떨어집니다.

### 권장 순서
1. top-k / argmax / sampling GPU화
2. logits post-process GPU화
3. next token 준비까지 GPU resident 유지
4. truly 필요한 시점에만 CPU로 token id readback

즉 “토큰 하나 생성할 때 CPU가 하는 일”을 최소화해야 합니다.

---

## 5) hybrid 모드는 디버깅/검증용으로 유지하고, 성능 목표는 **full-gpu steady state**로 분리해야 함

현재 `full-cpu / hybrid / full-gpu` 모드는 매우 유용합니다.  
하지만 성능 최적화에서는 hybrid가 자주 함정이 됩니다.

### 이유
- layer 경계마다 synchronization
- CPU/GPU ownership 전환
- intermediate tensor materialization
- command buffer fragmentation

그래서 benchmark/optimization 체계를 분리하세요.

### 추천
- correctness 모드: hybrid 허용
- performance 모드: full-gpu only
- benchmark 결과도 prefill/decode 각각 full-gpu 기준선 별도 측정

당신 Makefile도 이미 PyTorch 비교 벤치 경로를 갖고 있으니 이 체계화가 좋습니다.

```makefile name=Mac/gpu/Makefile url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/Makefile#L217-L229
$(PYTHON_BIN) tools/benchmark_full_gpu_vs_pytorch.py \
    --infer-bin $(INFER_BIN) \
    --manifest $(REAL_QWEN3_MANIFEST) \
    --hf-model $(HF_QWEN3_MODEL_DIR) \
...
```

---

## 6) command buffer / encoder scheduling을 “연산 correctness”가 아니라 **decode latency budget** 기준으로 재설계해야 함

LLM decode에서는 토큰당 launch overhead가 커집니다.  
SoC가 PyTorch보다 이길 수 있는 몇 안 되는 강점이 바로 이 low-level scheduling control입니다.

### 제안
- 토큰당 여러 작은 kernel launch를 줄이기
- 가능한 op fusion:
  - RMSNorm + linear input prep
  - QKV projection packing
  - attention score/postprocess 일부 결합
- command buffer reuse 패턴 연구
- pipeline state cache warmup 추가

당신 plan 문서의 방향과 정확히 일치합니다.

```markdown name=Mac/gpu/plan/README.md url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/plan/README.md#L34-L43
3. Hard modularization
...
4. Optimization is mandatory
```

---

## 7) KV cache를 모델 정확성 자료구조가 아니라 **GPU access pattern 자료구조**로 다시 봐야 함

당신 mission에서 KV cache fused path 설계를 이미 우선순위로 두고 있습니다.  
이건 매우 맞습니다.

### 최적화 포인트
- head-major / seq-major / tile-major layout 중 Apple GPU에 맞는 레이아웃 검증
- append/write 패턴과 attention read 패턴을 동시에 고려
- prefill과 decode에서 다른 access pattern이면 dual layout도 고려
- page/block 단위 cache layout 고려

KV cache는 단순 저장소가 아니라 **attention bandwidth를 좌우하는 핵심 자료구조**입니다.

---

## 8) Apple GPU 전용 autotuning / specialization layer가 필요함

PyTorch는 범용성이 높아서 모든 workload에 최적일 수는 없습니다.  
SoC는 전용 runtime이므로 다음을 직접 할 수 있습니다.

### 해야 할 것
- hidden size / head dim / seq len / kv len 별 kernel variant
- threadgroup size autotune
- simdgroup reduction 전략 비교
- pipeline compile key를 `(op, dtype, tile, causal, head_dim, quant_scheme)` 수준으로 세분화

�� 단순한 pipeline cache를 넘어서 **shape-aware specialization cache**가 필요합니다.

---

## 9) benchmark를 “전체 tok/s”만 보지 말고 병목 분해형으로 바꿔야 함

이미 comparison 계획이 있지만, 앞으로는 아래를 강제 측정하는 게 좋습니다.

### 최소 측정 항목
- model load/prepack time
- first token latency
- steady-state decode tok/s
- projection matmul 비중
- attention 비중
- KV cache read/write 시간
- sampler/readback 시간
- CPU sync 횟수
- command buffer count per token

PyTorch와 비교할 때도 총량보다 **왜 차이가 나는지**가 중요합니다.

---

## 6. 한 문장으로 비교 요약

### PyTorch
“모델을 MPS/Metal device로 이동시키면, 프레임워크가 backend dispatch와 최적화 패스를 통해 GPU 실행을 관리한다.”

### SoC
“모델을 GPU용 버퍼/텐서/커맨드/커널 단위로 직접 적재하고 스케줄링하는 전용 Metal runtime을 구현한다.”

그래서:
- **개발 생산성/안정성**은 PyTorch가 강하고
- **전용 inference 최적화 잠재력**은 SoC가 더 큽니다.

하지만 그 잠재력이 현실 성능이 되려면, 지금 단계에서 가장 중요한 것은  
**��그냥 Metal 위에서 돈다”에서 “Apple GPU에 맞게 prepack·quantize·fuse된 decode runtime이다”로 가는 것**입니다.

---

## 7. 실질적 개선 우선순위 Top 5

바로 실행 우선순위만 뽑으면:

1. **weight prepack 체계 확립**
   - layer별 원본 weight → GPU 전용 packed blob
2. **fp16 end-to-end 완성**
   - 특히 projection matmul 경로
3. **decode loop GPU resident화**
   - sampler/readback 포함 CPU 개입 최소화
4. **quantized fused linear(8-bit → 4-bit) 도입**
   - dequant + matmul fusion
5. **KV cache layout / command scheduling specialization**
   - prefill/decode 분리 최적화




————————————————

Claude sonnet 4.6 - pytorch

이제 두 코드베이스를 충분히 분석했습니다. 종합적인 분석을 제공하겠습니다.

---

## 분석 결과

---

## 1. PyTorch MPS에서 Metal GPU에 모델을 올리는 방식

PyTorch의 MPS 백엔드는 다음 계층 구조로 GPU에 모델을 탑재합니다.

### 1-1. 디바이스 초기화 (`MPSDevice`)

```objective-c++ name=aten/src/ATen/mps/MPSDevice.mm url=https://github.com/pytorch/pytorch/blob/d44c4eebc4f6ccd932c13af376fecca72584c530/aten/src/ATen/mps/MPSDevice.mm#L25-L46
// 싱글턴 패턴으로 시스템 기본 GPU 선택
NSArray* devices = [MTLCopyAllDevices() autorelease];
for (...) {
    if ([device isLowPower]) continue;  // Intel iGPU 제외
    if (![device supportsFamily:MTLGPUFamilyMac2]) continue;  // Metal 2.0 미지원 제외
    _mtl_device = [device retain];
}
```

`MPSDevice::getInstance()`가 싱글턴을 반환하며, Apple Silicon GPU 하나만 선택합니다.

### 1-2. 메모리 할당 (`MPSHeapAllocatorImpl`) — **Heap 기반 풀링**

```objective-c++ name=aten/src/ATen/mps/MPSAllocator.mm url=https://github.com/pytorch/pytorch/blob/d44c4eebc4f6ccd932c13af376fecca72584c530/aten/src/ATen/mps/MPSAllocator.mm#L41-L54
// 3종류의 MTLHeap 풀을 미리 생성
m_pools.emplace(BufferPool::Kind::SHARED_LARGE, ...);   // 대형 버퍼
m_pools.emplace(BufferPool::Kind::SHARED_SMALL, ...);   // 소형 버퍼 (<= kMaxSmallAlloc)
m_pools.emplace(BufferPool::Kind::SCALAR, ...);         // 스칼라 값
```

핵심 특성:
- **MTLHeap 기반 서브 할당**: `newBufferWithLength` 대신 Heap에서 버퍼를 서브 할당 → 할당 오버헤드 최소화
- **버퍼 재사용 (GC 기반)**: 반환된 버퍼를 풀에 보관 후 재사용 (`get_free_buffer`)
- **GC watermark**: `PYTORCH_MPS_HIGH_WATERMARK_RATIO` / `LOW_WATERMARK_RATIO` 환경변수로 메모리 압력 제어

### 1-3. 텐서 생성 → GPU 탑재 흐름

```
model.to("mps")
  └─> 각 파라미터 Tensor.to(device="mps")
        └─> empty_mps() [EmptyTensor.cpp]
              └─> GetMPSAllocator()->allocate(nbytes)
                    └─> MPSHeapAllocatorImpl::malloc()
                          └─> alloc_buffer_block() → MTLBuffer (Heap에서 서브할당)
```

모델 가중치는 CPU → MPS 복사 시 `copy_and_sync()`를 통해 blit encoder로 전송됩니다.

### 1-4. 커맨드 실행 스트림 (`MPSStream`) — **비동기 배치 실행**

```objective-c++ name=aten/src/ATen/mps/MPSStream.mm url=https://github.com/pytorch/pytorch/blob/d44c4eebc4f6ccd932c13af376fecca72584c530/aten/src/ATen/mps/MPSStream.mm#L52-L70
// 싱글 커맨드 버퍼를 재사용 — 하나의 버퍼에 여러 인코더 누적
MPSCommandBuffer* MPSStream::commandBuffer() {
    if (!_commandBuffer) {
        _commandBuffer = [MPSCommandBuffer commandBufferFromCommandQueue:_commandQueue].retain;
    }
    return _commandBuffer;  // ← 기존 버퍼를 재사용
}
// 하나의 커맨드 인코더도 재사용
id<MTLComputeCommandEncoder> MPSStream::commandEncoder() {
    if (!_commandEncoder) {
        _commandEncoder = [commandBuffer() computeCommandEncoder].retain;
    }
    return _commandEncoder;  // ← 기존 인코더를 재사용 (kernel coalescing)
}
```

`commitAndContinue`와 **Kernel Coalescing**을 통해 다수의 커널을 하나의 커맨드 버퍼에 묶어 제출합니다.

---

## 2. SoC 레포의 GPU 사용 방식 및 PyTorch와의 차이

### 2-1. SoC의 GPU 탑재 방식

**디바이스 초기화 (`MetalContext`):**

```objective-c++ name=Mac/gpu/src/metal/metal_context.mm url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/src/metal/metal_context.mm#L114-L190
// MTLCreateSystemDefaultDevice() — 단순 디바이스 취득
id<MTLDevice> device = MTLCreateSystemDefaultDevice();
id<MTLCommandQueue> command_queue = [device newCommandQueue];
// metallib 로드 또는 런타임 소스 컴파일
library = [device newLibraryWithURL:...] ?? LoadLibraryFromSource(device, ...);
```

**버퍼 할당 (`MetalBuffer`):**

```objective-c++ name=Mac/gpu/src/buffer/metal_buffer.mm url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/src/buffer/metal_buffer.mm#L26-L50
// 직�� newBufferWithLength 호출 — Heap 없음, 풀 없음
id<MTLBuffer> buffer = [device newBufferWithLength:size_bytes 
                                           options:MTLResourceStorageModeShared];
```

**커맨드 버퍼 실행 (`FinalizeCommandBuffer`):**

```objective-c++ name=Mac/gpu/src/metal/metal_context.mm url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/src/metal/metal_context.mm#L222-L273
// 매 Op마다: commit → waitUntilCompleted (동기 블로킹)
[command_buffer commit];
[command_buffer waitUntilCompleted];
```

### 2-2. 핵심 차이점 비교표

| 항목 | PyTorch MPS | SoC Mac/gpu | 영향 |
|------|-------------|-------------|------|
| **버퍼 할당** | MTLHeap 기반 서브할당 + 재사용 풀 | `newBufferWithLength` 직접 호출 | SoC는 매 할당마다 Metal driver 진입 비용 발생 |
| **커맨드 버퍼** | 싱글 스트림 재사용 + Kernel Coalescing | Op마다 새 커맨드 버퍼 생성 | **SoC: ~395 CB/token** → 최대 20ms 순수 오버헤드 |
| **동기화** | `COMMIT_ADAPTIVE` / `commitAndContinue` 비동기 | 매 Op마다 `waitUntilCompleted` | SoC는 CPU가 GPU 완료를 항상 대기 |
| **메모리 정책** | High/Low watermark GC, 자동 캐시 해제 | `BufferArena` (선형 bump 할당, reset) | SoC는 GC 없음, 파편화 없지만 유연성 낮음 |
| **데이터 타입** | float16 완전 지원 (half 연산 네이티브) | float32 전용 (f16 가중치 → f32 업캐스트) | SoC는 메모리 대역폭 2배 낭비 |
| **리덕션 커널** | MPSGraph 기반 (병렬 최적화됨) | 단일 스레드 row-wise loop | SoC 1024 원소 처리 시 1 thread |
| **쉐이더 로딩** | 미리 컴파일된 metallib 필수 | metallib OR 런타임 소스 컴파일 | 런타임 컴파일 시 초기화 지연 |
| **Private 버퍼** | 가중치에 `MTLResourceStorageModePrivate` 사용 | 모든 버퍼가 `StorageModeShared` | SoC는 GPU-only 최적화 불가 |

---

## 3. SoC 레포에서 GPU 사용 최적화를 위한 개선 방향

이미 `00_OPTIMIZATION_MASTER_PLAN.md`에 방향성이 잘 잡혀 있습니다. 코드 분석 결과를 바탕으로 구체적 구현 관점에서 우선순위를 정리합니다.

### 🔴 Priority 1: Command Buffer Batching (B1 — 예상 2~5x)

**현재 문제**: `FinalizeCommandBuffer`에서 op마다 `commit + waitUntilCompleted` 동기 블로킹.

**개선 방향**: PyTorch의 `MPSStream` 패턴 채택. `CommandStream`이 커맨드 버퍼 하나를 유지하고, 인코더를 누적 후 forward pass 마지막에 한 번만 commit.

```
현재: [Op1: CB→commit→wait] [Op2: CB→commit→wait] ... x395
목표: [CB 오픈] [Enc→Op1] [Enc→Op2] ... [CB commit → wait] x1
```

`CommandStream` 클래스가 이미 존재하므로(`command_stream.h` import 확인), 그 안에 "lazy commit" 모드 추가가 적합합니다.

---

### 🔴 Priority 2: Float16 가중치 + 연산 (B2 — 예상 ~2x)

**현재 문제**: 모델 로더가 f16 → f32 업캐스트. M4 기준 f32 weight=~1.3GB → 메모리 대역폭 77 tok/s 상한.

**개선 방향**:
- `QwenModelLoader`에서 f16 가중치를 그대로 `MTLResourceStorageModePrivate` 버퍼에 적재
- `matmul_f32_f16rhs_*` 커널(이미 존재함 — `LibraryHasRequiredRuntimeKernels` 체크 목록 참조)을 기본으로 사용
- Activation은 float32 유지 가능 (정확도 보존), 가중치만 f16으로 충분

---

### 🟠 Priority 3: Private Buffer for Weights (B5 연계 — 예상 5~10%)

**현재 문제**: 가중치가 `MTLResourceStorageModeShared` → CPU/GPU 공유 → Apple Silicon에서는 실질적 성능 차이 없음. 그러나 `StorageModePrivate`는 GPU 캐시 친화성 향상.

**개선 방향**:
- `MetalBuffer::CreatePrivateInitialized`가 **이미 구현되어 있음** (staging → blit copy 방식)
- 모델 로더에서 가중치 버퍼를 `CreatePrivate`로 생성하도록 변경

---

### 🟠 Priority 4: Per-Step 버퍼 재사용 (B5 — 예상 10~20%)

**현재 문제**: `UploadGpuTokenIds`와 `CreateGpuFloatTensor`가 decode 루프에서 매 스텝 `MetalBuffer::CreateShared` 호출 (`infer.mm` 614-626행 참조).

```objective-c++ name=Mac/gpu/infer.mm url=https://github.com/SongWooYoung/SoC/blob/41780f5ce139f56cd4fbda72b758a23737478c03/Mac/gpu/infer.mm#L614-L628
// 매 step마다 새로운 MetalBuffer 생성 → Metal driver 비용 반복
const soc::gpu::DeviceTensor decode_tensor = UploadGpuTokenIds(context,
                                                               {running_token_ids.back()}, ...);
const soc::gpu::DeviceTensor decode_hidden = CreateGpuFloatTensor(context,
                                                                  {1, ...}, ...);
```

**개선 방향**: `GenerationContext`에 decode용 `decode_token_buffer`, `decode_hidden_buffer`를 멤버로 사전 할당하고 재사용. `BufferArena`를 활용하거나 `MetalBuffer`를 한 번만 생성.

---

### 🟡 Priority 5: SIMD Group 리덕션 커널 (B3 — 예상 5~10x per kernel)

**현재 문제**: `rms_norm_f32_rowwise`와 `softmax_f32_rowwise`가 단일 스레드로 1024 원소 순회.

**개선 방향**: `simd_sum()` / `simd_max()` 사용한 32-thread 협력 리덕션으로 교체. `rms_norm_f32_rowwise_simd`는 이미 커널 목록에 있으므로(`LibraryHasRequiredRuntimeKernels`) 실제 디스패치 경로만 연결 필요.

---

### 🟡 Priority 6: Tile Size 확장 (B6 — 예상 10~20%)

**현재 문제**: `tile_rows=4, tile_columns=32, inner_tile=16` → threadgroup 메모리 9.2KB 사용 (32KB 중 29%).

**개선 방향**: `tile_rows=8, tile_columns=64, inner_tile=32` 로 증가 → 동일 threadgroup 메모리 내에서 2~4배 더 많은 데이터 재사용. `MatMulParams`의 `preferred_tile_*` 필드가 이미 준비되어 있어 실험적으로 조정 가능.

---

### 요약: 개선 우선순위

```
[즉시 구현, 최대 효과]
1. Command Buffer Batching    → ~395 CB/token → 1~3으로 (2~5x)
2. Float16 가중치 파이프라인  → 메모리 대역폭 절반 (2x)

[중기 구현]
3. Per-step 버퍼 재사용       → 매 스텝 Metal driver 호출 제거
4. Private 버퍼 가중치        → GPU 전용 캐시 효율화
5. SIMD 리덕션 커널 연결      → rms_norm/softmax 병렬화

[최적화 마무리]
6. MatMul 타일 크기 튜닝      → 32KB threadgroup 완전 활용
```

현재 예상 decode 속도 ~10~30 tok/s에서, Priority 1+2만 구현해도 M4 기준 **100+ tok/s** 목표 달성이 가능합니다. `CommandStream`과 `matmul_f32_f16rhs_*` 커널이 이미 존재하므로 연결 비용이 낮습니다.