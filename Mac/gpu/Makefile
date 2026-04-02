CXX := clang++

CXXFLAGS := -O2 -std=c++17 -Wall -Wextra -Wpedantic -MMD -MP -I. -Iinclude -I../cpu
OBJCXXFLAGS := $(CXXFLAGS) -fobjc-arc
CPU_CXXFLAGS := $(CXXFLAGS) -I../cpu
CPU_OBJCXXFLAGS := $(OBJCXXFLAGS) -I../cpu
LDFLAGS := -framework Foundation -framework Metal

SRC_DIR := src
MODEL_DIR := models
TEST_DIR := test
SHADER_DIR := shaders
BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj
BIN_DIR := $(BUILD_DIR)/bin
AIR_DIR := $(BUILD_DIR)/air
SHADER_BUILD_DIR := $(BUILD_DIR)/shaders
REPORT_DIR := reports
QUICK_REPORT_DIR := $(REPORT_DIR)/quick
METALLIB := $(SHADER_BUILD_DIR)/gpu.metallib
SHADER_ENTRY := $(SHADER_DIR)/gpu_kernels.metal
MAIN_TARGET := gpu_bootstrap
INFER_TARGET := gpu_infer
BOOTSTRAP_TEST_TARGET := test_metal_bootstrap
RMSNORM_TEST_TARGET := test_rms_norm
LINEAR_TEST_TARGET := test_linear
ASSET_TEST_TARGET := test_asset_loader
PHASE3_TEST_TARGET := test_phase3_kernels
MODULE_TEST_TARGET := test_qwen3_modules
RUNTIME_TEST_TARGET := test_generation_context
INTEGRATION_TEST_TARGET := test_real_bundle_regression
QWEN3_5_SCAFFOLD_TEST_TARGET := test_qwen3_5_scaffold
MATMUL_BENCH_TARGET := bench_matmul
BASELINE_BENCH_TARGET := bench_baseline
PHASE1_TEST_TARGET := test_phase1_batching
MAIN_BIN := $(BIN_DIR)/$(MAIN_TARGET)
INFER_BIN := $(BIN_DIR)/$(INFER_TARGET)
BOOTSTRAP_TEST_BIN := $(BIN_DIR)/$(BOOTSTRAP_TEST_TARGET)
RMSNORM_TEST_BIN := $(BIN_DIR)/$(RMSNORM_TEST_TARGET)
LINEAR_TEST_BIN := $(BIN_DIR)/$(LINEAR_TEST_TARGET)
ASSET_TEST_BIN := $(BIN_DIR)/$(ASSET_TEST_TARGET)
PHASE3_TEST_BIN := $(BIN_DIR)/$(PHASE3_TEST_TARGET)
MODULE_TEST_BIN := $(BIN_DIR)/$(MODULE_TEST_TARGET)
RUNTIME_TEST_BIN := $(BIN_DIR)/$(RUNTIME_TEST_TARGET)
INTEGRATION_TEST_BIN := $(BIN_DIR)/$(INTEGRATION_TEST_TARGET)
QWEN3_5_SCAFFOLD_TEST_BIN := $(BIN_DIR)/$(QWEN3_5_SCAFFOLD_TEST_TARGET)
MATMUL_BENCH_BIN := $(BIN_DIR)/$(MATMUL_BENCH_TARGET)
BASELINE_BENCH_BIN := $(BIN_DIR)/$(BASELINE_BENCH_TARGET)
PHASE1_TEST_BIN := $(BIN_DIR)/$(PHASE1_TEST_TARGET)
REAL_QWEN3_MANIFEST := $(abspath ../../models/cpp/qwen3-0.6b/manifest.json)
REAL_QWEN3_F16_MANIFEST := $(abspath ../../models/cpp/qwen3-0.6b-f16/manifest.json)
REAL_QWEN3_Q4_DECODE_MANIFEST := $(abspath ../../models/cpp/qwen3-0.6b-q4-decode/manifest.json)
SWEEP_SCRIPT := tools/run_real_bundle_sweep.sh
REAL_BUNDLE_REPORT := $(abspath $(REPORT_DIR)/test_real_bundle_regression_report.md)
SWEEP_SUMMARY := $(abspath $(REPORT_DIR)/test_real_bundle_sweep_summary.md)
SWEEP_REPORT_DIR := $(abspath $(REPORT_DIR)/sweep_cases)
QUICK_REAL_BUNDLE_REPORT := $(abspath $(QUICK_REPORT_DIR)/test_real_bundle_regression_report.md)
QUICK_SWEEP_SUMMARY := $(abspath $(QUICK_REPORT_DIR)/test_real_bundle_sweep_summary.md)
QUICK_SWEEP_REPORT_DIR := $(abspath $(QUICK_REPORT_DIR)/sweep_cases)
DECODE_STEADY_OUTPUT_DIR := $(abspath $(REPORT_DIR)/decode_steady_state)
DECODE_STEADY_REPORT := $(abspath $(REPORT_DIR)/decode_steady_state.md)
DECODE_STEADY_Q4_OUTPUT_DIR := $(abspath $(REPORT_DIR)/decode_steady_state_q4_safe_batch)
DECODE_STEADY_Q4_REPORT := $(abspath $(REPORT_DIR)/decode_steady_state_q4_safe_batch.md)
LAYER_BENCH_OUTPUT_DIR := $(abspath $(REPORT_DIR)/layer_split_benchmark)
LAYER_BENCH_REPORT := $(abspath $(REPORT_DIR)/layer_split_benchmark.md)
FULL_GPU_VS_PYTORCH_OUTPUT_DIR := $(abspath $(REPORT_DIR)/full_gpu_vs_pytorch)
FULL_GPU_VS_PYTORCH_REPORT := $(abspath $(REPORT_DIR)/full_gpu_vs_pytorch.md)
FULL_GPU_VS_PYTORCH_PREFILL_BATCH_LONG_OUTPUT_DIR := $(abspath $(REPORT_DIR)/full_gpu_vs_pytorch_prefill_attention_batch_long_prompt)
FULL_GPU_VS_PYTORCH_PREFILL_BATCH_LONG_REPORT := $(abspath $(REPORT_DIR)/full_gpu_vs_pytorch_prefill_attention_batch_long_prompt.md)
FULL_GPU_VS_PYTORCH_Q4_OUTPUT_DIR := $(abspath $(REPORT_DIR)/full_gpu_vs_pytorch_q4_decode_safe_batch)
FULL_GPU_VS_PYTORCH_Q4_REPORT := $(abspath $(REPORT_DIR)/full_gpu_vs_pytorch_q4_decode_safe_batch.md)
FULL_GPU_VS_PYTORCH_Q4_BLOCKPREP_OUTPUT_DIR := $(abspath $(REPORT_DIR)/full_gpu_vs_pytorch_q4_decode_safe_blockprep)
FULL_GPU_VS_PYTORCH_Q4_BLOCKPREP_REPORT := $(abspath $(REPORT_DIR)/full_gpu_vs_pytorch_q4_decode_safe_blockprep.md)
DECODE_HANG_REPORT := $(abspath $(REPORT_DIR)/decode_hang_check_q4_decode_safe_batch.json)
DECODE_HANG_BLOCKPREP_REPORT := $(abspath $(REPORT_DIR)/decode_hang_check_q4_decode_safe_blockprep.json)
HF_QWEN3_MODEL_DIR := $(abspath ../../models/hf/qwen3-0.6b)
HF_QWEN3_5_4B_MODEL_DIR := $(abspath ../../models/raw/qwen3_5-4b)
QWEN3_5_4B_MANIFEST := $(abspath ../../models/cpp/qwen3_5-4b/manifest.json)
QWEN3_5_PROMPT_SUITE_INPUT := $(abspath benchmark_data/qwen3_5_4b/input_list.jsonl)
QWEN3_5_PROMPT_SUITE_OUTPUT_DIR := $(abspath $(REPORT_DIR)/qwen3_5_4b_prompt_suite)
QWEN3_5_PROMPT_SUITE_TAG ?= current
QWEN3_5_PROMPT_SUITE_OUTPUT := $(abspath $(QWEN3_5_PROMPT_SUITE_OUTPUT_DIR)/output_list_$(QWEN3_5_PROMPT_SUITE_TAG).jsonl)
QWEN3_5_PROMPT_SUITE_OUTPUT_JSON := $(abspath $(QWEN3_5_PROMPT_SUITE_OUTPUT_DIR)/output_list_$(QWEN3_5_PROMPT_SUITE_TAG).pretty.json)
QWEN3_5_PROMPT_SUITE_OUTPUT_MD := $(abspath $(QWEN3_5_PROMPT_SUITE_OUTPUT_DIR)/output_list_$(QWEN3_5_PROMPT_SUITE_TAG).pretty.md)
QWEN3_5_PROMPT_SUITE_MAX_NEW_TOKENS ?= 0
QWEN3_5_PROMPT_SUITE_LIMIT ?= 0
QWEN3_5_PROMPT_SUITE_DTYPE ?= float16
QWEN3_5_PROMPT_SUITE_CPP_ENV ?= SOC_GPU_ENABLE_EXPERIMENTAL_QWEN3_5_PROJECTION_GPU=1
QWEN3_5_OUTPUT_EVAL_TAG ?= current
QWEN3_5_OUTPUT_EVAL_OUTPUT := $(abspath $(REPORT_DIR)/output_eval_$(QWEN3_5_OUTPUT_EVAL_TAG).json)
QWEN3_5_OUTPUT_EVAL_MAX_NEW_TOKENS ?= 64
QWEN3_5_OUTPUT_EVAL_LIMIT ?= 0
QWEN3_5_OUTPUT_EVAL_DTYPE ?= float16
QWEN3_5_OUTPUT_EVAL_TEMPERATURE ?= 0
QWEN3_5_OUTPUT_EVAL_TOP_K ?= 1
QWEN3_5_OUTPUT_EVAL_CPP_ENV ?= SOC_GPU_ENABLE_EXPERIMENTAL_QWEN3_5_PROJECTION_GPU=1
QWEN3_5_BOUNDARY_EVAL_TAG ?= current
QWEN3_5_BOUNDARY_EVAL_OUTPUT := $(abspath $(REPORT_DIR)/output_eval_boundary_$(QWEN3_5_BOUNDARY_EVAL_TAG).json)
QWEN3_5_BOUNDARY_EVAL_LIMIT ?= 0
QWEN3_5_BOUNDARY_EVAL_CPP_ENV ?= SOC_GPU_ENABLE_EXPERIMENTAL_QWEN3_5_PROJECTION_GPU=1
LAYER_BENCH_PROMPT ?= Hello world
LAYER_BENCH_MAX_NEW_TOKENS ?= 64
FULL_GPU_VS_PYTORCH_PROMPT ?= Hello world
FULL_GPU_VS_PYTORCH_LONG_PROMPT ?= Summarize the design tradeoffs between low-level Metal command scheduling, prompt cache restore, and bounded decode batching on Apple Silicon in two concise paragraphs. Then explain why a long prefill prompt stresses command-buffer and encoder overhead more clearly than a short greeting prompt.
FULL_GPU_VS_PYTORCH_MAX_NEW_TOKENS ?= 64
FULL_GPU_VS_PYTORCH_GPU_RUNS ?= 3
FULL_GPU_VS_PYTORCH_PYTORCH_RUNS ?= 3
FULL_GPU_VS_PYTORCH_PYTORCH_WARMUP ?= 1
FULL_GPU_VS_PYTORCH_PYTORCH_DTYPE ?= float32
DECODE_STEADY_MAX_NEW_TOKENS ?= 32
DECODE_STEADY_RUNS ?= 3
ACTIVE_REAL_BUNDLE_REPORT ?= $(REAL_BUNDLE_REPORT)
ACTIVE_SWEEP_SUMMARY ?= $(SWEEP_SUMMARY)
ACTIVE_SWEEP_REPORT_DIR ?= $(SWEEP_REPORT_DIR)
REAL_BUNDLE_MAX_NEW_TOKENS ?= 8
SWEEP_MAX_NEW_TOKENS ?= 8
PYTHON_BIN ?= $(if $(wildcard ../../.venv/bin/python),$(abspath ../../.venv/bin/python),python3)

METAL := xcrun -sdk macosx metal
METALLIB_TOOL := xcrun -sdk macosx metallib

CPP_SRCS := $(shell find $(SRC_DIR) $(MODEL_DIR) \( -path '$(MODEL_DIR)/*/test/*' -o -path '$(MODEL_DIR)/*/test' \) -prune -o -type f -name '*.cpp' -print 2>/dev/null)
MM_SRCS := $(shell find $(SRC_DIR) $(MODEL_DIR) \( -path '$(MODEL_DIR)/*/test/*' -o -path '$(MODEL_DIR)/*/test' \) -prune -o -type f -name '*.mm' -print 2>/dev/null)
CPU_CPP_SRCS := $(wildcard ../cpu/src/*.cpp)

CPP_OBJS := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(CPP_SRCS))
MM_OBJS := $(patsubst %.mm,$(OBJ_DIR)/%.o,$(MM_SRCS))
CPU_CPP_OBJS := $(patsubst ../cpu/src/%.cpp,$(OBJ_DIR)/cpu/%.o,$(CPU_CPP_SRCS))
MAIN_OBJ := $(OBJ_DIR)/main.o
INFER_OBJ := $(OBJ_DIR)/infer.o
BOOTSTRAP_TEST_OBJ := $(OBJ_DIR)/test/metal/test_metal_bootstrap.o
RMSNORM_TEST_OBJ := $(OBJ_DIR)/test/kernel/test_rms_norm.o
LINEAR_TEST_OBJ := $(OBJ_DIR)/test/kernel/test_linear.o
ASSET_TEST_OBJ := $(OBJ_DIR)/test/asset/test_asset_loader.o
PHASE3_TEST_OBJ := $(OBJ_DIR)/test/kernel/test_phase3_kernels.o
MODULE_TEST_OBJ := $(OBJ_DIR)/models/qwen3/test/test_qwen3_modules.o
RUNTIME_TEST_OBJ := $(OBJ_DIR)/test/runtime/test_generation_context.o
INTEGRATION_TEST_OBJ := $(OBJ_DIR)/test/integration/test_real_bundle_regression.o
QWEN3_5_SCAFFOLD_TEST_OBJ := $(OBJ_DIR)/models/qwen3_5/test/test_qwen3_5_scaffold.o
MATMUL_BENCH_OBJ := $(OBJ_DIR)/benchmark/bench_matmul.o
BASELINE_BENCH_OBJ := $(OBJ_DIR)/test/gpu_optimization/bench_baseline.o
PHASE1_TEST_OBJ := $(OBJ_DIR)/test/gpu_optimization/test_phase1_batching.o
COMMAND_STREAM_OBJ := $(OBJ_DIR)/src/metal/command_stream.o
AIR_FILE := $(AIR_DIR)/gpu_kernels.air
DEPFILES := $(CPP_OBJS:.o=.d) $(MM_OBJS:.o=.d) $(CPU_CPP_OBJS:.o=.d) $(MAIN_OBJ:.o=.d) $(BOOTSTRAP_TEST_OBJ:.o=.d) $(RMSNORM_TEST_OBJ:.o=.d) $(LINEAR_TEST_OBJ:.o=.d) $(ASSET_TEST_OBJ:.o=.d) $(PHASE3_TEST_OBJ:.o=.d) $(MODULE_TEST_OBJ:.o=.d) $(RUNTIME_TEST_OBJ:.o=.d) $(INTEGRATION_TEST_OBJ:.o=.d) $(QWEN3_5_SCAFFOLD_TEST_OBJ:.o=.d) $(MATMUL_BENCH_OBJ:.o=.d) $(BASELINE_BENCH_OBJ:.o=.d)

DEPFILES += $(INFER_OBJ:.o=.d)

.PHONY: all build-shaders build-main build-infer build-tests build-benchmarks bench-baseline check integration integration-real-bundle integration-real-bundle-f16 integration-sweep integration-sweep-quick run infer benchmark benchmark-layer-split benchmark-layer-split-f16 benchmark-full-gpu-vs-pytorch benchmark-full-gpu-vs-pytorch-prefill-attention-batch-long-prompt benchmark-full-gpu-vs-pytorch-q4-decode-safe-batch benchmark-full-gpu-vs-pytorch-q4-decode-safe-blockprep benchmark-decode-steady-state benchmark-decode-steady-state-q4-decode-safe-batch benchmark-qwen3_5-prompt-suite output-eval-qwen3_5 output-eval-qwen3_5-boundary check-decode-hang-q4-decode-safe-batch check-decode-hang-q4-decode-safe-blockprep regression real-bundle-regression real-bundle-regression-quick clean

all: build-main

build-shaders:
	@if xcrun -sdk macosx --find metal >/dev/null 2>&1 && xcrun -sdk macosx --find metallib >/dev/null 2>&1; then \
		$(MAKE) $(METALLIB); \
	else \
		echo "Skipping metallib build: Metal Toolchain is not installed"; \
	fi

build-main: $(MAIN_BIN)

build-infer: $(INFER_BIN)

build-tests: $(BOOTSTRAP_TEST_BIN) $(RMSNORM_TEST_BIN) $(LINEAR_TEST_BIN) $(ASSET_TEST_BIN) $(PHASE3_TEST_BIN) $(MODULE_TEST_BIN) $(RUNTIME_TEST_BIN) $(INTEGRATION_TEST_BIN) $(QWEN3_5_SCAFFOLD_TEST_BIN)

build-benchmarks: $(MATMUL_BENCH_BIN) $(BASELINE_BENCH_BIN)

bench-baseline: $(BASELINE_BENCH_BIN)
	$(BASELINE_BENCH_BIN)

test-phase1: $(PHASE1_TEST_BIN)
	$(PHASE1_TEST_BIN)

check: $(BOOTSTRAP_TEST_BIN) $(RMSNORM_TEST_BIN) $(LINEAR_TEST_BIN) $(ASSET_TEST_BIN) $(PHASE3_TEST_BIN) $(MODULE_TEST_BIN) $(RUNTIME_TEST_BIN) $(QWEN3_5_SCAFFOLD_TEST_BIN)
	$(BOOTSTRAP_TEST_BIN)
	$(RMSNORM_TEST_BIN)
	$(LINEAR_TEST_BIN)
	$(ASSET_TEST_BIN)
	$(PHASE3_TEST_BIN)
	$(MODULE_TEST_BIN)
	$(RUNTIME_TEST_BIN)
	$(QWEN3_5_SCAFFOLD_TEST_BIN)

integration: $(INTEGRATION_TEST_BIN)
	$(INTEGRATION_TEST_BIN)

integration-real-bundle: $(INTEGRATION_TEST_BIN)
	@test -f $(REAL_QWEN3_MANIFEST) || (echo "missing real export bundle: $(REAL_QWEN3_MANIFEST)" && exit 1)
	@echo "Running real-bundle regression with REAL_BUNDLE_MAX_NEW_TOKENS=$(REAL_BUNDLE_MAX_NEW_TOKENS)"
	SOC_QWEN3_MANIFEST=$(REAL_QWEN3_MANIFEST) \
	SOC_QWEN_MAX_NEW_TOKENS=$(REAL_BUNDLE_MAX_NEW_TOKENS) \
	SOC_GPU_REPORT_PATH=$(ACTIVE_REAL_BUNDLE_REPORT) \
	$(INTEGRATION_TEST_BIN)

integration-real-bundle-f16: $(INTEGRATION_TEST_BIN)
	@test -f $(REAL_QWEN3_F16_MANIFEST) || (echo "missing float16 real export bundle: $(REAL_QWEN3_F16_MANIFEST)" && exit 1)
	@echo "Running float16 real-bundle regression with REAL_BUNDLE_MAX_NEW_TOKENS=$(REAL_BUNDLE_MAX_NEW_TOKENS)"
	SOC_QWEN3_MANIFEST=$(REAL_QWEN3_F16_MANIFEST) \
	SOC_QWEN_MAX_NEW_TOKENS=$(REAL_BUNDLE_MAX_NEW_TOKENS) \
	SOC_GPU_REPORT_PATH=$(ACTIVE_REAL_BUNDLE_REPORT) \
	$(INTEGRATION_TEST_BIN)

integration-sweep: $(INTEGRATION_TEST_BIN)
	@test -f $(REAL_QWEN3_MANIFEST) || (echo "missing real export bundle: $(REAL_QWEN3_MANIFEST)" && exit 1)
	@echo "Running prompt sweep with SWEEP_MAX_NEW_TOKENS=$(SWEEP_MAX_NEW_TOKENS)"
	SOC_QWEN3_MANIFEST=$(REAL_QWEN3_MANIFEST) \
	SOC_QWEN_MAX_NEW_TOKENS=$(SWEEP_MAX_NEW_TOKENS) \
	SOC_GPU_SWEEP_SUMMARY=$(ACTIVE_SWEEP_SUMMARY) \
	SOC_GPU_SWEEP_REPORT_DIR=$(ACTIVE_SWEEP_REPORT_DIR) \
	$(SWEEP_SCRIPT)

integration-sweep-quick:
	$(MAKE) SWEEP_MAX_NEW_TOKENS=1 \
		ACTIVE_SWEEP_SUMMARY=$(QUICK_SWEEP_SUMMARY) \
		ACTIVE_SWEEP_REPORT_DIR=$(QUICK_SWEEP_REPORT_DIR) \
		integration-sweep

run: $(MAIN_BIN)
	$(MAIN_BIN)

infer: $(INFER_BIN)
	$(INFER_BIN) --help

benchmark: $(MATMUL_BENCH_BIN)
	$(MATMUL_BENCH_BIN)

benchmark-layer-split: $(INFER_BIN)
	@test -f $(REAL_QWEN3_MANIFEST) || (echo "missing real export bundle: $(REAL_QWEN3_MANIFEST)" && exit 1)
	$(PYTHON_BIN) tools/benchmark_layer_split.py \
		--infer-bin $(INFER_BIN) \
		--manifest $(REAL_QWEN3_MANIFEST) \
		--output-dir $(LAYER_BENCH_OUTPUT_DIR) \
		--report-path $(LAYER_BENCH_REPORT) \
		--prompt "$(LAYER_BENCH_PROMPT)" \
		--max-new-tokens $(LAYER_BENCH_MAX_NEW_TOKENS)

benchmark-layer-split-f16: $(INFER_BIN)
	@test -f $(REAL_QWEN3_F16_MANIFEST) || (echo "missing float16 real export bundle: $(REAL_QWEN3_F16_MANIFEST)" && exit 1)
	$(PYTHON_BIN) tools/benchmark_layer_split.py \
		--infer-bin $(INFER_BIN) \
		--manifest $(REAL_QWEN3_F16_MANIFEST) \
		--output-dir $(LAYER_BENCH_OUTPUT_DIR) \
		--report-path $(LAYER_BENCH_REPORT) \
		--prompt "$(LAYER_BENCH_PROMPT)" \
		--max-new-tokens $(LAYER_BENCH_MAX_NEW_TOKENS)

benchmark-full-gpu-vs-pytorch: $(INFER_BIN)
	@test -f $(REAL_QWEN3_MANIFEST) || (echo "missing real export bundle: $(REAL_QWEN3_MANIFEST)" && exit 1)
	@test -d $(HF_QWEN3_MODEL_DIR) || (echo "missing hf model dir: $(HF_QWEN3_MODEL_DIR)" && exit 1)
	$(PYTHON_BIN) tools/benchmark_full_gpu_vs_pytorch.py \
		--infer-bin $(INFER_BIN) \
		--manifest $(REAL_QWEN3_MANIFEST) \
		--hf-model $(HF_QWEN3_MODEL_DIR) \
		--output-dir $(FULL_GPU_VS_PYTORCH_OUTPUT_DIR) \
		--report-path $(FULL_GPU_VS_PYTORCH_REPORT) \
		--prompt "$(FULL_GPU_VS_PYTORCH_PROMPT)" \
		--max-new-tokens $(FULL_GPU_VS_PYTORCH_MAX_NEW_TOKENS) \
		--gpu-runs $(FULL_GPU_VS_PYTORCH_GPU_RUNS) \
		--pytorch-runs $(FULL_GPU_VS_PYTORCH_PYTORCH_RUNS) \
		--pytorch-warmup $(FULL_GPU_VS_PYTORCH_PYTORCH_WARMUP) \
		--python-bin $(PYTHON_BIN) \
		--pytorch-script ../benchmark_pytorch.py \
		--pytorch-dtype $(FULL_GPU_VS_PYTORCH_PYTORCH_DTYPE)

benchmark-qwen3_5-prompt-suite: $(INFER_BIN)
	@test -f $(QWEN3_5_4B_MANIFEST) || (echo "missing qwen3_5 export bundle: $(QWEN3_5_4B_MANIFEST)" && exit 1)
	@test -d $(HF_QWEN3_5_4B_MODEL_DIR) || (echo "missing qwen3_5 hf model dir: $(HF_QWEN3_5_4B_MODEL_DIR)" && exit 1)
	@test -f $(QWEN3_5_PROMPT_SUITE_INPUT) || (echo "missing prompt suite input: $(QWEN3_5_PROMPT_SUITE_INPUT)" && exit 1)
	$(PYTHON_BIN) tools/benchmark_qwen3_5_prompt_suite.py \
		--input-list $(QWEN3_5_PROMPT_SUITE_INPUT) \
		--output-jsonl $(QWEN3_5_PROMPT_SUITE_OUTPUT) \
		--output-json $(QWEN3_5_PROMPT_SUITE_OUTPUT_JSON) \
		--output-md $(QWEN3_5_PROMPT_SUITE_OUTPUT_MD) \
		--infer-bin $(INFER_BIN) \
		--manifest $(QWEN3_5_4B_MANIFEST) \
		--hf-model $(HF_QWEN3_5_4B_MODEL_DIR) \
		--model-type qwen3_5 \
		--dtype $(QWEN3_5_PROMPT_SUITE_DTYPE) \
		--max-new-tokens $(QWEN3_5_PROMPT_SUITE_MAX_NEW_TOKENS) \
		--tag $(QWEN3_5_PROMPT_SUITE_TAG) \
		--limit $(QWEN3_5_PROMPT_SUITE_LIMIT) \
		--cpp-env $(QWEN3_5_PROMPT_SUITE_CPP_ENV)

output-eval-qwen3_5: $(INFER_BIN)
	@test -f $(QWEN3_5_4B_MANIFEST) || (echo "missing qwen3_5 export bundle: $(QWEN3_5_4B_MANIFEST)" && exit 1)
	@test -d $(HF_QWEN3_5_4B_MODEL_DIR) || (echo "missing qwen3_5 hf model dir: $(HF_QWEN3_5_4B_MODEL_DIR)" && exit 1)
	@test -f $(QWEN3_5_PROMPT_SUITE_INPUT) || (echo "missing prompt suite input: $(QWEN3_5_PROMPT_SUITE_INPUT)" && exit 1)
	$(PYTHON_BIN) test/output_eval_qwen3_5.py \
		--input-list $(QWEN3_5_PROMPT_SUITE_INPUT) \
		--output-json $(QWEN3_5_OUTPUT_EVAL_OUTPUT) \
		--infer-bin $(INFER_BIN) \
		--manifest $(QWEN3_5_4B_MANIFEST) \
		--hf-model $(HF_QWEN3_5_4B_MODEL_DIR) \
		--model-type qwen3_5 \
		--dtype $(QWEN3_5_OUTPUT_EVAL_DTYPE) \
		--max-new-tokens $(QWEN3_5_OUTPUT_EVAL_MAX_NEW_TOKENS) \
		--max-seq-len 256 \
		--temperature $(QWEN3_5_OUTPUT_EVAL_TEMPERATURE) \
		--top-k $(QWEN3_5_OUTPUT_EVAL_TOP_K) \
		--tag $(QWEN3_5_OUTPUT_EVAL_TAG) \
		--limit $(QWEN3_5_OUTPUT_EVAL_LIMIT) \
		--cpp-env $(QWEN3_5_OUTPUT_EVAL_CPP_ENV)

output-eval-qwen3_5-boundary: $(INFER_BIN)
	@test -f $(QWEN3_5_4B_MANIFEST) || (echo "missing qwen3_5 export bundle: $(QWEN3_5_4B_MANIFEST)" && exit 1)
	@test -d $(HF_QWEN3_5_4B_MODEL_DIR) || (echo "missing qwen3_5 hf model dir: $(HF_QWEN3_5_4B_MODEL_DIR)" && exit 1)
	@test -f $(QWEN3_5_PROMPT_SUITE_INPUT) || (echo "missing prompt suite input: $(QWEN3_5_PROMPT_SUITE_INPUT)" && exit 1)
	$(PYTHON_BIN) test/output_eval_qwen3_5_boundary.py \
		--input-list $(QWEN3_5_PROMPT_SUITE_INPUT) \
		--output-json $(QWEN3_5_BOUNDARY_EVAL_OUTPUT) \
		--infer-bin $(INFER_BIN) \
		--manifest $(QWEN3_5_4B_MANIFEST) \
		--hf-model $(HF_QWEN3_5_4B_MODEL_DIR) \
		--hf-probe-script tools/qwen3_5_transformers_parity_probe.py \
		--tag $(QWEN3_5_BOUNDARY_EVAL_TAG) \
		--limit $(QWEN3_5_BOUNDARY_EVAL_LIMIT) \
		--cpp-env $(QWEN3_5_BOUNDARY_EVAL_CPP_ENV)

benchmark-full-gpu-vs-pytorch-prefill-attention-batch-long-prompt: $(INFER_BIN)
	@test -f $(REAL_QWEN3_MANIFEST) || (echo "missing real export bundle: $(REAL_QWEN3_MANIFEST)" && exit 1)
	@test -d $(HF_QWEN3_MODEL_DIR) || (echo "missing hf model dir: $(HF_QWEN3_MODEL_DIR)" && exit 1)
	$(PYTHON_BIN) tools/benchmark_full_gpu_vs_pytorch.py \
		--infer-bin $(INFER_BIN) \
		--manifest $(REAL_QWEN3_MANIFEST) \
		--hf-model $(HF_QWEN3_MODEL_DIR) \
		--output-dir $(FULL_GPU_VS_PYTORCH_PREFILL_BATCH_LONG_OUTPUT_DIR) \
		--report-path $(FULL_GPU_VS_PYTORCH_PREFILL_BATCH_LONG_REPORT) \
		--prompt "$(FULL_GPU_VS_PYTORCH_LONG_PROMPT)" \
		--max-new-tokens $(FULL_GPU_VS_PYTORCH_MAX_NEW_TOKENS) \
		--gpu-runs $(FULL_GPU_VS_PYTORCH_GPU_RUNS) \
		--pytorch-runs $(FULL_GPU_VS_PYTORCH_PYTORCH_RUNS) \
		--pytorch-warmup $(FULL_GPU_VS_PYTORCH_PYTORCH_WARMUP) \
		--python-bin $(PYTHON_BIN) \
		--pytorch-script ../benchmark_pytorch.py \
		--pytorch-dtype $(FULL_GPU_VS_PYTORCH_PYTORCH_DTYPE) \
		--gpu-env SOC_GPU_ENABLE_EXPERIMENTAL_PREFILL_ATTENTION_BATCH=1

benchmark-full-gpu-vs-pytorch-q4-decode-safe-batch: $(INFER_BIN)
	@test -f $(REAL_QWEN3_Q4_DECODE_MANIFEST) || (echo "missing q4 decode export bundle: $(REAL_QWEN3_Q4_DECODE_MANIFEST)" && exit 1)
	@test -d $(HF_QWEN3_MODEL_DIR) || (echo "missing hf model dir: $(HF_QWEN3_MODEL_DIR)" && exit 1)
	$(PYTHON_BIN) tools/benchmark_full_gpu_vs_pytorch.py \
		--infer-bin $(INFER_BIN) \
		--manifest $(REAL_QWEN3_Q4_DECODE_MANIFEST) \
		--hf-model $(HF_QWEN3_MODEL_DIR) \
		--output-dir $(FULL_GPU_VS_PYTORCH_Q4_OUTPUT_DIR) \
		--report-path $(FULL_GPU_VS_PYTORCH_Q4_REPORT) \
		--prompt "$(FULL_GPU_VS_PYTORCH_PROMPT)" \
		--max-new-tokens $(FULL_GPU_VS_PYTORCH_MAX_NEW_TOKENS) \
		--gpu-runs $(FULL_GPU_VS_PYTORCH_GPU_RUNS) \
		--pytorch-runs $(FULL_GPU_VS_PYTORCH_PYTORCH_RUNS) \
		--pytorch-warmup $(FULL_GPU_VS_PYTORCH_PYTORCH_WARMUP) \
		--python-bin $(PYTHON_BIN) \
		--pytorch-script ../benchmark_pytorch.py \
		--pytorch-dtype $(FULL_GPU_VS_PYTORCH_PYTORCH_DTYPE) \
		--gpu-env SOC_GPU_ENABLE_EXPERIMENTAL_Q4_DECODE=1 \
		--gpu-env SOC_GPU_ENABLE_EXPERIMENTAL_SAFE_DECODE_BATCH=1

benchmark-full-gpu-vs-pytorch-q4-decode-safe-blockprep: $(INFER_BIN)
	@test -f $(REAL_QWEN3_Q4_DECODE_MANIFEST) || (echo "missing q4 decode export bundle: $(REAL_QWEN3_Q4_DECODE_MANIFEST)" && exit 1)
	@test -d $(HF_QWEN3_MODEL_DIR) || (echo "missing hf model dir: $(HF_QWEN3_MODEL_DIR)" && exit 1)
	$(PYTHON_BIN) tools/benchmark_full_gpu_vs_pytorch.py \
		--infer-bin $(INFER_BIN) \
		--manifest $(REAL_QWEN3_Q4_DECODE_MANIFEST) \
		--hf-model $(HF_QWEN3_MODEL_DIR) \
		--output-dir $(FULL_GPU_VS_PYTORCH_Q4_BLOCKPREP_OUTPUT_DIR) \
		--report-path $(FULL_GPU_VS_PYTORCH_Q4_BLOCKPREP_REPORT) \
		--prompt "$(FULL_GPU_VS_PYTORCH_PROMPT)" \
		--max-new-tokens $(FULL_GPU_VS_PYTORCH_MAX_NEW_TOKENS) \
		--gpu-runs $(FULL_GPU_VS_PYTORCH_GPU_RUNS) \
		--pytorch-runs $(FULL_GPU_VS_PYTORCH_PYTORCH_RUNS) \
		--pytorch-warmup $(FULL_GPU_VS_PYTORCH_PYTORCH_WARMUP) \
		--python-bin $(PYTHON_BIN) \
		--pytorch-script ../benchmark_pytorch.py \
		--pytorch-dtype $(FULL_GPU_VS_PYTORCH_PYTORCH_DTYPE) \
		--gpu-env SOC_GPU_ENABLE_EXPERIMENTAL_Q4_DECODE=1 \
		--gpu-env SOC_GPU_ENABLE_EXPERIMENTAL_SAFE_DECODE_BATCH=1 \
		--gpu-env SOC_GPU_ENABLE_EXPERIMENTAL_BLOCK_PREP_BATCH=1

benchmark-decode-steady-state: $(INFER_BIN)
	@test -f $(REAL_QWEN3_MANIFEST) || (echo "missing real export bundle: $(REAL_QWEN3_MANIFEST)" && exit 1)
	$(PYTHON_BIN) tools/benchmark_decode_steady_state.py \
		--infer-bin $(INFER_BIN) \
		--manifest $(REAL_QWEN3_MANIFEST) \
		--output-dir $(DECODE_STEADY_OUTPUT_DIR) \
		--report-path $(DECODE_STEADY_REPORT) \
		--prompt "$(FULL_GPU_VS_PYTORCH_PROMPT)" \
		--max-new-tokens $(DECODE_STEADY_MAX_NEW_TOKENS) \
		--runs $(DECODE_STEADY_RUNS)

benchmark-decode-steady-state-q4-decode-safe-batch: $(INFER_BIN)
	@test -f $(REAL_QWEN3_Q4_DECODE_MANIFEST) || (echo "missing q4 decode export bundle: $(REAL_QWEN3_Q4_DECODE_MANIFEST)" && exit 1)
	$(PYTHON_BIN) tools/benchmark_decode_steady_state.py \
		--infer-bin $(INFER_BIN) \
		--manifest $(REAL_QWEN3_Q4_DECODE_MANIFEST) \
		--output-dir $(DECODE_STEADY_Q4_OUTPUT_DIR) \
		--report-path $(DECODE_STEADY_Q4_REPORT) \
		--prompt "$(FULL_GPU_VS_PYTORCH_PROMPT)" \
		--max-new-tokens $(DECODE_STEADY_MAX_NEW_TOKENS) \
		--runs $(DECODE_STEADY_RUNS) \
		--gpu-env SOC_GPU_ENABLE_EXPERIMENTAL_Q4_DECODE=1 \
		--gpu-env SOC_GPU_ENABLE_EXPERIMENTAL_SAFE_DECODE_BATCH=1

check-decode-hang-q4-decode-safe-batch: $(INFER_BIN)
	@test -f $(REAL_QWEN3_Q4_DECODE_MANIFEST) || (echo "missing q4 decode export bundle: $(REAL_QWEN3_Q4_DECODE_MANIFEST)" && exit 1)
	$(PYTHON_BIN) tools/check_decode_hang.py \
		--infer-bin $(INFER_BIN) \
		--manifest $(REAL_QWEN3_Q4_DECODE_MANIFEST) \
		--prompt "$(FULL_GPU_VS_PYTORCH_PROMPT)" \
		--max-new-tokens 8 \
		--timeout-seconds 30 \
		--output $(DECODE_HANG_REPORT) \
		--gpu-env SOC_GPU_ENABLE_EXPERIMENTAL_Q4_DECODE=1 \
		--gpu-env SOC_GPU_ENABLE_EXPERIMENTAL_SAFE_DECODE_BATCH=1

check-decode-hang-q4-decode-safe-blockprep: $(INFER_BIN)
	@test -f $(REAL_QWEN3_Q4_DECODE_MANIFEST) || (echo "missing q4 decode export bundle: $(REAL_QWEN3_Q4_DECODE_MANIFEST)" && exit 1)
	$(PYTHON_BIN) tools/check_decode_hang.py \
		--infer-bin $(INFER_BIN) \
		--manifest $(REAL_QWEN3_Q4_DECODE_MANIFEST) \
		--prompt "$(FULL_GPU_VS_PYTORCH_PROMPT)" \
		--max-new-tokens 8 \
		--timeout-seconds 30 \
		--output $(DECODE_HANG_BLOCKPREP_REPORT) \
		--gpu-env SOC_GPU_ENABLE_EXPERIMENTAL_Q4_DECODE=1 \
		--gpu-env SOC_GPU_ENABLE_EXPERIMENTAL_SAFE_DECODE_BATCH=1 \
		--gpu-env SOC_GPU_ENABLE_EXPERIMENTAL_BLOCK_PREP_BATCH=1

regression:
	$(MAKE) clean
	$(MAKE) build-shaders
	$(MAKE) build-main
	$(MAKE) build-tests
	$(MAKE) check

real-bundle-regression:
	$(MAKE) build-shaders
	$(MAKE) build-tests
	$(MAKE) check
	$(MAKE) integration-real-bundle

real-bundle-regression-quick:
	$(MAKE) REAL_BUNDLE_MAX_NEW_TOKENS=1 \
		ACTIVE_REAL_BUNDLE_REPORT=$(QUICK_REAL_BUNDLE_REPORT) \
		integration-real-bundle

$(MAIN_BIN): $(CPP_OBJS) $(MM_OBJS) $(MAIN_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CPP_OBJS) $(MM_OBJS) $(MAIN_OBJ) -o $@ $(LDFLAGS)

$(INFER_BIN): $(CPP_OBJS) $(MM_OBJS) $(CPU_CPP_OBJS) $(INFER_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CPP_OBJS) $(MM_OBJS) $(CPU_CPP_OBJS) $(INFER_OBJ) -o $@ $(LDFLAGS)

$(BOOTSTRAP_TEST_BIN): $(CPP_OBJS) $(MM_OBJS) $(CPU_CPP_OBJS) $(BOOTSTRAP_TEST_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CPP_OBJS) $(MM_OBJS) $(CPU_CPP_OBJS) $(BOOTSTRAP_TEST_OBJ) -o $@ $(LDFLAGS)

$(RMSNORM_TEST_BIN): $(CPP_OBJS) $(MM_OBJS) $(CPU_CPP_OBJS) $(RMSNORM_TEST_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CPP_OBJS) $(MM_OBJS) $(CPU_CPP_OBJS) $(RMSNORM_TEST_OBJ) -o $@ $(LDFLAGS)

$(LINEAR_TEST_BIN): $(CPP_OBJS) $(MM_OBJS) $(CPU_CPP_OBJS) $(LINEAR_TEST_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CPP_OBJS) $(MM_OBJS) $(CPU_CPP_OBJS) $(LINEAR_TEST_OBJ) -o $@ $(LDFLAGS)

$(ASSET_TEST_BIN): $(CPP_OBJS) $(MM_OBJS) $(CPU_CPP_OBJS) $(ASSET_TEST_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CPP_OBJS) $(MM_OBJS) $(CPU_CPP_OBJS) $(ASSET_TEST_OBJ) -o $@ $(LDFLAGS)

$(PHASE3_TEST_BIN): $(CPP_OBJS) $(MM_OBJS) $(CPU_CPP_OBJS) $(PHASE3_TEST_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CPP_OBJS) $(MM_OBJS) $(CPU_CPP_OBJS) $(PHASE3_TEST_OBJ) -o $@ $(LDFLAGS)

$(MODULE_TEST_BIN): $(CPP_OBJS) $(MM_OBJS) $(CPU_CPP_OBJS) $(MODULE_TEST_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CPP_OBJS) $(MM_OBJS) $(CPU_CPP_OBJS) $(MODULE_TEST_OBJ) -o $@ $(LDFLAGS)

$(RUNTIME_TEST_BIN): $(CPP_OBJS) $(MM_OBJS) $(CPU_CPP_OBJS) $(RUNTIME_TEST_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CPP_OBJS) $(MM_OBJS) $(CPU_CPP_OBJS) $(RUNTIME_TEST_OBJ) -o $@ $(LDFLAGS)

$(INTEGRATION_TEST_BIN): $(CPP_OBJS) $(MM_OBJS) $(CPU_CPP_OBJS) $(INTEGRATION_TEST_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CPP_OBJS) $(MM_OBJS) $(CPU_CPP_OBJS) $(INTEGRATION_TEST_OBJ) -o $@ $(LDFLAGS)

$(QWEN3_5_SCAFFOLD_TEST_BIN): $(CPP_OBJS) $(MM_OBJS) $(CPU_CPP_OBJS) $(QWEN3_5_SCAFFOLD_TEST_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CPP_OBJS) $(MM_OBJS) $(CPU_CPP_OBJS) $(QWEN3_5_SCAFFOLD_TEST_OBJ) -o $@ $(LDFLAGS)

$(MATMUL_BENCH_BIN): $(CPP_OBJS) $(MM_OBJS) $(MATMUL_BENCH_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CPP_OBJS) $(MM_OBJS) $(MATMUL_BENCH_OBJ) -o $@ $(LDFLAGS)

$(BASELINE_BENCH_BIN): $(CPP_OBJS) $(MM_OBJS) $(BASELINE_BENCH_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CPP_OBJS) $(MM_OBJS) $(BASELINE_BENCH_OBJ) -o $@ $(LDFLAGS)

$(PHASE1_TEST_BIN): $(CPP_OBJS) $(MM_OBJS) $(PHASE1_TEST_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CPP_OBJS) $(MM_OBJS) $(PHASE1_TEST_OBJ) -o $@ $(LDFLAGS)

$(METALLIB): $(AIR_FILE)
	@mkdir -p $(SHADER_BUILD_DIR)
	$(METALLIB_TOOL) $< -o $@


$(AIR_FILE): $(SHADER_ENTRY)
	@mkdir -p $(dir $@)
	$(METAL) -c $< -o $@

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: %.mm
	@mkdir -p $(dir $@)
	$(CXX) $(OBJCXXFLAGS) -c $< -o $@

$(OBJ_DIR)/cpu/%.o: ../cpu/src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPU_CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/test/integration/%.o: test/integration/%.mm
	@mkdir -p $(dir $@)
	$(CXX) $(CPU_OBJCXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)

-include $(DEPFILES)
