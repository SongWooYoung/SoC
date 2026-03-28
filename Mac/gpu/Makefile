CXX := clang++

CXXFLAGS := -O2 -std=c++17 -Wall -Wextra -Wpedantic -MMD -MP -Iinclude -I../cpu
OBJCXXFLAGS := $(CXXFLAGS) -fobjc-arc
CPU_CXXFLAGS := $(CXXFLAGS) -I../cpu
CPU_OBJCXXFLAGS := $(OBJCXXFLAGS) -I../cpu
LDFLAGS := -framework Foundation -framework Metal

SRC_DIR := src
TEST_DIR := test
SHADER_DIR := shaders
BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj
BIN_DIR := $(BUILD_DIR)/bin
AIR_DIR := $(BUILD_DIR)/air
SHADER_BUILD_DIR := $(BUILD_DIR)/shaders
REPORT_DIR := $(BUILD_DIR)/reports
QUICK_REPORT_DIR := $(REPORT_DIR)/quick
METALLIB := $(SHADER_BUILD_DIR)/gpu.metallib
SHADER_ENTRY := $(SHADER_DIR)/gpu_kernels.metal
MAIN_TARGET := gpu_bootstrap
BOOTSTRAP_TEST_TARGET := test_metal_bootstrap
RMSNORM_TEST_TARGET := test_rms_norm
LINEAR_TEST_TARGET := test_linear
ASSET_TEST_TARGET := test_asset_loader
PHASE3_TEST_TARGET := test_phase3_kernels
MODULE_TEST_TARGET := test_qwen_modules
RUNTIME_TEST_TARGET := test_generation_context
INTEGRATION_TEST_TARGET := test_real_bundle_regression
MATMUL_BENCH_TARGET := bench_matmul
MAIN_BIN := $(BIN_DIR)/$(MAIN_TARGET)
BOOTSTRAP_TEST_BIN := $(BIN_DIR)/$(BOOTSTRAP_TEST_TARGET)
RMSNORM_TEST_BIN := $(BIN_DIR)/$(RMSNORM_TEST_TARGET)
LINEAR_TEST_BIN := $(BIN_DIR)/$(LINEAR_TEST_TARGET)
ASSET_TEST_BIN := $(BIN_DIR)/$(ASSET_TEST_TARGET)
PHASE3_TEST_BIN := $(BIN_DIR)/$(PHASE3_TEST_TARGET)
MODULE_TEST_BIN := $(BIN_DIR)/$(MODULE_TEST_TARGET)
RUNTIME_TEST_BIN := $(BIN_DIR)/$(RUNTIME_TEST_TARGET)
INTEGRATION_TEST_BIN := $(BIN_DIR)/$(INTEGRATION_TEST_TARGET)
MATMUL_BENCH_BIN := $(BIN_DIR)/$(MATMUL_BENCH_TARGET)
REAL_QWEN3_MANIFEST := $(abspath ../../models/cpp/qwen3-0.6b/manifest.json)
SWEEP_SCRIPT := tools/run_real_bundle_sweep.sh
REAL_BUNDLE_REPORT := $(abspath $(REPORT_DIR)/test_real_bundle_regression_report.md)
SWEEP_SUMMARY := $(abspath $(REPORT_DIR)/test_real_bundle_sweep_summary.md)
SWEEP_REPORT_DIR := $(abspath $(REPORT_DIR)/sweep_cases)
QUICK_REAL_BUNDLE_REPORT := $(abspath $(QUICK_REPORT_DIR)/test_real_bundle_regression_report.md)
QUICK_SWEEP_SUMMARY := $(abspath $(QUICK_REPORT_DIR)/test_real_bundle_sweep_summary.md)
QUICK_SWEEP_REPORT_DIR := $(abspath $(QUICK_REPORT_DIR)/sweep_cases)
ACTIVE_REAL_BUNDLE_REPORT ?= $(REAL_BUNDLE_REPORT)
ACTIVE_SWEEP_SUMMARY ?= $(SWEEP_SUMMARY)
ACTIVE_SWEEP_REPORT_DIR ?= $(SWEEP_REPORT_DIR)
REAL_BUNDLE_MAX_NEW_TOKENS ?= 8
SWEEP_MAX_NEW_TOKENS ?= 8

METAL := xcrun -sdk macosx metal
METALLIB_TOOL := xcrun -sdk macosx metallib

CPP_SRCS := $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*/*.cpp)
MM_SRCS := $(wildcard $(SRC_DIR)/*.mm) $(wildcard $(SRC_DIR)/*/*.mm)
CPU_CPP_SRCS := $(wildcard ../cpu/src/*.cpp)

CPP_OBJS := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(CPP_SRCS))
MM_OBJS := $(patsubst %.mm,$(OBJ_DIR)/%.o,$(MM_SRCS))
CPU_CPP_OBJS := $(patsubst ../cpu/src/%.cpp,$(OBJ_DIR)/cpu/%.o,$(CPU_CPP_SRCS))
MAIN_OBJ := $(OBJ_DIR)/main.o
BOOTSTRAP_TEST_OBJ := $(OBJ_DIR)/test/metal/test_metal_bootstrap.o
RMSNORM_TEST_OBJ := $(OBJ_DIR)/test/kernel/test_rms_norm.o
LINEAR_TEST_OBJ := $(OBJ_DIR)/test/kernel/test_linear.o
ASSET_TEST_OBJ := $(OBJ_DIR)/test/asset/test_asset_loader.o
PHASE3_TEST_OBJ := $(OBJ_DIR)/test/kernel/test_phase3_kernels.o
MODULE_TEST_OBJ := $(OBJ_DIR)/test/module/test_qwen_modules.o
RUNTIME_TEST_OBJ := $(OBJ_DIR)/test/runtime/test_generation_context.o
INTEGRATION_TEST_OBJ := $(OBJ_DIR)/test/integration/test_real_bundle_regression.o
MATMUL_BENCH_OBJ := $(OBJ_DIR)/benchmark/bench_matmul.o
AIR_FILE := $(AIR_DIR)/gpu_kernels.air
DEPFILES := $(CPP_OBJS:.o=.d) $(MM_OBJS:.o=.d) $(CPU_CPP_OBJS:.o=.d) $(MAIN_OBJ:.o=.d) $(BOOTSTRAP_TEST_OBJ:.o=.d) $(RMSNORM_TEST_OBJ:.o=.d) $(LINEAR_TEST_OBJ:.o=.d) $(ASSET_TEST_OBJ:.o=.d) $(PHASE3_TEST_OBJ:.o=.d) $(MODULE_TEST_OBJ:.o=.d) $(RUNTIME_TEST_OBJ:.o=.d) $(INTEGRATION_TEST_OBJ:.o=.d) $(MATMUL_BENCH_OBJ:.o=.d)

.PHONY: all build-shaders build-main build-tests build-benchmarks check integration integration-real-bundle integration-sweep integration-sweep-quick run benchmark regression real-bundle-regression real-bundle-regression-quick clean

all: build-main

build-shaders:
	@if xcrun -sdk macosx --find metal >/dev/null 2>&1 && xcrun -sdk macosx --find metallib >/dev/null 2>&1; then \
		$(MAKE) $(METALLIB); \
	else \
		echo "Skipping metallib build: Metal Toolchain is not installed"; \
	fi

build-main: $(MAIN_BIN)

build-tests: $(BOOTSTRAP_TEST_BIN) $(RMSNORM_TEST_BIN) $(LINEAR_TEST_BIN) $(ASSET_TEST_BIN) $(PHASE3_TEST_BIN) $(MODULE_TEST_BIN) $(RUNTIME_TEST_BIN) $(INTEGRATION_TEST_BIN)

build-benchmarks: $(MATMUL_BENCH_BIN)

check: $(BOOTSTRAP_TEST_BIN) $(RMSNORM_TEST_BIN) $(LINEAR_TEST_BIN) $(ASSET_TEST_BIN) $(PHASE3_TEST_BIN) $(MODULE_TEST_BIN) $(RUNTIME_TEST_BIN)
	$(BOOTSTRAP_TEST_BIN)
	$(RMSNORM_TEST_BIN)
	$(LINEAR_TEST_BIN)
	$(ASSET_TEST_BIN)
	$(PHASE3_TEST_BIN)
	$(MODULE_TEST_BIN)
	$(RUNTIME_TEST_BIN)

integration: $(INTEGRATION_TEST_BIN)
	$(INTEGRATION_TEST_BIN)

integration-real-bundle: $(INTEGRATION_TEST_BIN)
	@test -f $(REAL_QWEN3_MANIFEST) || (echo "missing real export bundle: $(REAL_QWEN3_MANIFEST)" && exit 1)
	@echo "Running real-bundle regression with REAL_BUNDLE_MAX_NEW_TOKENS=$(REAL_BUNDLE_MAX_NEW_TOKENS)"
	SOC_QWEN3_MANIFEST=$(REAL_QWEN3_MANIFEST) \
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

benchmark: $(MATMUL_BENCH_BIN)
	$(MATMUL_BENCH_BIN)

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

$(BOOTSTRAP_TEST_BIN): $(CPP_OBJS) $(MM_OBJS) $(BOOTSTRAP_TEST_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CPP_OBJS) $(MM_OBJS) $(BOOTSTRAP_TEST_OBJ) -o $@ $(LDFLAGS)

$(RMSNORM_TEST_BIN): $(CPP_OBJS) $(MM_OBJS) $(RMSNORM_TEST_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CPP_OBJS) $(MM_OBJS) $(RMSNORM_TEST_OBJ) -o $@ $(LDFLAGS)

$(LINEAR_TEST_BIN): $(CPP_OBJS) $(MM_OBJS) $(LINEAR_TEST_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CPP_OBJS) $(MM_OBJS) $(LINEAR_TEST_OBJ) -o $@ $(LDFLAGS)

$(ASSET_TEST_BIN): $(CPP_OBJS) $(MM_OBJS) $(ASSET_TEST_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CPP_OBJS) $(MM_OBJS) $(ASSET_TEST_OBJ) -o $@ $(LDFLAGS)

$(PHASE3_TEST_BIN): $(CPP_OBJS) $(MM_OBJS) $(PHASE3_TEST_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CPP_OBJS) $(MM_OBJS) $(PHASE3_TEST_OBJ) -o $@ $(LDFLAGS)

$(MODULE_TEST_BIN): $(CPP_OBJS) $(MM_OBJS) $(MODULE_TEST_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CPP_OBJS) $(MM_OBJS) $(MODULE_TEST_OBJ) -o $@ $(LDFLAGS)

$(RUNTIME_TEST_BIN): $(CPP_OBJS) $(MM_OBJS) $(RUNTIME_TEST_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CPP_OBJS) $(MM_OBJS) $(RUNTIME_TEST_OBJ) -o $@ $(LDFLAGS)

$(INTEGRATION_TEST_BIN): $(CPP_OBJS) $(MM_OBJS) $(CPU_CPP_OBJS) $(INTEGRATION_TEST_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CPP_OBJS) $(MM_OBJS) $(CPU_CPP_OBJS) $(INTEGRATION_TEST_OBJ) -o $@ $(LDFLAGS)

$(MATMUL_BENCH_BIN): $(CPP_OBJS) $(MM_OBJS) $(MATMUL_BENCH_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CPP_OBJS) $(MM_OBJS) $(MATMUL_BENCH_OBJ) -o $@ $(LDFLAGS)

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