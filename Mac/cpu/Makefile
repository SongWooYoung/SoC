CXX := clang++

CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -Wpedantic -MMD -MP -I.

SRC_DIR := src
BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj
BIN_DIR := $(BUILD_DIR)/bin
TEST_DIR := test
TARGET := main
TEST_TARGET := test_nn_ops
RUNTIME_TEST_TARGET := test_runtime_assets
TENSOR_TEST_TARGET := test_tensor_runtime
BOOTSTRAP_TEST_TARGET := test_model_bootstrap
MODULE_TEST_TARGET := test_nn_modules
MODEL_LOADER_TEST_TARGET := test_model_loader
EXPORTED_E2E_TEST_TARGET := test_exported_e2e
HF_EXPORT_INTEGRATION_TEST_TARGET := test_hf_export_integration
REAL_QWEN3_MANIFEST := $(abspath ../../models/cpp/qwen3-0.6b/manifest.json)
MAIN_SRC := main.cpp
TEST_SRC := $(TEST_DIR)/test_nn_ops.cpp
RUNTIME_TEST_SRC := $(TEST_DIR)/test_runtime_assets.cpp
TENSOR_TEST_SRC := $(TEST_DIR)/test_tensor_runtime.cpp
BOOTSTRAP_TEST_SRC := $(TEST_DIR)/test_model_bootstrap.cpp
MODULE_TEST_SRC := $(TEST_DIR)/test_nn_modules.cpp
MODEL_LOADER_TEST_SRC := $(TEST_DIR)/test_model_loader.cpp
EXPORTED_E2E_TEST_SRC := $(TEST_DIR)/test_exported_e2e.cpp
HF_EXPORT_INTEGRATION_TEST_SRC := $(TEST_DIR)/test_hf_export_integration.cpp

MAIN_BIN := $(BIN_DIR)/$(TARGET)
TEST_BIN := $(BIN_DIR)/$(TEST_TARGET)
RUNTIME_TEST_BIN := $(BIN_DIR)/$(RUNTIME_TEST_TARGET)
TENSOR_TEST_BIN := $(BIN_DIR)/$(TENSOR_TEST_TARGET)
BOOTSTRAP_TEST_BIN := $(BIN_DIR)/$(BOOTSTRAP_TEST_TARGET)
MODULE_TEST_BIN := $(BIN_DIR)/$(MODULE_TEST_TARGET)
MODEL_LOADER_TEST_BIN := $(BIN_DIR)/$(MODEL_LOADER_TEST_TARGET)
EXPORTED_E2E_TEST_BIN := $(BIN_DIR)/$(EXPORTED_E2E_TEST_TARGET)
HF_EXPORT_INTEGRATION_TEST_BIN := $(BIN_DIR)/$(HF_EXPORT_INTEGRATION_TEST_TARGET)

COMMON_SRCS := $(wildcard $(SRC_DIR)/*.cpp)
COMMON_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/src/%.o,$(COMMON_SRCS))
OBJS := $(OBJ_DIR)/main.o $(COMMON_OBJS)
TEST_OBJS := $(OBJ_DIR)/test_nn_ops.o $(COMMON_OBJS)
RUNTIME_TEST_OBJS := $(OBJ_DIR)/test_runtime_assets.o $(COMMON_OBJS)
TENSOR_TEST_OBJS := $(OBJ_DIR)/test_tensor_runtime.o $(COMMON_OBJS)
BOOTSTRAP_TEST_OBJS := $(OBJ_DIR)/test_model_bootstrap.o $(COMMON_OBJS)
MODULE_TEST_OBJS := $(OBJ_DIR)/test_nn_modules.o $(COMMON_OBJS)
MODEL_LOADER_TEST_OBJS := $(OBJ_DIR)/test_model_loader.o $(COMMON_OBJS)
EXPORTED_E2E_TEST_OBJS := $(OBJ_DIR)/test_exported_e2e.o $(COMMON_OBJS)
HF_EXPORT_INTEGRATION_TEST_OBJS := $(OBJ_DIR)/test_hf_export_integration.o $(COMMON_OBJS)
DEPFILES := $(OBJS:.o=.d) $(TEST_OBJS:.o=.d) $(RUNTIME_TEST_OBJS:.o=.d) $(TENSOR_TEST_OBJS:.o=.d) $(BOOTSTRAP_TEST_OBJS:.o=.d) $(MODULE_TEST_OBJS:.o=.d) $(MODEL_LOADER_TEST_OBJS:.o=.d) $(EXPORTED_E2E_TEST_OBJS:.o=.d) $(HF_EXPORT_INTEGRATION_TEST_OBJS:.o=.d)

.PHONY: all main build-main build-tests build-integration test integration check-core check-goldens regression run clean \
	$(TEST_TARGET) $(RUNTIME_TEST_TARGET) $(TENSOR_TEST_TARGET) $(BOOTSTRAP_TEST_TARGET) $(MODULE_TEST_TARGET) $(MODEL_LOADER_TEST_TARGET) $(EXPORTED_E2E_TEST_TARGET) $(HF_EXPORT_INTEGRATION_TEST_TARGET) \
	integration-qwen3-golden integration-qwen3-chat-golden integration-qwen3-goldens

all: build-main

main: build-main

build-main: $(MAIN_BIN)

build-tests: $(TEST_BIN) $(RUNTIME_TEST_BIN) $(TENSOR_TEST_BIN) $(BOOTSTRAP_TEST_BIN) $(MODULE_TEST_BIN) $(MODEL_LOADER_TEST_BIN) $(EXPORTED_E2E_TEST_BIN)

build-integration: $(HF_EXPORT_INTEGRATION_TEST_BIN)

test: build-tests

integration: build-integration

check-core: build-tests
	$(TEST_BIN)
	$(RUNTIME_TEST_BIN)
	$(TENSOR_TEST_BIN)
	$(BOOTSTRAP_TEST_BIN)
	$(MODULE_TEST_BIN)
	$(MODEL_LOADER_TEST_BIN)
	$(EXPORTED_E2E_TEST_BIN)

check-goldens: integration-qwen3-goldens

regression:
	$(MAKE) clean
	$(MAKE) build-tests
	$(MAKE) build-integration
	$(MAKE) check-core
	$(MAKE) check-goldens

integration-qwen3-goldens: integration-qwen3-golden integration-qwen3-chat-golden

integration-qwen3-golden: build-integration
	@test -f $(REAL_QWEN3_MANIFEST) || (echo "missing real export bundle: $(REAL_QWEN3_MANIFEST)" && exit 1)
	SOC_HF_MANIFEST=$(REAL_QWEN3_MANIFEST) \
	SOC_HF_PROMPT='Hello, world!' \
	SOC_HF_APPLY_CHAT_TEMPLATE=0 \
	SOC_HF_MAX_NEW_TOKENS=1 \
	SOC_HF_EXPECT_PROMPT_TOKEN_IDS='9707,11,1879,0' \
	SOC_HF_EXPECT_GENERATED_TOKEN_IDS='17980' \
	SOC_HF_EXPECT_GENERATED_TEXT=' Navigation' \
	$(HF_EXPORT_INTEGRATION_TEST_BIN)

integration-qwen3-chat-golden: build-integration
	@test -f $(REAL_QWEN3_MANIFEST) || (echo "missing real export bundle: $(REAL_QWEN3_MANIFEST)" && exit 1)
	SOC_HF_MANIFEST=$(REAL_QWEN3_MANIFEST) \
	SOC_HF_PROMPT='Hello, world!' \
	SOC_HF_APPLY_CHAT_TEMPLATE=1 \
	SOC_HF_ENABLE_THINKING=0 \
	SOC_HF_MAX_NEW_TOKENS=1 \
	SOC_HF_EXPECT_PROMPT_TOKEN_IDS='151644,8948,198,2610,525,264,10950,17847,13,151645,198,151644,872,198,9707,11,1879,0,151645,198,151644,77091,198' \
	SOC_HF_EXPECT_GENERATED_TOKEN_IDS='334' \
	SOC_HF_EXPECT_GENERATED_TEXT='**' \
	$(HF_EXPORT_INTEGRATION_TEST_BIN)

$(TEST_TARGET): $(TEST_BIN)

$(RUNTIME_TEST_TARGET): $(RUNTIME_TEST_BIN)

$(TENSOR_TEST_TARGET): $(TENSOR_TEST_BIN)

$(BOOTSTRAP_TEST_TARGET): $(BOOTSTRAP_TEST_BIN)

$(MODULE_TEST_TARGET): $(MODULE_TEST_BIN)

$(MODEL_LOADER_TEST_TARGET): $(MODEL_LOADER_TEST_BIN)

$(EXPORTED_E2E_TEST_TARGET): $(EXPORTED_E2E_TEST_BIN)

$(HF_EXPORT_INTEGRATION_TEST_TARGET): $(HF_EXPORT_INTEGRATION_TEST_BIN)

$(MAIN_BIN): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $^ -o $@

$(TEST_BIN): $(TEST_OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $^ -o $@

$(RUNTIME_TEST_BIN): $(RUNTIME_TEST_OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $^ -o $@

$(TENSOR_TEST_BIN): $(TENSOR_TEST_OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $^ -o $@

$(BOOTSTRAP_TEST_BIN): $(BOOTSTRAP_TEST_OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $^ -o $@

$(MODULE_TEST_BIN): $(MODULE_TEST_OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $^ -o $@

$(MODEL_LOADER_TEST_BIN): $(MODEL_LOADER_TEST_OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $^ -o $@

$(EXPORTED_E2E_TEST_BIN): $(EXPORTED_E2E_TEST_OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $^ -o $@

$(HF_EXPORT_INTEGRATION_TEST_BIN): $(HF_EXPORT_INTEGRATION_TEST_OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $^ -o $@

$(OBJ_DIR)/main.o: $(MAIN_SRC)
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/test_nn_ops.o: $(TEST_SRC)
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/test_runtime_assets.o: $(RUNTIME_TEST_SRC)
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/test_tensor_runtime.o: $(TENSOR_TEST_SRC)
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/test_model_bootstrap.o: $(BOOTSTRAP_TEST_SRC)
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/test_nn_modules.o: $(MODULE_TEST_SRC)
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/test_model_loader.o: $(MODEL_LOADER_TEST_SRC)
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/test_exported_e2e.o: $(EXPORTED_E2E_TEST_SRC)
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/test_hf_export_integration.o: $(HF_EXPORT_INTEGRATION_TEST_SRC)
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/src/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)/src
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: build-main
	$(MAIN_BIN)

clean:
	rm -rf $(BUILD_DIR)
	rm -f *.a *.dylib *.so lib*.a lib*.dylib lib*.so

-include $(DEPFILES)