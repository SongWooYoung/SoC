CXX := clang++

CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -Wpedantic -I.

SRC_DIR := src
BUILD_DIR := build
TARGET := main
TEST_TARGET := test_nn_ops
RUNTIME_TEST_TARGET := test_runtime_assets
TENSOR_TEST_TARGET := test_tensor_runtime
BOOTSTRAP_TEST_TARGET := test_model_bootstrap
MODULE_TEST_TARGET := test_nn_modules
MODEL_LOADER_TEST_TARGET := test_model_loader
EXPORTED_E2E_TEST_TARGET := test_exported_e2e
MAIN_SRC := main.cpp
TEST_SRC := test_nn_ops.cpp
RUNTIME_TEST_SRC := test_runtime_assets.cpp
TENSOR_TEST_SRC := test_tensor_runtime.cpp
BOOTSTRAP_TEST_SRC := test_model_bootstrap.cpp
MODULE_TEST_SRC := test_nn_modules.cpp
MODEL_LOADER_TEST_SRC := test_model_loader.cpp
EXPORTED_E2E_TEST_SRC := test_exported_e2e.cpp

COMMON_SRCS := $(wildcard $(SRC_DIR)/*.cpp)
COMMON_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/src/%.o,$(COMMON_SRCS))
OBJS := $(BUILD_DIR)/main.o $(COMMON_OBJS)
TEST_OBJS := $(BUILD_DIR)/test_nn_ops.o $(COMMON_OBJS)
RUNTIME_TEST_OBJS := $(BUILD_DIR)/test_runtime_assets.o $(COMMON_OBJS)
TENSOR_TEST_OBJS := $(BUILD_DIR)/test_tensor_runtime.o $(COMMON_OBJS)
BOOTSTRAP_TEST_OBJS := $(BUILD_DIR)/test_model_bootstrap.o $(COMMON_OBJS)
MODULE_TEST_OBJS := $(BUILD_DIR)/test_nn_modules.o $(COMMON_OBJS)
MODEL_LOADER_TEST_OBJS := $(BUILD_DIR)/test_model_loader.o $(COMMON_OBJS)
EXPORTED_E2E_TEST_OBJS := $(BUILD_DIR)/test_exported_e2e.o $(COMMON_OBJS)

.PHONY: all run clean test

all: $(TARGET)

test: $(TEST_TARGET) $(RUNTIME_TEST_TARGET) $(TENSOR_TEST_TARGET) $(BOOTSTRAP_TEST_TARGET) $(MODULE_TEST_TARGET) $(MODEL_LOADER_TEST_TARGET) $(EXPORTED_E2E_TEST_TARGET)

$(TARGET): $(OBJS)
	$(CXX) $^ -o $@

$(TEST_TARGET): $(TEST_OBJS)
	$(CXX) $^ -o $@

$(RUNTIME_TEST_TARGET): $(RUNTIME_TEST_OBJS)
	$(CXX) $^ -o $@

$(TENSOR_TEST_TARGET): $(TENSOR_TEST_OBJS)
	$(CXX) $^ -o $@

$(BOOTSTRAP_TEST_TARGET): $(BOOTSTRAP_TEST_OBJS)
	$(CXX) $^ -o $@

$(MODULE_TEST_TARGET): $(MODULE_TEST_OBJS)
	$(CXX) $^ -o $@

$(MODEL_LOADER_TEST_TARGET): $(MODEL_LOADER_TEST_OBJS)
	$(CXX) $^ -o $@

$(EXPORTED_E2E_TEST_TARGET): $(EXPORTED_E2E_TEST_OBJS)
	$(CXX) $^ -o $@

$(BUILD_DIR)/main.o: $(MAIN_SRC)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/test_nn_ops.o: $(TEST_SRC)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/test_runtime_assets.o: $(RUNTIME_TEST_SRC)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/test_tensor_runtime.o: $(TENSOR_TEST_SRC)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/test_model_bootstrap.o: $(BOOTSTRAP_TEST_SRC)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/test_nn_modules.o: $(MODULE_TEST_SRC)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/test_model_loader.o: $(MODEL_LOADER_TEST_SRC)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/test_exported_e2e.o: $(EXPORTED_E2E_TEST_SRC)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/src/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)/src
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(TARGET) $(TEST_TARGET) $(RUNTIME_TEST_TARGET) $(TENSOR_TEST_TARGET) $(BOOTSTRAP_TEST_TARGET) $(MODULE_TEST_TARGET) $(MODEL_LOADER_TEST_TARGET) $(EXPORTED_E2E_TEST_TARGET)
	rm -f *.a *.dylib *.so lib*.a lib*.dylib lib*.so