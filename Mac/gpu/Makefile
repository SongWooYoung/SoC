CXX = clang++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -DACCELERATE_NEW_LAPACK

BUILD_DIR = build
MODEL_DIR = /Volumes/990pro/Documents/SoC/models/raw/qwen3_5-4b
CONFIG_JSON = $(MODEL_DIR)/config.json
TOKENIZER_JSON = $(MODEL_DIR)/tokenizer.json

# MLX
MLX_ROOT = ../../.repo_cache/mlx
MLX_CXXFLAGS = -std=c++20 -O2 -I$(MLX_ROOT)
MLX_LDFLAGS = -L$(MLX_ROOT)/build -lmlx -framework Metal -framework Foundation -framework Accelerate -framework QuartzCore

# Sources
JSON_SRC = utils/json.cpp

# Frameworks (for Accelerate BLAS/vDSP)
FRAMEWORKS = -framework Accelerate

# Tests
TEST_CONFIG_SRC = test/test_config.cpp
TEST_MLX_CONFIG_SRC = test/test_mlx_config.cpp
TEST_MLX_API_SRC = test/test_mlx_api.cpp
TEST_TOKENIZER_SRC = test/test_tokenizer.cpp
TEST_MODELING_SRC = test/test_modeling.cpp
TEST_VISION_SRC = test/test_vision.cpp
TEST_VALIDATION_SRC = test/test_validation.cpp
TEST_MLX_PHASE2_SRC = test/test_mlx_phase2.cpp
TEST_OUTPUT_EVAL_SRC = test/test_output_eval.cpp
TEST_MLX_OUTPUT_EVAL_SRC = test/test_mlx_output_eval.cpp
TEST_MLX_QUANTIZED_OUTPUT_EVAL_SRC = test/test_mlx_quantized_output_eval.cpp
REFERENCE_JSON = test/reference.json
PROMPT_SUITE_JSON = test/prompt_suite.json
RESULT_DIR = test/result
PYTHON = /Volumes/990pro/Documents/SoC/.venv/bin/python
OUTPUT_EVAL_TAG ?= suite
OUTPUT_EVAL_MAX_NEW_TOKENS ?= 256
OUTPUT_EVAL_JSON = $(RESULT_DIR)/output_eval_$(OUTPUT_EVAL_TAG).json

.PHONY: all clean test test_config test_mlx_config test_mlx_api test_mlx_phase2 test_tokenizer test_modeling test_vision test_validation test_output_eval test_mlx_output_eval test_mlx_quantized_output_eval

all: $(BUILD_DIR)/test_config $(BUILD_DIR)/test_mlx_config $(BUILD_DIR)/test_mlx_api $(BUILD_DIR)/test_tokenizer $(BUILD_DIR)/test_modeling $(BUILD_DIR)/test_vision $(BUILD_DIR)/test_validation $(BUILD_DIR)/test_output_eval $(BUILD_DIR)/test_mlx_output_eval $(BUILD_DIR)/test_mlx_quantized_output_eval

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/test_config: $(TEST_CONFIG_SRC) $(JSON_SRC) models/qwen3_5_py_cpp/config.h utils/json.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I. $(TEST_CONFIG_SRC) $(JSON_SRC) -o $@

$(BUILD_DIR)/test_mlx_config: $(TEST_MLX_CONFIG_SRC) $(JSON_SRC) models/qwen3_5_mlx/config.h models/qwen3_5_py_cpp/config.h utils/json.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I. $(TEST_MLX_CONFIG_SRC) $(JSON_SRC) -o $@

$(BUILD_DIR)/test_mlx_api: $(TEST_MLX_API_SRC) | $(BUILD_DIR)
	$(CXX) $(MLX_CXXFLAGS) -I. $(MLX_LDFLAGS) $(TEST_MLX_API_SRC) -o $@

$(BUILD_DIR)/test_mlx_phase2: $(TEST_MLX_PHASE2_SRC) models/qwen3_5_mlx/language.h models/qwen3_5_mlx/gated_delta.h models/qwen3_5_mlx/mlx_helpers.h models/qwen3_5_mlx/weight_loader.h | $(BUILD_DIR)
	$(CXX) $(MLX_CXXFLAGS) -I. $(MLX_LDFLAGS) $(TEST_MLX_PHASE2_SRC) -o $@

$(BUILD_DIR)/test_tokenizer: $(TEST_TOKENIZER_SRC) $(JSON_SRC) models/qwen3_5_py_cpp/tokenization.h utils/json.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I. $(TEST_TOKENIZER_SRC) $(JSON_SRC) -o $@

$(BUILD_DIR)/test_modeling: $(TEST_MODELING_SRC) $(JSON_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I. $(FRAMEWORKS) $(TEST_MODELING_SRC) $(JSON_SRC) -o $@

$(BUILD_DIR)/test_vision: $(TEST_VISION_SRC) $(JSON_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I. $(FRAMEWORKS) $(TEST_VISION_SRC) $(JSON_SRC) -o $@

test: test_config test_mlx_config test_tokenizer

test_config: $(BUILD_DIR)/test_config
	$(BUILD_DIR)/test_config $(CONFIG_JSON)

test_mlx_config: $(BUILD_DIR)/test_mlx_config
	$(BUILD_DIR)/test_mlx_config $(CONFIG_JSON)

test_mlx_api: $(BUILD_DIR)/test_mlx_api
	MLX_METAL_PATH=$(MLX_ROOT)/build/mlx/backend/metal/kernels $(BUILD_DIR)/test_mlx_api

test_mlx_phase2: $(BUILD_DIR)/test_mlx_phase2
	MLX_METAL_PATH=$(MLX_ROOT)/build/mlx/backend/metal/kernels $(BUILD_DIR)/test_mlx_phase2

test_tokenizer: $(BUILD_DIR)/test_tokenizer
	$(BUILD_DIR)/test_tokenizer $(TOKENIZER_JSON)

test_modeling: $(BUILD_DIR)/test_modeling
	$(BUILD_DIR)/test_modeling $(MODEL_DIR)

test_vision: $(BUILD_DIR)/test_vision
	$(BUILD_DIR)/test_vision $(MODEL_DIR)

$(BUILD_DIR)/test_validation: $(TEST_VALIDATION_SRC) $(JSON_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I. $(FRAMEWORKS) $(TEST_VALIDATION_SRC) $(JSON_SRC) -o $@

test_validation: $(BUILD_DIR)/test_validation
	$(BUILD_DIR)/test_validation $(MODEL_DIR) $(REFERENCE_JSON)

$(BUILD_DIR)/test_output_eval: $(TEST_OUTPUT_EVAL_SRC) $(JSON_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I. $(FRAMEWORKS) $(TEST_OUTPUT_EVAL_SRC) $(JSON_SRC) -o $@

$(BUILD_DIR)/test_mlx_output_eval: $(TEST_MLX_OUTPUT_EVAL_SRC) $(JSON_SRC) | $(BUILD_DIR)
	$(CXX) $(MLX_CXXFLAGS) -I. $(TEST_MLX_OUTPUT_EVAL_SRC) $(JSON_SRC) -o $@ $(MLX_LDFLAGS)

$(BUILD_DIR)/test_mlx_quantized_output_eval: $(TEST_MLX_QUANTIZED_OUTPUT_EVAL_SRC) $(JSON_SRC) | $(BUILD_DIR)
	$(CXX) $(MLX_CXXFLAGS) -I. $(TEST_MLX_QUANTIZED_OUTPUT_EVAL_SRC) $(JSON_SRC) -o $@ $(MLX_LDFLAGS)

test_output_eval: $(BUILD_DIR)/test_output_eval
	mkdir -p $(RESULT_DIR)
	$(PYTHON) test/output_eval_suite.py \
		--model-dir $(MODEL_DIR) \
		--prompt-suite $(PROMPT_SUITE_JSON) \
		--cpp-bin $(BUILD_DIR)/test_output_eval \
		--output $(OUTPUT_EVAL_JSON) \
		--max-new-tokens $(OUTPUT_EVAL_MAX_NEW_TOKENS)

test_mlx_output_eval: $(BUILD_DIR)/test_mlx_output_eval
	mkdir -p $(RESULT_DIR)
	MLX_METAL_PATH=$(MLX_ROOT)/build/mlx/backend/metal/kernels $(BUILD_DIR)/test_mlx_output_eval $(MODEL_DIR) $(PROMPT_SUITE_JSON) $(RESULT_DIR)/mlx_output_eval.json 64

test_mlx_quantized_output_eval: $(BUILD_DIR)/test_mlx_quantized_output_eval
	mkdir -p $(RESULT_DIR)
	MLX_METAL_PATH=$(MLX_ROOT)/build/mlx/backend/metal/kernels $(BUILD_DIR)/test_mlx_quantized_output_eval /Volumes/990pro/.lmstudio/models/mlx-community/Qwen3.5-9B-MLX-8bit $(PROMPT_SUITE_JSON) $(RESULT_DIR)/mlx_quantized_output_eval.json 64

clean:
	rm -rf $(BUILD_DIR)
