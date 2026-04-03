CXX = clang++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -DACCELERATE_NEW_LAPACK

BUILD_DIR = build
MODEL_DIR = /Volumes/990pro/Documents/SoC/models/raw/qwen3_5-4b
CONFIG_JSON = $(MODEL_DIR)/config.json
TOKENIZER_JSON = $(MODEL_DIR)/tokenizer.json

# Sources
JSON_SRC = utils/json.cpp

# Frameworks (for Accelerate BLAS/vDSP)
FRAMEWORKS = -framework Accelerate

# Tests
TEST_CONFIG_SRC = test/test_config.cpp
TEST_TOKENIZER_SRC = test/test_tokenizer.cpp
TEST_MODELING_SRC = test/test_modeling.cpp
TEST_VISION_SRC = test/test_vision.cpp
TEST_VALIDATION_SRC = test/test_validation.cpp
REFERENCE_JSON = test/reference.json

.PHONY: all clean test test_config test_tokenizer test_modeling test_vision test_validation

all: $(BUILD_DIR)/test_config $(BUILD_DIR)/test_tokenizer $(BUILD_DIR)/test_modeling $(BUILD_DIR)/test_vision $(BUILD_DIR)/test_validation

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/test_config: $(TEST_CONFIG_SRC) $(JSON_SRC) models/qwen3_5/config.h utils/json.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I. $(TEST_CONFIG_SRC) $(JSON_SRC) -o $@

$(BUILD_DIR)/test_tokenizer: $(TEST_TOKENIZER_SRC) $(JSON_SRC) models/qwen3_5/tokenization.h utils/json.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I. $(TEST_TOKENIZER_SRC) $(JSON_SRC) -o $@

$(BUILD_DIR)/test_modeling: $(TEST_MODELING_SRC) $(JSON_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I. $(FRAMEWORKS) $(TEST_MODELING_SRC) $(JSON_SRC) -o $@

$(BUILD_DIR)/test_vision: $(TEST_VISION_SRC) $(JSON_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I. $(FRAMEWORKS) $(TEST_VISION_SRC) $(JSON_SRC) -o $@

test: test_config test_tokenizer

test_config: $(BUILD_DIR)/test_config
	$(BUILD_DIR)/test_config $(CONFIG_JSON)

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

clean:
	rm -rf $(BUILD_DIR)
