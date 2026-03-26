CXX := clang++

CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -Wpedantic -I.

SRC_DIR := src
BUILD_DIR := build
TARGET := main
TEST_TARGET := test_nn_ops
MAIN_SRC := main.cpp
TEST_SRC := test_nn_ops.cpp

# Template implementations are in headers, no cpp files needed
SRCS :=
OBJS := $(BUILD_DIR)/main.o
TEST_OBJS := $(BUILD_DIR)/test_nn_ops.o

.PHONY: all run clean test

all: $(TARGET)

test: $(TEST_TARGET)

$(TARGET): $(OBJS)
	$(CXX) $^ -o $@

$(TEST_TARGET): $(TEST_OBJS)
	$(CXX) $^ -o $@

$(BUILD_DIR)/main.o: $(MAIN_SRC)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/test_nn_ops.o: $(TEST_SRC)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(TARGET) $(TEST_TARGET)
	rm -f *.a *.dylib *.so lib*.a lib*.dylib lib*.so