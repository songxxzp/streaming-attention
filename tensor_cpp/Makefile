# Makefile for Tensor C++ Library (llama.cpp style)
# Non-template, float-only implementation with OpenMP and MPI support

CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -Wall -Wextra
INCLUDES = -I./include

# OpenMP flags
OMP_FLAGS = -fopenmp

# MPI flags
MPI_CXX = /usr/bin/mpicxx
MPIFLAGS = -fopenmp
MPI_INCLUDES = -I/usr/lib/x86_64-linux-gnu/openmpi/include

# Source files
SRCS = src/tensor.cpp src/ops.cpp
OBJS = $(SRCS:%.cpp=%.o)

# Test files
TEST_SRC = tests/test_ops.cpp
ATTENTION_TEST_SRC = tests/test_attention.cpp
STREAMING_TEST_SRC = tests/test_streaming_attention.cpp
TORCH_VALIDATION_SRC = tests/torch_validation.cpp
EXAMPLES_SRC = examples/basic_usage.cpp

# Output directories
BUILD_DIR = build
RESULTS_DIR = results

# Targets
.PHONY: all clean test test-attention test-streaming torch-validation examples run run-examples help

all: $(BUILD_DIR) $(BUILD_DIR)/test_ops $(BUILD_DIR)/test_attention $(BUILD_DIR)/basic_usage

# Create build directory
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(RESULTS_DIR)

# Compile source files
src/%.o: src/%.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) $(INCLUDES) -c -o $@ $<

# Basic test (OpenMP)
$(BUILD_DIR)/test_ops: $(OBJS) $(TEST_SRC) | $(BUILD_DIR)
	@echo "Building test_ops..."
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) $(INCLUDES) -o $@ $(OBJS) $(TEST_SRC)
	@echo "Built: $@"

# Basic test (MPI)
$(BUILD_DIR)/test_ops_mpi: $(OBJS) $(TEST_SRC) | $(BUILD_DIR)
	@echo "Building test_ops_mpi..."
	$(MPI_CXX) $(CXXFLAGS) $(MPIFLAGS) $(INCLUDES) $(MPI_INCLUDES) -o $@ $(OBJS) $(TEST_SRC)
	@echo "Built: $@"

# Attention test (OpenMP)
$(BUILD_DIR)/test_attention: $(OBJS) $(ATTENTION_TEST_SRC) | $(BUILD_DIR)
	@echo "Building test_attention..."
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) $(INCLUDES) -o $@ $(OBJS) $(ATTENTION_TEST_SRC)
	@echo "Built: $@"

# Attention test (MPI)
$(BUILD_DIR)/test_attention_mpi: $(OBJS) $(ATTENTION_TEST_SRC) | $(BUILD_DIR)
	@echo "Building test_attention_mpi..."
	$(MPI_CXX) $(CXXFLAGS) $(MPIFLAGS) $(INCLUDES) $(MPI_INCLUDES) -o $@ $(OBJS) $(ATTENTION_TEST_SRC)
	@echo "Built: $@"

# Streaming attention test (OpenMP)
$(BUILD_DIR)/test_streaming: $(OBJS) $(STREAMING_TEST_SRC) | $(BUILD_DIR)
	@echo "Building test_streaming..."
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) $(INCLUDES) -o $@ $(OBJS) $(STREAMING_TEST_SRC)
	@echo "Built: $@"

# Streaming attention test (MPI)
$(BUILD_DIR)/test_streaming_mpi: $(OBJS) $(STREAMING_TEST_SRC) | $(BUILD_DIR)
	@echo "Building test_streaming_mpi..."
	$(MPI_CXX) $(CXXFLAGS) $(MPIFLAGS) $(INCLUDES) $(MPI_INCLUDES) -o $@ $(OBJS) $(STREAMING_TEST_SRC)
	@echo "Built: $@"

# Examples
$(BUILD_DIR)/basic_usage: $(OBJS) $(EXAMPLES_SRC) | $(BUILD_DIR)
	@echo "Building basic_usage..."
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) $(INCLUDES) -o $@ $(OBJS) $(EXAMPLES_SRC)
	@echo "Built: $@"

# Run tests
run: $(BUILD_DIR)/test_ops
	@echo "Running basic tests..."
	@./$(BUILD_DIR)/test_ops

run-mpi: $(BUILD_DIR)/test_ops_mpi
	@echo "Running tests with MPI (4 processes)..."
	@mpirun -np 4 ./$(BUILD_DIR)/test_ops_mpi

run-mpi-n:
	@echo "Running tests with $(N) MPI processes..."
	@mpirun -np $(N) ./$(BUILD_DIR)/test_ops_mpi

# Attention tests
test-attention: $(BUILD_DIR)/test_attention
	@echo "Running attention tests..."
	@./$(BUILD_DIR)/test_attention

test-attention-mpi: $(BUILD_DIR)/test_attention_mpi
	@echo "Running attention tests with MPI (4 processes)..."
	@mpirun -np 4 ./$(BUILD_DIR)/test_attention_mpi

# Streaming attention tests
test-streaming: $(BUILD_DIR)/test_streaming
	@echo "Running streaming attention tests..."
	@./$(BUILD_DIR)/test_streaming

test-streaming-mpi: $(BUILD_DIR)/test_streaming_mpi
	@echo "Running streaming attention tests with MPI (4 processes)..."
	@mpirun -np 4 ./$(BUILD_DIR)/test_streaming_mpi

# PyTorch validation tests
torch-validation: $(BUILD_DIR)/torch_validation
	@echo "Running PyTorch validation tests..."
	@./$(BUILD_DIR)/torch_validation

$(BUILD_DIR)/torch_validation: $(OBJS) $(TORCH_VALIDATION_SRC) | $(BUILD_DIR)
	@echo "Building torch_validation..."
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) $(INCLUDES) -o $@ $(OBJS) $(TORCH_VALIDATION_SRC)
	@echo "Built: $@"

# Examples
.PHONY: examples run-examples
examples: $(BUILD_DIR)/basic_usage

run-examples: $(BUILD_DIR)/basic_usage
	@echo "Running examples..."
	@./$(BUILD_DIR)/basic_usage

# Clean
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)/*.o $(BUILD_DIR)/test_ops* $(BUILD_DIR)/basic_usage
	@echo "Clean complete"

# Help
help:
	@echo "Tensor C++ Library (llama.cpp style)"
	@echo "======================================"
	@echo ""
	@echo "Targets:"
	@echo "  all                - Build all tests and examples"
	@echo "  run                - Run basic operator tests (OpenMP)"
	@echo "  run-mpi            - Run tests with MPI (4 processes)"
	@echo "  run-mpi-n          - Run tests with N MPI processes (N=8 make run-mpi-n)"
	@echo "  test-attention     - Run attention tests (Self & Cross)"
	@echo "  test-attention-mpi - Run attention tests with MPI"
	@echo "  test-streaming     - Run streaming attention tests"
	@echo "  test-streaming-mpi - Run streaming attention tests with MPI"
	@echo "  torch-validation   - Run PyTorch validation tests"
	@echo "  examples           - Build usage examples"
	@echo "  run-examples       - Build and run examples"
	@echo "  clean              - Remove build artifacts"
	@echo "  help               - Show this help message"
	@echo ""
	@echo "Project Structure:"
	@echo "  include/tensor_cpp/ - Public headers"
	@echo "  src/                - Implementation files"
	@echo "  tests/             - Test suite"
	@echo "  examples/          - Usage examples"
	@echo "  build/             - Compiled binaries"
	@echo ""
	@echo "Examples:"
	@echo "  make                  # Build all"
	@echo "  make run              # Run basic tests"
	@echo "  make test-attention   # Run attention tests"
	@echo "  make run-examples     # Run examples"
	@echo "  make run-mpi          # Run MPI tests"
