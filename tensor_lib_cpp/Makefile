# Makefile for Tensor Library C++ with OpenMP and MPI support

CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -Wall -I./include

# OpenMP flags
OMP_FLAGS = -fopenmp

# MPI flags
MPI_CXX = /usr/bin/mpicxx
MPIFLAGS = -fopenmp

# Include directories (for MPI)
INCLUDES = -I./include -I/usr/lib/x86_64-linux-gnu/openmpi/include

# Source files
TEST_SRC = tests/test_ops.cpp
ATTENTION_TEST_SRC = tests/test_attention.cpp
EXAMPLES_SRC = examples/basic_usage.cpp

# Output directories
BUILD_DIR = build
RESULTS_DIR = results

# Targets
.PHONY: all clean run run-mpi test-attention test-attention-mpi examples help

all: $(BUILD_DIR)/test_ops $(BUILD_DIR)/test_attention examples

# Regular test (without MPI)
$(BUILD_DIR)/test_ops: $(TEST_SRC)
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(RESULTS_DIR)
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) $(INCLUDES) -o $@ $<
	@echo "Built: $@"

# MPI test
$(BUILD_DIR)/test_ops_mpi: $(TEST_SRC)
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(RESULTS_DIR)
	$(MPI_CXX) $(CXXFLAGS) $(MPIFLAGS) $(INCLUDES) -o $@ $<
	@echo "Built: $@"

# Attention test (OpenMP only)
$(BUILD_DIR)/test_attention: $(ATTENTION_TEST_SRC)
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(RESULTS_DIR)
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) $(INCLUDES) -o $@ $<
	@echo "Built: $@"

# Attention test (MPI)
$(BUILD_DIR)/test_attention_mpi: $(ATTENTION_TEST_SRC)
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(RESULTS_DIR)
	$(MPI_CXX) $(CXXFLAGS) $(MPIFLAGS) $(INCLUDES) -o $@ $<
	@echo "Built: $@"

# Examples
.PHONY: examples
examples: $(BUILD_DIR)/basic_usage

$(BUILD_DIR)/basic_usage: $(EXAMPLES_SRC)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) $(INCLUDES) -o $@ $<
	@echo "Built: $@"

# Run examples
.PHONY: run-examples
run-examples: $(BUILD_DIR)/basic_usage
	@echo "Running examples..."
	@./$(BUILD_DIR)/basic_usage

# Run regular tests
run: $(BUILD_DIR)/test_ops
	@echo "Running tests..."
	@./$(BUILD_DIR)/test_ops

# Run MPI tests
run-mpi: $(BUILD_DIR)/test_ops_mpi
	@echo "Running MPI tests with 4 processes..."
	@mpirun -np 4 --allow-run-as-root ./$(BUILD_DIR)/test_ops_mpi

# Run MPI tests with custom number of processes
run-mpi-n:
	@echo "Running MPI tests with $(N) processes..."
	@mpirun -np $(N) --allow-run-as-root ./$(BUILD_DIR)/test_ops_mpi

# Run attention tests (OpenMP)
test-attention: $(BUILD_DIR)/test_attention
	@echo "Running attention tests..."
	@./$(BUILD_DIR)/test_attention

# Run attention tests (MPI)
test-attention-mpi: $(BUILD_DIR)/test_attention_mpi
	@echo "Running attention tests with MPI (4 processes)..."
	@mpirun -np 4 ./$(BUILD_DIR)/test_attention_mpi

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)/*
	@echo "Clean complete"

help:
	@echo "Tensor Library C++ - Build System"
	@echo "==================================="
	@echo ""
	@echo "Targets:"
	@echo "  all                - Build all tests and examples"
	@echo "  run                - Run basic operator tests"
	@echo "  run-mpi            - Run basic operator tests with MPI (4 processes)"
	@echo "  run-mpi-n          - Run tests with N processes (use N=8 make run-mpi-n)"
	@echo "  test-attention     - Run attention tests (Self & Cross)"
	@echo "  test-attention-mpi - Run attention tests with MPI"
	@echo "  examples           - Build usage examples"
	@echo "  run-examples       - Build and run examples"
	@echo "  clean              - Remove build artifacts"
	@echo "  help               - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make                    # Build all"
	@echo "  make run                # Run basic tests"
	@echo "  make run-mpi            # Run MPI tests"
	@echo "  make test-attention     # Run attention tests"
	@echo "  make test-attention-mpi # Run attention tests with MPI"
	@echo "  make run-examples       # Build and run examples"
	@echo "  N=8 make run-mpi-n      # Run MPI tests with 8 processes"
