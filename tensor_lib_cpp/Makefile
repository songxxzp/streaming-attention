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

# Output directories
BUILD_DIR = build
RESULTS_DIR = results

# Targets
.PHONY: all clean run run-mpi help

all: $(BUILD_DIR)/test_ops

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

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)/*
	@echo "Clean complete"

help:
	@echo "Tensor Library C++ - Build System"
	@echo "==================================="
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build regular test executable (OpenMP only)"
	@echo "  run       - Run regular tests"
	@echo "  run-mpi   - Run MPI tests with 4 processes"
	@echo "  run-mpi-n - Run MPI tests with N processes (use N=8 make run-mpi-n)"
	@echo "  clean     - Remove build artifacts"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make                    # Build all"
	@echo "  make run                # Run tests"
	@echo "  make run-mpi            # Run MPI tests"
	@echo "  N=8 make run-mpi-n       # Run MPI tests with 8 processes"
