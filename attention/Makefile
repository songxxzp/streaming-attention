# Makefile for Streaming Block Attention
# 编译所有测试程序

CXX = g++
MPICXX = mpicxx
CXXFLAGS = -std=c++17 -O3 -march=native -I.
OPENMP_FLAGS = -fopenmp

# 源文件和目标文件
SRC_DIR = src
TESTS_DIR = tests

# Serial executables
SERIAL_TARGETS = test_naive test_streaming

# OpenMP executables
OPENMP_TARGETS = test_naive_omp test_streaming_omp

# MPI executables
MPI_TARGETS = test_naive_mpi test_streaming_mpi

# All targets
ALL_TARGETS = $(SERIAL_TARGETS) $(OPENMP_TARGETS) $(MPI_TARGETS)

.PHONY: all serial openmp mpi clean help

# Default target: build serial and openmp
all: serial openmp

# Build serial versions
serial: $(SERIAL_TARGETS)

test_naive: $(TESTS_DIR)/test_naive.cpp $(SRC_DIR)/naive_serial.cpp $(SRC_DIR)/attention.h
	@echo "Compiling $@..."
	@$(CXX) $(CXXFLAGS) $(TESTS_DIR)/test_naive.cpp $(SRC_DIR)/naive_serial.cpp -o $@
	@echo "✓ $@ built successfully"

test_streaming: $(TESTS_DIR)/test_streaming.cpp $(SRC_DIR)/streaming_serial.cpp $(SRC_DIR)/attention.h
	@echo "Compiling $@..."
	@$(CXX) $(CXXFLAGS) $(TESTS_DIR)/test_streaming.cpp $(SRC_DIR)/streaming_serial.cpp -o $@
	@echo "✓ $@ built successfully"

# Build OpenMP versions
openmp: $(OPENMP_TARGETS)

test_naive_omp: $(TESTS_DIR)/test_naive_omp.cpp $(SRC_DIR)/naive_omp.cpp $(SRC_DIR)/attention.h
	@echo "Compiling $@..."
	@$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) $(TESTS_DIR)/test_naive_omp.cpp $(SRC_DIR)/naive_omp.cpp -o $@
	@echo "✓ $@ built successfully"

test_streaming_omp: $(TESTS_DIR)/test_streaming_omp.cpp $(SRC_DIR)/streaming_omp.cpp $(SRC_DIR)/streaming_serial.cpp $(SRC_DIR)/attention.h
	@echo "Compiling $@..."
	@$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) $(TESTS_DIR)/test_streaming_omp.cpp $(SRC_DIR)/streaming_omp.cpp $(SRC_DIR)/streaming_serial.cpp -o $@
	@echo "✓ $@ built successfully"

# Build MPI versions
mpi: $(MPI_TARGETS)

test_naive_mpi: $(TESTS_DIR)/test_naive_mpi.cpp $(SRC_DIR)/naive_mpi.cpp $(SRC_DIR)/attention.h
	@echo "Compiling $@..."
	@$(MPICXX) $(CXXFLAGS) $(OPENMP_FLAGS) $(TESTS_DIR)/test_naive_mpi.cpp $(SRC_DIR)/naive_mpi.cpp -o $@
	@echo "✓ $@ built successfully"

test_streaming_mpi: $(TESTS_DIR)/test_streaming_mpi.cpp $(SRC_DIR)/streaming_mpi.cpp $(SRC_DIR)/streaming_serial.cpp $(SRC_DIR)/attention.h
	@echo "Compiling $@..."
	@$(MPICXX) $(CXXFLAGS) $(OPENMP_FLAGS) $(TESTS_DIR)/test_streaming_mpi.cpp $(SRC_DIR)/streaming_mpi.cpp $(SRC_DIR)/streaming_serial.cpp -o $@
	@echo "✓ $@ built successfully"

# Clean all build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -f $(ALL_TARGETS)
	@echo "✓ Clean complete"

# Help target
help:
	@echo "Streaming Block Attention - Makefile"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build serial and OpenMP versions (default)"
	@echo "  serial   - Build serial versions only"
	@echo "  openmp   - Build OpenMP versions only"
	@echo "  mpi      - Build MPI versions (requires mpicxx)"
	@echo "  clean    - Remove all built executables"
	@echo "  help     - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make              # Build serial and OpenMP versions"
	@echo "  make all          # Same as above"
	@echo "  make mpi          # Build MPI versions"
	@echo "  make clean        # Clean all build artifacts"
	@echo ""
	@echo "Running tests:"
	@echo "  ./test_naive 1024 128 64"
	@echo "  OMP_NUM_THREADS=4 ./test_naive_omp 1024 128 64"
	@echo "  mpirun -np 2 ./test_naive_mpi 1024 128 4"
	@echo "  python scripts/compare_attention_full.py --help"
