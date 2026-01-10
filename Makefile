# Makefile for Streaming Attention Project

CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -Wall

# OpenMP flags
OMP_FLAGS = -fopenmp

# MPI flags
MPI_CXX = /usr/bin/mpicxx
MPIFLAGS =

# Include directories
INCLUDES = -I. -I/usr/lib/x86_64-linux-gnu/openmpi/include

# Phase 1 targets
test_correctness: test_correctness.cpp attention/naive_serial.o attention/streaming_serial.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^

attention/naive_serial.o: attention/naive_serial.cpp attention/attention.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

attention/streaming_serial.o: attention/streaming_serial.cpp attention/attention.h utils/softmax_online.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

# Phase 2 targets (OpenMP)
test_omp: test_omp.cpp attention/naive_serial.o attention/streaming_serial.o attention/streaming_omp.o
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) $(INCLUDES) -o $@ $^

attention/streaming_omp.o: attention/streaming_omp.cpp attention/attention.h utils/softmax_online.h
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) $(INCLUDES) -c -o $@ $<

# Phase 3 targets (MPI + OpenMP)
# Use system MPI compiler
MPI_CXX = /usr/bin/mpicxx

test_mpi: test_mpi.cpp attention/naive_serial.o attention/streaming_serial.o attention/streaming_mpi.o
	$(MPI_CXX) $(CXXFLAGS) $(OMP_FLAGS) $(INCLUDES) -o $@ $^

attention/streaming_mpi.o: attention/streaming_mpi.cpp attention/attention.h utils/softmax_online.h
	$(MPI_CXX) $(CXXFLAGS) $(OMP_FLAGS) $(INCLUDES) -c -o $@ $<

# Phony targets
.PHONY: all clean run_phase1 run_phase2 run_phase3

all: test_correctness test_omp

clean:
	rm -f test_correctness test_omp test_mpi
	rm -f attention/*.o

run_phase1: test_correctness
	./test_correctness --T 2048 --d 128 --block 64

run_phase2: test_omp
	./test_omp --T 4096 --d 128 --block 64

run_phase3: test_mpi
	mpirun -np 4 ./test_mpi --T 8192 --d 128 --block 64
