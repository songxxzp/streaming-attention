# Tensor C++ Library Tests

This directory contains comprehensive tests for the Tensor C++ library, organized by category.

## 📁 Directory Structure

```
tests/
├── unit/              # Unit tests for individual components
├── integration/       # Integration tests for complete workflows
├── benchmark/         # Performance benchmarks
├── validation/        # Validation tests and profiling tools
└── README.md         # This file
```

---

## 🧪 Unit Tests (`unit/`)

Tests for individual components and operators.

| Test File | Description |
|-----------|-------------|
| `test_simple.cpp` | Basic compilation and runtime test |
| `test_ops.cpp` | Core tensor operations (matmul, add, RMSNorm, etc.) |
| `test_attention.cpp` | Self-attention and cross-attention operators |
| `test_streaming_attention.cpp` | Streaming attention operators |
| `test_avx_ops.cpp` | AVX SIMD optimized operators |
| `test_mpi_simple.cpp` | Basic MPI functionality test |
| `test_mpi_ops.cpp` | MPI parallelized operators |
| `test_tensor_parallel.cpp` | Tensor parallelism implementation |
| `test_weights_broadcast.cpp` | MPI weights broadcasting |

**Usage:**
```bash
# Run all unit tests
cd build
./test_simple
./test_ops
./test_attention
./test_streaming_attention
./test_avx_ops

# Run MPI tests (requires MPI)
mpirun -np 2 ./test_mpi_simple
mpirun -np 4 ./test_mpi_ops
mpirun -np 2 ./test_tensor_parallel
```

---

## 🔗 Integration Tests (`integration/`)

End-to-end tests for complete Qwen3 model workflows.

| Test File | Description |
|-----------|-------------|
| `test_qwen3.cpp` | Full Qwen3 forward pass test |
| `test_qwen3_verify.cpp` | Output consistency verification |
| `test_qwen3_decode.cpp` | Decode test with tokenizer verification |
| `test_qwen3_generate.cpp` | Autoregressive generation test |
| `test_qwen3_generate_with_cache.cpp` | Generation with KV cache |
| `test_qwen3_mpi_simple.cpp` | Simple MPI Qwen3 test |

**Usage:**
```bash
# Run integration tests
cd build
./test_qwen3                              # Full forward pass
./test_qwen3_verify                       # Verify consistency
./test_qwen3_decode                       # Decode test
./test_qwen3_generate                     # Generate text
./test_qwen3_generate_with_cache          # Generate with KV cache

# MPI integration test
mpirun -np 2 ./test_qwen3_mpi_simple
```

**Model Loading:**
Integration tests require the Qwen3 model weights:
```bash
# Default model path
/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors

# Set custom path via environment (if supported)
export QWEN3_MODEL_PATH=/path/to/model
```

---

## ⚡ Benchmark Tests (`benchmark/`)

Performance benchmarks for comparing different implementations.

| Test File | Description |
|-----------|-------------|
| `benchmark_attention.cpp` | Standard vs Streaming attention performance |
| `benchmark_qwen3.cpp` | Main Qwen3 benchmark (Prefill/Decode throughput) |
| `benchmark_performance.cpp` | MPI and AVX comprehensive benchmark |
| `benchmark_qwen3_versions.cpp` | Compare Baseline/AVX2/MPI/MPI+AVX2 |
| `benchmark_avx2_versions.cpp` | Compare AVX2 versions (old vs V2) |

**Usage:**
```bash
# Run benchmarks
cd build

# Attention benchmark
OMP_NUM_THREADS=16 ./benchmark_attention

# Main Qwen3 benchmark (Prefill + Decode)
OMP_NUM_THREADS=16 ./benchmark_qwen3

# Version comparison
OMP_NUM_THREADS=16 ./benchmark_qwen3_versions

# AVX2 detailed comparison
OMP_NUM_THREADS=16 ./benchmark_avx2_versions

# MPI benchmark (requires MPI)
mpirun -np 2 ./benchmark_performance
```

**Expected Performance (Qwen3-0.6B, OMP_NUM_THREADS=16):**
- **Baseline**: ~4.0s (seq_len=4), ~15.6s (seq_len=32)
- **AVX2 V2**: ~1.2s (seq_len=4), ~7.7s (seq_len=32)
- **Speedup**: 1.6-3.3x depending on sequence length

---

## ✅ Validation Tests (`validation/`)

Tests for validating correctness against reference implementations.

| Test File | Description |
|-----------|-------------|
| `torch_validation.cpp` | Compare C++ outputs with PyTorch reference |
| `test_align_qwen3.cpp` | Alignment test with PyTorch |
| `profile_avx2.cpp` | Profile AVX2 MLP implementation |

**Usage:**
```bash
cd build

# Validate against PyTorch
./torch_validation

# Alignment test
./test_align_qwen3

# Profile AVX2 (requires model)
OMP_NUM_THREADS=16 ./profile_avx2
```

---

## 🚀 Quick Start

### 1. Build All Tests

```bash
cd tensor_cpp
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### 2. Run Basic Test

```bash
./test_simple
```

### 3. Run Full Model Test

```bash
./test_qwen3
```

### 4. Run Performance Benchmark

```bash
OMP_NUM_THREADS=16 ./benchmark_qwen3
```

---

## 📊 Test Categories Summary

| Category | Tests | Purpose |
|----------|-------|---------|
| **Unit** | 9 | Test individual components |
| **Integration** | 6 | Test complete workflows |
| **Benchmark** | 5 | Performance measurement |
| **Validation** | 3 | Correctness verification |
| **Total** | 23 | Comprehensive coverage |

---

## 🔧 Configuration

### OpenMP Threads

Control parallelism with `OMP_NUM_THREADS`:
```bash
# Single thread (baseline)
OMP_NUM_THREADS=1 ./test_qwen3

# 16 threads (optimal for most systems)
OMP_NUM_THREADS=16 ./test_qwen3

# All available cores
OMP_NUM_THREADS=$(/usr/bin/nproc) ./test_qwen3
```

### MPI Processes

Run distributed tests with MPI:
```bash
# 2 processes
mpirun -np 2 ./test_qwen3_mpi_simple

# 4 processes
mpirun -np 4 ./benchmark_performance

# Specify hosts (for multi-node)
mpirun -np 4 --hostfile hosts.txt ./test_qwen3_mpi_simple
```

---

## 🐛 Debugging

### Enable Debug Output

Some tests support debug output via environment variables or compile-time flags:

```bash
# Recompile with debug symbols
cmake -DCMAKE_BUILD_TYPE=Debug ..
make clean && make -j$(nproc)

# Run with debugger
gdb ./test_qwen3

# Run with valgrind for memory checks
valgrind --leak-check=full ./test_qwen3
```

### Common Issues

1. **Model not found**
   - Ensure model weights are at the expected path
   - Check file permissions

2. **MPI errors**
   - Verify MPI installation: `mpirun --version`
   - Check hostname resolution: `ping $(hostname)`

3. **AVX2 not working**
   - Check CPU support: `grep avx2 /proc/cpuinfo`
   - Verify compiler flags: `cmake .. | grep avx2`

---

## 📝 Adding New Tests

1. **Choose category**: Place test in appropriate directory (`unit/`, `integration/`, `benchmark/`, `validation/`)

2. **Add to CMakeLists.txt**:
   ```cmake
   # Unit test example
   add_executable(test_my_feature tests/unit/test_my_feature.cpp)
   target_link_libraries(test_my_feature PRIVATE tensor_cpp)
   ```

3. **Follow naming convention**:
   - Unit tests: `test_<component>.cpp`
   - Integration tests: `test_<workflow>.cpp`
   - Benchmarks: `benchmark_<subject>.cpp`
   - Validation: `<purpose>_validation.cpp`

4. **Document purpose**: Add `@file` and `@brief` comments at the top

---

## 📚 Related Documentation

- **Main README**: `../README.md`
- **Source Code**: `../src/`
- **Headers**: `../include/tensor_cpp/`
- **Examples**: `../examples/`

---

## 🎯 Testing Guidelines

### Unit Tests
- ✅ Test single functionality
- ✅ Fast execution (< 1 second)
- ✅ No external dependencies

### Integration Tests
- ✅ Test complete workflows
- ✅ Include real model loading
- ✅ Verify end-to-end correctness

### Benchmark Tests
- ✅ Measure time/memory
- ✅ Compare multiple versions
- ✅ Report throughput metrics

### Validation Tests
- ✅ Compare with reference (PyTorch)
- ✅ Check numerical accuracy
- ✅ Profile performance bottlenecks

---

## 📧 Contact

For issues or questions about tests, please refer to the main project documentation.
