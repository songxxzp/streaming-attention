# Tensor Library C++ - Project Structure

## Directory Layout

```
tensor_lib_cpp/
├── include/tensor_lib/      # Public headers (all user-accessible interfaces)
│   ├── tensor.h             # Tensor class declarations
│   ├── tensor_impl.tpp      # Tensor template implementations
│   └── ops.h                # Operator implementations (add, attention, etc.)
│
├── tests/                   # Test suite
│   ├── test_ops.cpp         # Basic operator tests
│   └── test_attention.cpp   # Attention-specific tests
│
├── examples/                # Usage examples
│   └── basic_usage.cpp      # Comprehensive examples
│
├── build/                   # Compiled binaries (generated)
│   ├── test_ops
│   ├── test_ops_mpi
│   ├── test_attention
│   └── test_attention_mpi
│
├── results/                 # Test results (generated)
│   ├── test_results.txt
│   └── attention_test_results.txt
│
├── docs/                    # Documentation
│   └── PROJECT_STRUCTURE.md # This file
│
├── Makefile                 # Build configuration
├── README.md                # Project documentation
└── .gitignore              # Git ignore rules
```

## File Organization

### Header-Only Library Design

This is a **header-only template library**. All implementations are in header files because:
- C++ templates require source to be available at compile time
- No separate compilation step needed
- Easier integration into user projects

### Why `tensor_impl.tpp` in `include/`?

The `.tpp` file contains template implementations that must be included by `tensor.h`. It's in `include/` because:
1. It's part of the public API (included by `tensor.h`)
2. Must be in the include path for compilation
3. Standard practice for header-only template libraries

### Why `src/` was removed

The `src/` directory is typically used for:
- `.cpp` implementation files (for non-template code)
- Private implementation details

Since this is a header-only template library:
- All code must be in headers
- No separate `.cpp` files needed
- `src/` would remain empty

## Key Components

### `tensor.h` (Declaration)
- Tensor class template declaration
- Shape class
- Type aliases (TensorF, TensorD, etc.)

### `tensor_impl.tpp` (Implementation)
- All Tensor method implementations
- Only included by tensor.h (not directly by users)

### `ops.h` (Operators)
- Element-wise operations (add, etc.)
- Reduction operations (argmax)
- Neural network operators (linear, embedding, rms_norm)
- Attention mechanisms (self_attention, cross_attention)
- Position encoding (RoPE)
- Activation functions (swiglu)
- MPI operations (all_reduce_sum, broadcast)

## Build System

### Targets
```bash
make                    # Build all tests
make run                # Run basic tests
make test-attention     # Run attention tests
make run-mpi            # Run tests with MPI
make clean              # Remove build artifacts
make help               # Show all targets
```

### Compilation Models
- **Serial**: Standard C++ compilation
- **OpenMP**: Multi-threading on shared memory
- **OpenMP + MPI**: Distributed multi-processing

## Integration into Your Project

### Option 1: Copy Headers
```bash
cp -r include/tensor_lib /path/to/your/project/include/
```

### Option 2: Use as Submodule
```bash
git submodule add <repo-url> external/tensor_lib_cpp
```

Then in your Makefile:
```makefile
CXXFLAGS += -Iexternal/tensor_lib_cpp/include -fopenmp
```

## Coding Conventions

1. **Naming**:
   - Classes: `PascalCase` (e.g., `Tensor`, `Shape`)
   - Functions: `snake_case` (e.g., `self_attention`, `add`)
   - Templates: `<typename T>` for generic types

2. **Memory**:
   - Use `std::move` for returning large tensors
   - Move semantics for efficient transfers

3. **Parallelization**:
   - OpenMP: `#pragma omp parallel for if(condition)`
   - MPI: Wrap in `#ifdef MPI_VERSION`

## Testing

Each test file covers:
- Correctness (verify results)
- Performance (timing benchmarks)
- Scaling (thread/process counts)

See `results/*.txt` for test outputs.
