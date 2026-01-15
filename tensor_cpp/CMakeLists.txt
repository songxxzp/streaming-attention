cmake_minimum_required(VERSION 3.16)
project(tensor_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find required packages
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

# Define MPI_VERSION since it's not automatically defined by CMake
if(MPI_CXX_FOUND)
    add_definitions(-DMPI_VERSION=3)
endif()

# Compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files
set(TENSOR_SOURCES
    src/tensor.cpp
    src/ops.cpp
    src/qwen3_ops.cpp
    src/qwen3_loader.cpp
    src/ops_mpi.cpp
    src/qwen3_ops_mpi.cpp
    src/ops_avx.cpp
    src/qwen3_ops_avx.cpp
    src/qwen3_ops_mpi_avx.cpp
    src/qwen3_tensor_parallel.cpp
    src/attention_avx.cpp
)

# Build tensor library
add_library(tensor_cpp STATIC ${TENSOR_SOURCES})
target_link_libraries(tensor_cpp PUBLIC
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
)

# Add AVX2 compile flags for AVX source files
set_source_files_properties(src/ops.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
set_source_files_properties(src/ops_avx.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
set_source_files_properties(src/qwen3_ops_avx.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
set_source_files_properties(src/qwen3_ops_mpi_avx.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
set_source_files_properties(src/attention_avx.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")

# ==============================================================================
# Test Executables - Organized by category
# ==============================================================================

# Unit Tests
add_executable(test_simple tests/unit/test_simple.cpp)
target_link_libraries(test_simple PRIVATE tensor_cpp)

add_executable(test_ops tests/unit/test_ops.cpp)
target_link_libraries(test_ops PRIVATE tensor_cpp)

add_executable(test_attention tests/unit/test_attention.cpp)
target_link_libraries(test_attention PRIVATE tensor_cpp)

add_executable(test_streaming_attention tests/unit/test_streaming_attention.cpp)
target_link_libraries(test_streaming_attention PRIVATE tensor_cpp)

add_executable(test_avx_ops tests/unit/test_avx_ops.cpp)
target_link_libraries(test_avx_ops PRIVATE tensor_cpp)

add_executable(test_mpi_simple tests/unit/test_mpi_simple.cpp)
target_link_libraries(test_mpi_simple PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

add_executable(test_mpi_ops tests/unit/test_mpi_ops.cpp)
target_link_libraries(test_mpi_ops PRIVATE tensor_cpp MPI::MPI_CXX OpenMP::OpenMP_CXX)

add_executable(test_tensor_parallel tests/unit/test_tensor_parallel.cpp)
target_link_libraries(test_tensor_parallel PRIVATE tensor_cpp MPI::MPI_CXX OpenMP::OpenMP_CXX)

add_executable(test_weights_broadcast tests/unit/test_weights_broadcast.cpp)
target_link_libraries(test_weights_broadcast PRIVATE tensor_cpp MPI::MPI_CXX OpenMP::OpenMP_CXX)

# Integration Tests
add_executable(test_qwen3 tests/integration/test_qwen3.cpp)
target_link_libraries(test_qwen3 PRIVATE tensor_cpp)

add_executable(test_qwen3_verify tests/integration/test_qwen3_verify.cpp)
target_link_libraries(test_qwen3_verify PRIVATE tensor_cpp)

add_executable(test_qwen3_decode tests/integration/test_qwen3_decode.cpp)
target_link_libraries(test_qwen3_decode PRIVATE tensor_cpp)

add_executable(test_qwen3_generate tests/integration/test_qwen3_generate.cpp)
target_link_libraries(test_qwen3_generate PRIVATE tensor_cpp)

add_executable(test_qwen3_generate_with_cache tests/integration/test_qwen3_generate_with_cache.cpp)
target_link_libraries(test_qwen3_generate_with_cache PRIVATE tensor_cpp)

add_executable(test_qwen3_mpi_simple tests/integration/test_qwen3_mpi_simple.cpp)
target_link_libraries(test_qwen3_mpi_simple PRIVATE tensor_cpp MPI::MPI_CXX OpenMP::OpenMP_CXX)

# Benchmark Tests
add_executable(benchmark_attention tests/benchmark/benchmark_attention.cpp)
target_link_libraries(benchmark_attention PRIVATE tensor_cpp)

add_executable(benchmark_qwen3 tests/benchmark/benchmark_qwen3.cpp)
target_link_libraries(benchmark_qwen3 PRIVATE tensor_cpp)

add_executable(benchmark_performance tests/benchmark/benchmark_performance.cpp)
target_link_libraries(benchmark_performance PRIVATE tensor_cpp MPI::MPI_CXX OpenMP::OpenMP_CXX)

add_executable(benchmark_qwen3_versions tests/benchmark/benchmark_qwen3_versions.cpp)
target_link_libraries(benchmark_qwen3_versions PRIVATE tensor_cpp MPI::MPI_CXX OpenMP::OpenMP_CXX)

add_executable(benchmark_avx2_versions tests/benchmark/benchmark_avx2_versions.cpp)
target_link_libraries(benchmark_avx2_versions PRIVATE tensor_cpp)

# Validation/Profiling Tests
add_executable(torch_validation tests/validation/torch_validation.cpp)
target_link_libraries(torch_validation PRIVATE tensor_cpp)

add_executable(test_align_qwen3 tests/validation/test_align_qwen3.cpp)
target_link_libraries(test_align_qwen3 PRIVATE tensor_cpp)

add_executable(profile_avx2 tests/validation/profile_avx2.cpp)
target_link_libraries(profile_avx2 PRIVATE tensor_cpp)
set_source_files_properties(tests/validation/profile_avx2.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")

# Example
add_executable(basic_usage examples/basic_usage.cpp)
target_link_libraries(basic_usage PRIVATE tensor_cpp)

# Installation
install(TARGETS test_ops test_attention test_qwen3 benchmark_attention benchmark_qwen3
        RUNTIME DESTINATION bin)

# Print configuration summary
message(STATUS "")
message(STATUS "========================================")
message(STATUS "  Tensor C++ Library Configuration")
message(STATUS "========================================")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenMP Found: ${OpenMP_CXX_FOUND}")
message(STATUS "OpenMP Version: ${OpenMP_CXX_VERSION}")
message(STATUS "MPI Found: ${MPI_CXX_FOUND}")
message(STATUS "MPI Compiler: ${MPI_CXX_COMPILER}")
message(STATUS "MPI Version: ${MPI_CXX_VERSION}")
message(STATUS "========================================")
message(STATUS "")
# 临时添加到CMakeLists.txt
add_executable(test_kv_cache_simple tests/test_kv_cache_simple.cpp)
target_link_libraries(test_kv_cache_simple PRIVATE tensor_cpp)
