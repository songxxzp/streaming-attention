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
)

# Build tensor library
add_library(tensor_cpp STATIC ${TENSOR_SOURCES})
target_link_libraries(tensor_cpp PUBLIC
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
)

# Test executables
add_executable(test_ops tests/test_ops.cpp)
target_link_libraries(test_ops PRIVATE tensor_cpp)

add_executable(test_attention tests/test_attention.cpp)
target_link_libraries(test_attention PRIVATE tensor_cpp)

add_executable(test_qwen3 tests/test_qwen3.cpp)
target_link_libraries(test_qwen3 PRIVATE tensor_cpp)

add_executable(test_qwen3_decode tests/test_qwen3_decode.cpp)
target_link_libraries(test_qwen3_decode PRIVATE tensor_cpp)

add_executable(test_qwen3_generate tests/test_qwen3_generate.cpp)
target_link_libraries(test_qwen3_generate PRIVATE tensor_cpp)

add_executable(test_qwen3_generate_with_cache tests/test_qwen3_generate_with_cache.cpp)
target_link_libraries(test_qwen3_generate_with_cache PRIVATE tensor_cpp)

add_executable(test_qwen3_verify tests/test_qwen3_verify.cpp)
target_link_libraries(test_qwen3_verify PRIVATE tensor_cpp)

add_executable(test_qwen3_logits tests/test_qwen3_logits.cpp)
target_link_libraries(test_qwen3_logits PRIVATE tensor_cpp)

add_executable(debug_kv_cache tests/debug_kv_cache.cpp)
target_link_libraries(debug_kv_cache PRIVATE tensor_cpp)

add_executable(debug_cache_data tests/debug_cache_data.cpp)
target_link_libraries(debug_cache_data PRIVATE tensor_cpp)

add_executable(benchmark_attention tests/benchmark_attention.cpp)
target_link_libraries(benchmark_attention PRIVATE tensor_cpp)

add_executable(benchmark_qwen3 tests/benchmark_qwen3.cpp)
target_link_libraries(benchmark_qwen3 PRIVATE tensor_cpp)

add_executable(test_mpi_simple tests/test_mpi_simple.cpp)
target_link_libraries(test_mpi_simple PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

add_executable(test_mpi_ops tests/test_mpi_ops.cpp)
target_link_libraries(test_mpi_ops PRIVATE tensor_cpp MPI::MPI_CXX OpenMP::OpenMP_CXX)

add_executable(test_avx_ops tests/test_avx_ops.cpp)
target_link_libraries(test_avx_ops PRIVATE tensor_cpp)

add_executable(benchmark_performance tests/benchmark_performance.cpp)
target_link_libraries(benchmark_performance PRIVATE tensor_cpp MPI::MPI_CXX OpenMP::OpenMP_CXX)

add_executable(test_tensor_parallel tests/test_tensor_parallel.cpp)
target_link_libraries(test_tensor_parallel PRIVATE tensor_cpp MPI::MPI_CXX OpenMP::OpenMP_CXX)

add_executable(benchmark_qwen3_versions tests/benchmark_qwen3_versions.cpp)
target_link_libraries(benchmark_qwen3_versions PRIVATE tensor_cpp MPI::MPI_CXX OpenMP::OpenMP_CXX)

add_executable(benchmark_qwen3_decode_versions tests/benchmark_qwen3_decode_versions.cpp)
target_link_libraries(benchmark_qwen3_decode_versions PRIVATE tensor_cpp MPI::MPI_CXX OpenMP::OpenMP_CXX)

add_executable(test_mpi_linear_debug tests/test_mpi_linear_debug.cpp)
target_link_libraries(test_mpi_linear_debug PRIVATE tensor_cpp MPI::MPI_CXX OpenMP::OpenMP_CXX)

add_executable(test_qwen3_mpi_simple tests/test_qwen3_mpi_simple.cpp)
target_link_libraries(test_qwen3_mpi_simple PRIVATE tensor_cpp MPI::MPI_CXX OpenMP::OpenMP_CXX)

add_executable(test_weights_broadcast tests/test_weights_broadcast.cpp)
target_link_libraries(test_weights_broadcast PRIVATE tensor_cpp MPI::MPI_CXX OpenMP::OpenMP_CXX)

add_executable(profile_avx2 tests/profile_avx2.cpp)
target_link_libraries(profile_avx2 PRIVATE tensor_cpp)

add_executable(simple_avx2_profile tests/simple_avx2_profile.cpp)
target_link_libraries(simple_avx2_profile PRIVATE)

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
