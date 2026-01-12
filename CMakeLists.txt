cmake_minimum_required(VERSION 3.10)
project(StreamingAttention)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall")

# Find OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Find MPI
find_package(MPI)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Phase 1: Serial implementations
# Disabled - test_correctness.cpp not found
# add_executable(test_correctness
#     test_correctness.cpp
#     attention/naive_serial.cpp
#     attention/streaming_serial.cpp
# )

# Phase 2: OpenMP version (to be implemented)
# Disabled - test_omp.cpp not found
# if(OpenMP_CXX_FOUND)
#     add_executable(test_omp
#         test_omp.cpp
#         attention/naive_serial.cpp
#         attention/streaming_serial.cpp
#         attention/streaming_omp.cpp
#     )
#     target_link_libraries(test_omp OpenMP::OpenMP_CXX)
# endif()

# Phase 3: MPI + OpenMP version (to be implemented)
# Disabled - test_mpi.cpp not found
# if(MPI_FOUND)
#     add_executable(test_mpi
#         test_mpi.cpp
#         attention/naive_serial.cpp
#         attention/streaming_serial.cpp
#         attention/streaming_mpi.cpp
#     )
#     target_link_libraries(test_mpi MPI::MPI_CXX)
#     if(OpenMP_CXX_FOUND)
#         target_link_libraries(test_mpi OpenMP::OpenMP_CXX)
#     endif()
# endif()
