/**
 * @file benchmark_performance.cpp
 * @brief Comprehensive performance benchmark for MPI and AVX
 */

#include <mpi.h>
#include "tensor_cpp/ops_mpi.h"
#include "tensor_cpp/ops_avx.h"
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace tensor_cpp::ops;

void benchmark_mpi_matmul() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "\n========================================\n";
        std::cout << "MPI Matrix Multiplication Benchmark\n";
        std::cout << "========================================\n";
    }

    int M = 1024, N = 1024, K = 1024;
    int iters = 10;

    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(N * K, 2.0f);
    std::vector<float> C(M * N);

    // Warmup
    mpi::matmul_mpi_omp(A.data(), B.data(), C.data(), M, N, K, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iters; ++iter) {
        mpi::matmul_mpi_omp(A.data(), B.data(), C.data(), M, N, K, MPI_COMM_WORLD);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    if (rank == 0) {
        double avg_time = elapsed.count() / iters;
        double gflops = (M * N * K * 2.0 * iters) / (elapsed.count() / 1000.0) / 1e9;

        std::cout << "  Matrix size: " << M << "x" << N << "x" << K << "\n";
        std::cout << "  MPI processes: " << size << "\n";
        std::cout << "  Iterations: " << iters << "\n";
        std::cout << "  Total time: " << elapsed.count() << " ms\n";
        std::cout << "  Avg time: " << avg_time << " ms\n";
        std::cout << "  Performance: " << gflops << " GFLOPS\n";
        std::cout << "  Per-process: " << gflops / size << " GFLOPS\n";
    }
}

void benchmark_avx_matmul() {
    std::cout << "\n========================================\n";
    std::cout << "AVX Matrix Multiplication Benchmark\n";
    std::cout << "========================================\n";

    int M = 1024, N = 1024, K = 1024;
    int iters = 10;

    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(N * K, 2.0f);

    // Warmup
    auto C_warmup = avx::matmul_avx(A.data(), B.data(), M, N, K);

    // Benchmark AVX
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iters; ++iter) {
        auto C = avx::matmul_avx(A.data(), B.data(), M, N, K);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> avx_time = end - start;

    // Benchmark scalar for comparison
    std::vector<float> C_scalar(M * N);
    start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 1; ++iter) {  // Only 1 iter for scalar (too slow)
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[j * K + k];
                }
                C_scalar[i * N + j] = sum;
            }
        }
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> scalar_time = end - start;

    double avg_avx = avx_time.count() / iters;
    double avg_scalar = scalar_time.count();
    double gflops_avx = (M * N * K * 2.0 * iters) / (avx_time.count() / 1000.0) / 1e9;
    double gflops_scalar = (M * N * K * 2.0) / (scalar_time.count() / 1000.0) / 1e9;

    std::cout << "  Matrix size: " << M << "x" << N << "x" << K << "\n";
    std::cout << "  Iterations: " << iters << " (AVX), 1 (scalar)\n";
    std::cout << "\n  Scalar:\n";
    std::cout << "    Time: " << avg_scalar << " ms\n";
    std::cout << "    Performance: " << gflops_scalar << " GFLOPS\n";
    std::cout << "\n  AVX2:\n";
    std::cout << "    Time: " << avg_avx << " ms\n";
    std::cout << "    Performance: " << gflops_avx << " GFLOPS\n";
    std::cout << "    Speedup: " << avg_scalar / avg_avx << "x\n";
    std::cout << "  AVX2 supported: " << (avx::is_avx2_supported() ? "YES" : "NO") << "\n";
    std::cout << "  AVX-512 supported: " << (avx::is_avx512_supported() ? "YES" : "NO") << "\n";
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "============================================================\n";
        std::cout << "     Performance Benchmark Suite\n";
        std::cout << "============================================================\n";
        std::cout << "MPI processes: " << size << "\n";
        std::cout << "OpenMP threads: ";
        #ifdef _OPENMP
        std::cout << omp_get_max_threads() << "\n";
        #else
        std::cout << "N/A\n";
        #endif
        std::cout << "============================================================\n";
    }

    // Run AVX benchmark (rank 0 only)
    if (rank == 0) {
        benchmark_avx_matmul();
    }

    // Run MPI benchmark
    benchmark_mpi_matmul();

    if (rank == 0) {
        std::cout << "\n============================================================\n";
        std::cout << "Benchmark completed!\n";
        std::cout << "============================================================\n";
    }

    MPI_Finalize();
    return 0;
}
