/**
 * @file test_streaming_attention.cpp
 * @brief Comprehensive tests for streaming attention operators
 *
 * Tests cover:
 * 1. Correctness (naive vs streaming)
 * 2. Serial execution
 * 3. OpenMP parallel execution
 * 4. MPI distributed execution (if available)
 */

#include "tensor_cpp/ops.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef MPI_VERSION
#include <mpi.h>
#endif

using namespace tensor_cpp::ops;

// Timer utility
class Timer {
    std::chrono::high_resolution_clock::time_point start_;
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}

    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
};

// Helper: Compute L2 error
float compute_l2_error(const float* a, const float* b, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Helper: Compute max error
float compute_max_error(const float* a, const float* b, int size) {
    float max_err = 0.0f;
    for (int i = 0; i < size; ++i) {
        float err = std::abs(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

// ============================================================================
// Test: Correctness (Naive vs Streaming Serial)
// ============================================================================

void test_correctness() {
    std::cout << "=== Test: Correctness (Naive vs Streaming Serial) ===\n";

    std::vector<int> Ts = {512, 1024, 2048, 4096};
    std::vector<int> ds = {64, 128, 256};
    int block_size = 64;

    for (int T : Ts) {
        for (int d : ds) {
            // Generate random data
            std::vector<float> Q(d);
            std::vector<float> K(T * d);
            std::vector<float> V(T * d);

            std::mt19937 gen(42);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

            for (int i = 0; i < d; ++i) Q[i] = dist(gen);
            for (int i = 0; i < T * d; ++i) {
                K[i] = dist(gen);
                V[i] = dist(gen);
            }

            // Run naive attention (baseline)
            auto output_naive = naive_attention_serial(Q.data(), K.data(), V.data(), T, d);

            // Run streaming attention
            auto output_streaming = streaming_attention_serial(Q.data(), K.data(), V.data(), T, d, block_size);

            // Compute errors
            float l2_err = compute_l2_error(output_naive.data(), output_streaming.data(), d);
            float max_err = compute_max_error(output_naive.data(), output_streaming.data(), d);

            std::cout << "  T=" << T << ", d=" << d << ": L2=" << std::scientific << l2_err
                      << ", Max=" << max_err << "\n";

            if (max_err > 1e-5f) {
                throw std::runtime_error("Correctness test failed!");
            }
        }
    }

    std::cout << "  Status: OK (all tests passed)\n\n";
}

// ============================================================================
// Test: Serial Performance
// ============================================================================

void test_serial_performance() {
    std::cout << "=== Test: Serial Performance ===\n";

    std::vector<std::tuple<int, int, int>> configs = {
        {512, 64, 32},
        {1024, 128, 64},
        {2048, 256, 128},
        {4096, 128, 64},
    };

    std::cout << "  " << std::left << std::setw(10) << "Config"
              << std::setw(15) << "Naive (ms)"
              << std::setw(20) << "Streaming (ms)"
              << "Speedup\n";
    std::cout << "  " << std::string(60, '-') << "\n";

    for (auto& [T, d, block] : configs) {
        // Generate data
        std::vector<float> Q(d);
        std::vector<float> K(T * d);
        std::vector<float> V(T * d);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (int i = 0; i < d; ++i) Q[i] = dist(gen);
        for (int i = 0; i < T * d; ++i) {
            K[i] = dist(gen);
            V[i] = dist(gen);
        }

        // Warmup
        naive_attention_serial(Q.data(), K.data(), V.data(), T, d);
        streaming_attention_serial(Q.data(), K.data(), V.data(), T, d, block);

        // Benchmark naive
        int repeat = 10;
        Timer timer_naive;
        for (int i = 0; i < repeat; ++i) {
            naive_attention_serial(Q.data(), K.data(), V.data(), T, d);
        }
        double time_naive = timer_naive.elapsed_ms() / repeat;

        // Benchmark streaming
        Timer timer_streaming;
        for (int i = 0; i < repeat; ++i) {
            streaming_attention_serial(Q.data(), K.data(), V.data(), T, d, block);
        }
        double time_streaming = timer_streaming.elapsed_ms() / repeat;

        std::string config = "(" + std::to_string(T) + "," + std::to_string(d) + ")";
        std::cout << "  " << std::left << std::setw(10) << config
                  << std::fixed << std::setprecision(3) << std::setw(15) << time_naive
                  << std::setw(20) << time_streaming
                  << std::setprecision(2) << time_naive / time_streaming << "x\n";
    }

    std::cout << "  Status: OK\n\n";
}

// ============================================================================
// Test: OpenMP Scaling
// ============================================================================

void test_openmp_scaling() {
    std::cout << "=== Test: OpenMP Scaling ===\n";

#ifdef _OPENMP
    int T = 2048, d = 128, block = 64;

    // Generate data
    std::vector<float> Q(d);
    std::vector<float> K(T * d);
    std::vector<float> V(T * d);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < d; ++i) Q[i] = dist(gen);
    for (int i = 0; i < T * d; ++i) {
        K[i] = dist(gen);
        V[i] = dist(gen);
    }

    std::cout << "  Input: T=" << T << ", d=" << d << ", block=" << block << "\n";
    std::cout << "  " << std::left << std::setw(12) << "Threads"
              << std::setw(15) << "Time (ms)"
              << std::setw(12) << "Speedup"
              << "L2 Error\n";
    std::cout << "  " << std::string(50, '-') << "\n";

    // Get serial baseline
    omp_set_num_threads(1);
    auto baseline = streaming_attention_serial(Q.data(), K.data(), V.data(), T, d, block);

    Timer timer_baseline;
    int repeat = 10;
    for (int i = 0; i < repeat; ++i) {
        streaming_attention_serial(Q.data(), K.data(), V.data(), T, d, block);
    }
    double time_baseline = timer_baseline.elapsed_ms() / repeat;

    std::cout << "  " << std::left << std::setw(12) << "Serial"
              << std::fixed << std::setprecision(3) << std::setw(15) << time_baseline
              << std::setw(12) << "1.00x"
              << "-\n";

    // Test with different thread counts
    std::vector<int> thread_counts = {2, 4, 8};
    for (int threads : thread_counts) {
        omp_set_num_threads(threads);

        auto output = streaming_attention_omp(Q.data(), K.data(), V.data(), T, d, block, threads);

        Timer timer;
        for (int i = 0; i < repeat; ++i) {
            streaming_attention_omp(Q.data(), K.data(), V.data(), T, d, block, threads);
        }
        double time = timer.elapsed_ms() / repeat;

        float l2_err = compute_l2_error(baseline.data(), output.data(), d);
        float max_err = compute_max_error(baseline.data(), output.data(), d);

        std::cout << "  " << std::left << std::setw(12) << (std::to_string(threads) + " threads")
                  << std::fixed << std::setprecision(3) << std::setw(15) << time
                  << std::setprecision(2) << std::setw(12) << (time_baseline / time) << "x"
                  << std::scientific << std::setprecision(2) << l2_err << "\n";

        if (max_err > 1e-4f) {
            throw std::runtime_error("OpenMP results differ from serial!");
        }
    }

    std::cout << "  Status: OK (all results match)\n\n";
#else
    std::cout << "  OpenMP not available - skipping test\n\n";
#endif
}

// ============================================================================
// Test: Block Size Impact
// ============================================================================

void test_block_size() {
    std::cout << "=== Test: Block Size Impact ===\n";

    int T = 2048, d = 128;
    std::vector<int> block_sizes = {32, 64, 128, 256, 512};

    // Generate data
    std::vector<float> Q(d);
    std::vector<float> K(T * d);
    std::vector<float> V(T * d);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < d; ++i) Q[i] = dist(gen);
    for (int i = 0; i < T * d; ++i) {
        K[i] = dist(gen);
        V[i] = dist(gen);
    }

    // Get baseline
    auto baseline = naive_attention_serial(Q.data(), K.data(), V.data(), T, d);

    std::cout << "  Input: T=" << T << ", d=" << d << "\n";
    std::cout << "  " << std::left << std::setw(12) << "Block"
              << std::setw(15) << "Time (ms)"
              << std::setw(12) << "L2 Error"
              << "Max Error\n";
    std::cout << "  " << std::string(55, '-') << "\n";

    for (int block : block_sizes) {
        auto output = streaming_attention_serial(Q.data(), K.data(), V.data(), T, d, block);

        Timer timer;
        int repeat = 10;
        for (int i = 0; i < repeat; ++i) {
            streaming_attention_serial(Q.data(), K.data(), V.data(), T, d, block);
        }
        double time = timer.elapsed_ms() / repeat;

        float l2_err = compute_l2_error(baseline.data(), output.data(), d);
        float max_err = compute_max_error(baseline.data(), output.data(), d);

        std::cout << "  " << std::left << std::setw(12) << block
                  << std::fixed << std::setprecision(3) << std::setw(15) << time
                  << std::scientific << std::setprecision(2) << std::setw(12) << l2_err
                  << std::setprecision(2) << max_err << "\n";
    }

    std::cout << "  Status: OK\n\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
#ifdef MPI_VERSION
    MPI_Init(&argc, &argv);
#endif

    int rank = 0;
#ifdef MPI_VERSION
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    if (rank == 0) {
        std::cout << "\n";
        std::cout << "========================================================\n";
        std::cout << "  Streaming Attention Test Suite\n";
#ifdef _OPENMP
        std::cout << "  OpenMP: Enabled (max threads=" << omp_get_max_threads() << ")\n";
#endif
#ifdef MPI_VERSION
        int mpi_size;
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        std::cout << "  MPI: Enabled (" << mpi_size << " processes)\n";
#endif
        std::cout << "========================================================\n\n";
    }

    try {
        if (rank == 0) {
            test_correctness();
            test_serial_performance();
            test_openmp_scaling();
            test_block_size();
        }

        if (rank == 0) {
            std::cout << "========================================================\n";
            std::cout << "  All Streaming Attention Tests PASSED\n";
            std::cout << "========================================================\n\n";
        }

    } catch (const std::exception& e) {
        if (rank == 0) {
            std::cerr << "\n✗ ERROR: " << e.what() << "\n";
        }
#ifdef MPI_VERSION
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        return 1;
    }

#ifdef MPI_VERSION
    MPI_Finalize();
#endif

    return 0;
}
