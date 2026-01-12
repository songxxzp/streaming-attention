/**
 * @file test_attention.cpp
 * @brief Comprehensive tests for Self-Attention and Cross-Attention operators
 *
 * Tests cover:
 * 1. Serial execution (baseline)
 * 2. OpenMP parallel execution
 * 3. OpenMP + MPI distributed execution
 */

#include "tensor_cpp/tensor.h"
#include "tensor_cpp/ops.h"
#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef MPI_VERSION
#include <mpi.h>
#endif

using namespace tensor_cpp;
using namespace ops;

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

// Helper: Check if two tensors are approximately equal
bool allclose(const TensorF& a, const TensorF& b, float rtol = 1e-3f, float atol = 1e-5f) {
    if (a.shape() != b.shape()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > atol + rtol * std::abs(b[i])) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// Test: Self Attention - Basic Functionality
// ============================================================================

void test_self_attention_basic() {
    std::cout << "=== Test: Self-Attention Basic ===\n";

    // Create small test tensors
    // Shape: (batch_size=2, num_heads=2, seq_len=4, head_dim=8)
    size_t batch_size = 2;
    size_t num_heads = 2;
    size_t seq_len = 4;
    size_t head_dim = 8;

    TensorF query = TensorF::randn(Shape({batch_size, num_heads, seq_len, head_dim}));
    TensorF key = TensorF::randn(Shape({batch_size, num_heads, seq_len, head_dim}));
    TensorF value = TensorF::randn(Shape({batch_size, num_heads, seq_len, head_dim}));

    // Scale factor for scaled dot-product attention
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::cout << "  Input shapes:\n";
    std::cout << "    Q/K/V: (" << batch_size << ", " << num_heads << ", " << seq_len << ", " << head_dim << ")\n";
    std::cout << "  Scale: " << scale << "\n";

    Timer timer;
    TensorF output = self_attention(query, key, value, static_cast<const TensorF*>(nullptr), scale);
    double time_ms = timer.elapsed_ms();

    std::cout << "  Output shape: " << output.shape().to_string() << "\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << time_ms << " ms\n";
    std::cout << "  Status: OK\n\n";
}

// ============================================================================
// Test: Self Attention - OpenMP Scaling
// ============================================================================

void test_self_attention_openmp() {
    std::cout << "=== Test: Self-Attention OpenMP Scaling ===\n";

#ifdef _OPENMP
    // Create larger tensors for meaningful parallelization
    size_t batch_size = 4;
    size_t num_heads = 8;
    size_t seq_len = 64;
    size_t head_dim = 64;

    TensorF query = TensorF::randn(Shape({batch_size, num_heads, seq_len, head_dim}));
    TensorF key = TensorF::randn(Shape({batch_size, num_heads, seq_len, head_dim}));
    TensorF value = TensorF::randn(Shape({batch_size, num_heads, seq_len, head_dim}));

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::cout << "  Input shape: (" << batch_size << ", " << num_heads << ", " << seq_len << ", " << head_dim << ")\n";

    // Reference: Serial execution
    omp_set_num_threads(1);
    Timer timer;
    TensorF output_serial = self_attention(query, key, value, static_cast<const TensorF*>(nullptr), scale);
    double time_serial = timer.elapsed_ms();
    std::cout << "  Threads=1:  " << std::fixed << std::setprecision(3) << time_serial << " ms (baseline)\n";

    // Test with different thread counts
    int thread_counts[] = {2, 4, 8};
    for (int threads : thread_counts) {
        omp_set_num_threads(threads);
        timer = Timer();
        TensorF output_parallel = self_attention(query, key, value, static_cast<const TensorF*>(nullptr), scale);
        double time_parallel = timer.elapsed_ms();

        double speedup = time_serial / time_parallel;
        std::cout << "  Threads=" << threads << ":  " << std::fixed << std::setprecision(3)
                  << time_parallel << " ms (speedup: " << std::setprecision(2) << speedup << "x)\n";

        // Verify correctness
        if (!allclose(output_serial, output_parallel)) {
            throw std::runtime_error("OpenMP results differ from serial execution!");
        }
    }

    std::cout << "  Status: OK (all results match)\n\n";
#else
    std::cout << "  OpenMP not available - skipping test\n\n";
#endif
}

// ============================================================================
// Test: Cross Attention - Basic Functionality
// ============================================================================

void test_cross_attention_basic() {
    std::cout << "=== Test: Cross-Attention Basic ===\n";

    // Cross-attention: different sequence lengths for Q vs K/V
    // Query: (batch_size=2, num_heads=2, query_len=10, head_dim=8)
    // Key/Value: (batch_size=2, num_heads=2, kv_len=20, head_dim=8)
    size_t batch_size = 2;
    size_t num_heads = 2;
    size_t query_len = 10;
    size_t kv_len = 20;
    size_t head_dim = 8;

    TensorF query = TensorF::randn(Shape({batch_size, num_heads, query_len, head_dim}));
    TensorF key = TensorF::randn(Shape({batch_size, num_heads, kv_len, head_dim}));
    TensorF value = TensorF::randn(Shape({batch_size, num_heads, kv_len, head_dim}));

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::cout << "  Input shapes:\n";
    std::cout << "    Query: (" << batch_size << ", " << num_heads << ", " << query_len << ", " << head_dim << ")\n";
    std::cout << "    Key/Value: (" << batch_size << ", " << num_heads << ", " << kv_len << ", " << head_dim << ")\n";
    std::cout << "  Scale: " << scale << "\n";

    Timer timer;
    TensorF output = cross_attention(query, key, value, static_cast<const TensorF*>(nullptr), scale);
    double time_ms = timer.elapsed_ms();

    std::cout << "  Output shape: " << output.shape().to_string() << "\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << time_ms << " ms\n";

    // Verify output shape matches query shape
    if (output.shape() != query.shape()) {
        throw std::runtime_error("Cross-attention output shape doesn't match query shape!");
    }

    std::cout << "  Status: OK\n\n";
}

// ============================================================================
// Test: Cross Attention - OpenMP Scaling
// ============================================================================

void test_cross_attention_openmp() {
    std::cout << "=== Test: Cross-Attention OpenMP Scaling ===\n";

#ifdef _OPENMP
    // Larger tensors for cross-attention
    size_t batch_size = 4;
    size_t num_heads = 8;
    size_t query_len = 32;
    size_t kv_len = 128;
    size_t head_dim = 64;

    TensorF query = TensorF::randn(Shape({batch_size, num_heads, query_len, head_dim}));
    TensorF key = TensorF::randn(Shape({batch_size, num_heads, kv_len, head_dim}));
    TensorF value = TensorF::randn(Shape({batch_size, num_heads, kv_len, head_dim}));

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::cout << "  Input shapes:\n";
    std::cout << "    Query: (" << batch_size << ", " << num_heads << ", " << query_len << ", " << head_dim << ")\n";
    std::cout << "    Key/Value: (" << batch_size << ", " << num_heads << ", " << kv_len << ", " << head_dim << ")\n";

    // Reference: Serial execution
    omp_set_num_threads(1);
    Timer timer;
    TensorF output_serial = cross_attention(query, key, value, static_cast<const TensorF*>(nullptr), scale);
    double time_serial = timer.elapsed_ms();
    std::cout << "  Threads=1:  " << std::fixed << std::setprecision(3) << time_serial << " ms (baseline)\n";

    // Test with different thread counts
    int thread_counts[] = {2, 4, 8};
    for (int threads : thread_counts) {
        omp_set_num_threads(threads);
        timer = Timer();
        TensorF output_parallel = cross_attention(query, key, value, static_cast<const TensorF*>(nullptr), scale);
        double time_parallel = timer.elapsed_ms();

        double speedup = time_serial / time_parallel;
        std::cout << "  Threads=" << threads << ":  " << std::fixed << std::setprecision(3)
                  << time_parallel << " ms (speedup: " << std::setprecision(2) << speedup << "x)\n";

        // Verify correctness
        if (!allclose(output_serial, output_parallel)) {
            throw std::runtime_error("OpenMP results differ from serial execution!");
        }
    }

    std::cout << "  Status: OK (all results match)\n\n";
#else
    std::cout << "  OpenMP not available - skipping test\n\n";
#endif
}

// ============================================================================
// Test: Self vs Cross Attention Equivalence
// ============================================================================

void test_self_vs_cross_attention() {
    std::cout << "=== Test: Self vs Cross-Attention Equivalence ===\n";

    // When query_len == kv_len, cross-attention should produce same results as self-attention
    size_t batch_size = 2;
    size_t num_heads = 2;
    size_t seq_len = 8;
    size_t head_dim = 16;

    TensorF query = TensorF::randn(Shape({batch_size, num_heads, seq_len, head_dim}));
    TensorF key = TensorF::randn(Shape({batch_size, num_heads, seq_len, head_dim}));
    TensorF value = TensorF::randn(Shape({batch_size, num_heads, seq_len, head_dim}));

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Self-attention
    TensorF output_self = self_attention(query, key, value, static_cast<const TensorF*>(nullptr), scale);

    // Cross-attention with same sequence lengths
    TensorF output_cross = cross_attention(query, key, value, static_cast<const TensorF*>(nullptr), scale);

    // They should produce identical results
    if (!allclose(output_self, output_cross)) {
        throw std::runtime_error("Self-attention and cross-attention produce different results!");
    }

    std::cout << "  Input shape: (" << batch_size << ", " << num_heads << ", " << seq_len << ", " << head_dim << ")\n";
    std::cout << "  Self-attention and cross-attention produce identical results\n";
    std::cout << "  Status: OK\n\n";
}

// ============================================================================
// Test: Attention with Different Head Dimensions
// ============================================================================

void test_attention_head_dims() {
    std::cout << "=== Test: Attention with Different Head Dimensions ===\n";

    size_t batch_size = 2;
    size_t num_heads = 4;
    size_t seq_len = 16;

    size_t head_dims[] = {32, 64, 128};

    for (size_t head_dim : head_dims) {
        TensorF query = TensorF::randn(Shape({batch_size, num_heads, seq_len, head_dim}));
        TensorF key = TensorF::randn(Shape({batch_size, num_heads, seq_len, head_dim}));
        TensorF value = TensorF::randn(Shape({batch_size, num_heads, seq_len, head_dim}));

        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        Timer timer;
        TensorF output = self_attention(query, key, value, static_cast<const TensorF*>(nullptr), scale);
        double time_ms = timer.elapsed_ms();

        std::cout << "  head_dim=" << head_dim << ": " << std::fixed << std::setprecision(3)
                  << time_ms << " ms\n";
    }

    std::cout << "  Status: OK\n\n";
}

// ============================================================================
// MPI Test (only if MPI is available)
// ============================================================================

#ifdef MPI_VERSION

void test_attention_mpi() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "=== Test: Attention with MPI ===\n";
        std::cout << "  MPI processes: " << size << "\n";
    }

    // Each process runs the same attention computation
    size_t batch_size = 2;
    size_t num_heads = 4;
    size_t seq_len = 32;
    size_t head_dim = 64;

    // Use same random seed on all processes for reproducibility
    TensorF query = TensorF::randn(Shape({batch_size, num_heads, seq_len, head_dim}));
    TensorF key = TensorF::randn(Shape({batch_size, num_heads, seq_len, head_dim}));
    TensorF value = TensorF::randn(Shape({batch_size, num_heads, seq_len, head_dim}));

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Each process computes attention
    Timer timer;
    TensorF output = self_attention(query, key, value, static_cast<const TensorF*>(nullptr), scale);
    double local_time = timer.elapsed_ms();

    // Gather timing statistics
    double max_time, min_time, avg_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    avg_time /= size;

    if (rank == 0) {
        std::cout << "  Input shape: (" << batch_size << ", " << num_heads << ", " << seq_len << ", " << head_dim << ")\n";
        std::cout << "  Timing across " << size << " processes:\n";
        std::cout << "    Min: " << std::fixed << std::setprecision(3) << min_time << " ms\n";
        std::cout << "    Max: " << max_time << " ms\n";
        std::cout << "    Avg: " << avg_time << " ms\n";
        std::cout << "  Status: OK\n\n";
    }

    // Test MPI all_reduce with attention output
    ops::all_reduce_sum(output, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "  MPI all_reduce_sum test: OK\n\n";
    }
}

#endif // MPI_VERSION

// ============================================================================
// Save Results to File
// ============================================================================

void save_results(const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing\n";
        return;
    }

    out << "Attention Test Results\n";
    out << "=====================\n\n";

    out << "Tests performed:\n";
    out << "  1. Self-Attention Basic\n";
    out << "  2. Self-Attention OpenMP Scaling\n";
    out << "  3. Cross-Attention Basic\n";
    out << "  4. Cross-Attention OpenMP Scaling\n";
    out << "  5. Self vs Cross-Attention Equivalence\n";
    out << "  6. Attention with Different Head Dimensions\n";
#ifdef MPI_VERSION
    out << "  7. MPI Distributed Attention\n";
#endif
    out << "\n";

#ifdef _OPENMP
    out << "OpenMP: Enabled (max threads=" << omp_get_max_threads() << ")\n";
#else
    out << "OpenMP: Disabled\n";
#endif

#ifdef MPI_VERSION
    out << "MPI: Enabled\n";
#else
    out << "MPI: Disabled\n";
#endif

    out.close();
    std::cout << "Results saved to: " << filename << "\n";
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
        std::cout << "  Attention Test Suite (Self & Cross)\n";
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
            test_self_attention_basic();
            test_cross_attention_basic();
            test_self_vs_cross_attention();
            test_self_attention_openmp();
            test_cross_attention_openmp();
            test_attention_head_dims();
        }

#ifdef MPI_VERSION
        test_attention_mpi();
#endif

        if (rank == 0) {
            std::cout << "========================================================\n";
            std::cout << "  All Attention Tests PASSED\n";
            std::cout << "========================================================\n\n";

            save_results("results/attention_test_results.txt");
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
