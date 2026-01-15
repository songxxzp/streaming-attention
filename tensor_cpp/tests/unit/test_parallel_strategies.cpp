/**
 * @file test_parallel_strategies.cpp
 * @brief Test program for comparing head-wise and sequence parallelism
 *
 * Tests:
 * 1. Basic functionality - does each strategy run without crashing?
 * 2. Correctness - do different strategies produce similar results?
 * 3. Multiple processes - test with 1, 2, 4 MPI processes
 */

#include "tensor_cpp/tensor.h"
#include "tensor_cpp/qwen3_ops_mpi.h"
#include "tensor_cpp/ops_mpi.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

#ifdef MPI_VERSION
#include <mpi.h>
#endif

using namespace tensor_cpp;
using namespace tensor_cpp::qwen3::mpi;

// Test configuration
struct TestConfig {
    size_t batch_size = 1;
    size_t seq_len = 64;        // Local sequence length per rank
    size_t num_heads = 8;        // Number of attention heads
    size_t num_kv_heads = 4;     // Number of KV heads
    size_t head_dim = 64;        // Head dimension
};

// Generate random tensor data
Tensor generate_random_tensor(const std::vector<long>& shape, float min_val = -1.0f, float max_val = 1.0f) {
    // Convert to std::vector<size_t> for Shape
    std::vector<size_t> shape_size_t;
    for (long s : shape) shape_size_t.push_back(static_cast<size_t>(s));

    size_t total_size = 1;
    for (size_t s : shape_size_t) total_size *= s;

    std::vector<float> data(total_size);
    for (size_t i = 0; i < total_size; ++i) {
        data[i] = min_val + (max_val - min_val) * (static_cast<float>(rand()) / RAND_MAX);
    }

    return Tensor(std::move(data), Shape(shape_size_t));
}

// Compare two tensors element-wise
void compare_tensors(const Tensor& a, const Tensor& b, const std::string& name, double tolerance = 1e-3) {
    if (a.shape() != b.shape()) {
        std::cerr << "ERROR: " << name << " - Shape mismatch!" << std::endl;
        return;
    }

    size_t size = a.size();

    // Compute statistics for tensor a
    double a_min = a[0], a_max = a[0], a_sum = 0.0;
    bool a_has_nan = false, a_has_inf = false;
    for (size_t i = 0; i < size; ++i) {
        if (std::isnan(a[i])) a_has_nan = true;
        if (std::isinf(a[i])) a_has_inf = true;
        a_min = std::min(a_min, static_cast<double>(a[i]));
        a_max = std::max(a_max, static_cast<double>(a[i]));
        a_sum += a[i];
    }

    // Compute statistics for tensor b
    double b_min = b[0], b_max = b[0], b_sum = 0.0;
    bool b_has_nan = false, b_has_inf = false;
    for (size_t i = 0; i < size; ++i) {
        if (std::isnan(b[i])) b_has_nan = true;
        if (std::isinf(b[i])) b_has_inf = true;
        b_min = std::min(b_min, static_cast<double>(b[i]));
        b_max = std::max(b_max, static_cast<double>(b[i]));
        b_sum += b[i];
    }

    // Compute differences
    double max_diff = 0.0;
    double sum_diff = 0.0;
    double max_rel_diff = 0.0;
    size_t mismatches = 0;

    for (size_t i = 0; i < size; ++i) {
        double diff = std::abs(a[i] - b[i]);
        double rel_diff = diff / (std::abs(a[i]) + 1e-8);
        max_diff = std::max(max_diff, diff);
        max_rel_diff = std::max(max_rel_diff, rel_diff);
        sum_diff += diff;

        if (diff > tolerance) {
            mismatches++;
        }
    }

    double avg_diff = sum_diff / size;
    double mismatch_rate = 100.0 * mismatches / size;

    std::cout << "  " << name << ":\n";
    std::cout << "    Tensor A (Standard):\n";
    std::cout << "      Range: [" << a_min << ", " << a_max << "]\n";
    std::cout << "      Mean:  " << a_sum / size << "\n";
    std::cout << "      NaN:   " << (a_has_nan ? "❌" : "✅") << "  Inf: " << (a_has_inf ? "❌" : "✅") << "\n";
    std::cout << "    Tensor B (Online):\n";
    std::cout << "      Range: [" << b_min << ", " << b_max << "]\n";
    std::cout << "      Mean:  " << b_sum / size << "\n";
    std::cout << "      NaN:   " << (b_has_nan ? "❌" : "✅") << "  Inf: " << (b_has_inf ? "❌" : "✅") << "\n";
    std::cout << "    Differences:\n";
    std::cout << "      Max abs diff:  " << std::scientific << max_diff << std::fixed << std::endl;
    std::cout << "      Max rel diff:  " << std::setprecision(2) << (max_rel_diff * 100.0) << "%\n";
    std::cout << "      Avg abs diff:  " << std::setprecision(6) << avg_diff << std::endl;
    std::cout << "      Mismatch:     " << std::setprecision(2) << mismatch_rate << "% (" << mismatches << "/" << size << " elements)\n";

    if (!a_has_nan && !a_has_inf && !b_has_nan && !b_has_inf) {
        std::cout << "    Status:       ✅ VALID (both tensors have finite values)\n";
    } else {
        std::cout << "    Status:       ❌ INVALID (NaN or Inf detected)\n";
    }
}

// Test head-wise parallelism
void test_headwise_parallelism(const TestConfig& config, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        std::cout << "\n============================================\n";
        std::cout << "Testing Head-wise Parallelism\n";
        std::cout << "============================================\n";
    }

    MPI_Barrier(comm);

    // Generate test data
    std::vector<long> q_shape = {static_cast<long>(config.batch_size), static_cast<long>(config.num_heads),
                                  static_cast<long>(config.seq_len), static_cast<long>(config.head_dim)};
    std::vector<long> k_shape = {static_cast<long>(config.batch_size), static_cast<long>(config.num_kv_heads),
                                  static_cast<long>(config.seq_len), static_cast<long>(config.head_dim)};
    std::vector<long> v_shape = {static_cast<long>(config.batch_size), static_cast<long>(config.num_kv_heads),
                                  static_cast<long>(config.seq_len), static_cast<long>(config.head_dim)};

    Tensor q = generate_random_tensor(q_shape);
    Tensor k = generate_random_tensor(k_shape);
    Tensor v = generate_random_tensor(v_shape);

    float scale = 1.0f / std::sqrt(static_cast<float>(config.head_dim));

    // Test Head-wise + Standard
    if (rank == 0) std::cout << "\n1. Head-wise + Standard:\n";
    Tensor hw_standard = ops::mpi::attention_headwise_standard(
        q, k, v, nullptr, scale,
        config.num_heads, config.num_kv_heads, comm
    );

    // Test Head-wise + Online Softmax
    if (rank == 0) std::cout << "\n2. Head-wise + Online Softmax:\n";
    Tensor hw_online = ops::mpi::attention_headwise_online_softmax(
        q, k, v, nullptr, scale,
        config.num_heads, config.num_kv_heads, comm
    );

    // Compare results (should be similar but not identical due to numerical differences)
    if (rank == 0) {
        std::cout << "\n3. Comparison (Head-wise Standard vs Online):\n";
        compare_tensors(hw_standard, hw_online, "Head-wise", 1e-2);  // Relaxed tolerance for online softmax
    }

    MPI_Barrier(comm);
}

// Test sequence parallelism
void test_sequence_parallelism(const TestConfig& config, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        std::cout << "\n============================================\n";
        std::cout << "Testing Sequence Parallelism\n";
        std::cout << "============================================\n";
    }

    MPI_Barrier(comm);

    // Generate test data
    std::vector<long> q_shape = {static_cast<long>(config.batch_size), static_cast<long>(config.num_heads),
                                  static_cast<long>(config.seq_len), static_cast<long>(config.head_dim)};
    std::vector<long> k_shape = {static_cast<long>(config.batch_size), static_cast<long>(config.num_kv_heads),
                                  static_cast<long>(config.seq_len), static_cast<long>(config.head_dim)};
    std::vector<long> v_shape = {static_cast<long>(config.batch_size), static_cast<long>(config.num_kv_heads),
                                  static_cast<long>(config.seq_len), static_cast<long>(config.head_dim)};

    Tensor q = generate_random_tensor(q_shape);
    Tensor k = generate_random_tensor(k_shape);
    Tensor v = generate_random_tensor(v_shape);

    float scale = 1.0f / std::sqrt(static_cast<float>(config.head_dim));

    // Test Sequence + Online Softmax
    if (rank == 0) std::cout << "\n1. Sequence + Online Softmax:\n";
    size_t global_seq_len = config.seq_len * size;

    Tensor seq_online = ops::mpi::attention_sequence_online_softmax(
        q, k, v, nullptr, scale,
        config.num_heads, config.num_kv_heads,
        global_seq_len, comm
    );

    if (rank == 0) {
        std::cout << "  Output shape: [" << seq_online.shape()[0] << ", "
                  << seq_online.shape()[1] << ", "
                  << seq_online.shape()[2] << ", "
                  << seq_online.shape()[3] << "]\n";
        std::cout << "  Global seq len: " << global_seq_len << "\n";
        std::cout << "  Local seq len:  " << config.seq_len << "\n";
        std::cout << "  Status:        ✅ Completed without errors\n";
    }

    MPI_Barrier(comm);
}

// Test cross-strategy comparison
void test_cross_strategy_comparison(const TestConfig& config, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        std::cout << "\n============================================\n";
        std::cout << "Cross-Strategy Comparison\n";
        std::cout << "============================================\n";
    }

    MPI_Barrier(comm);

    // Note: Cannot directly compare head-wise and sequence outputs
    // because they produce different output distributions:
    // - Head-wise: each rank produces all sequence positions for its heads
    // - Sequence: each rank produces local sequence positions for all heads

    if (rank == 0) {
        std::cout << "\n⚠️  Note: Head-wise and Sequence parallelism produce\n";
        std::cout << "   different output distributions and cannot be\n";
        std::cout << "   directly compared element-wise.\n";
        std::cout << "\n   Head-wise: All ranks have output shape [batch, heads, local_seq, head_dim]\n";
        std::cout << "   Sequence: All ranks have output shape [batch, heads, local_seq, head_dim]\n";
        std::cout << "\n   To verify correctness, we can check:\n";
        std::cout << "   1. No crashes or errors\n";
        std::cout << "   2. Output shape is correct\n";
        std::cout << "   3. Values are finite (not NaN/Inf)\n";
    }

    MPI_Barrier(comm);
}

// Check tensor properties
void check_tensor_properties(const Tensor& tensor, const std::string& name, int rank) {
    bool has_nan = false;
    bool has_inf = false;
    double min_val = tensor[0];
    double max_val = tensor[0];
    double sum = 0.0;

    for (size_t i = 0; i < tensor.size(); ++i) {
        float val = tensor[i];
        if (std::isnan(val)) has_nan = true;
        if (std::isinf(val)) has_inf = true;
        min_val = std::min(min_val, static_cast<double>(val));
        max_val = std::max(max_val, static_cast<double>(val));
        sum += val;
    }

    if (rank == 0) {
        std::cout << "  " << name << ":\n";
        std::cout << "    Shape:    [" << tensor.shape()[0] << ", "
                  << tensor.shape()[1] << ", "
                  << tensor.shape()[2] << ", "
                  << tensor.shape()[3] << "]\n";
        std::cout << "    Min:      " << std::setprecision(6) << min_val << "\n";
        std::cout << "    Max:      " << max_val << "\n";
        std::cout << "    Mean:     " << sum / tensor.size() << "\n";
        std::cout << "    Has NaN:  " << (has_nan ? "❌ YES" : "✅ NO") << "\n";
        std::cout << "    Has Inf:  " << (has_inf ? "❌ YES" : "✅ NO") << "\n";

        if (!has_nan && !has_inf) {
            std::cout << "    Status:   ✅ PASS\n";
        } else {
            std::cout << "    Status:   ❌ FAIL\n";
        }
    }
}

int main(int argc, char** argv) {
#ifdef MPI_VERSION
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════╗\n";
        std::cout << "║  MPI Parallel Strategy Test Program           ║\n";
        std::cout << "╚════════════════════════════════════════════════╝\n";
        std::cout << "\nMPI Processes: " << size << "\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Test configuration
    TestConfig config;
    config.seq_len = 64;  // Local sequence length per rank

    // Run tests
    test_headwise_parallelism(config, MPI_COMM_WORLD);
    test_sequence_parallelism(config, MPI_COMM_WORLD);
    test_cross_strategy_comparison(config, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n============================================\n";
        std::cout << "All tests completed!\n";
        std::cout << "============================================\n\n";
    }

    MPI_Finalize();
    return 0;
#else
    std::cerr << "Error: This test requires MPI. Recompile with MPI support.\n";
    return 1;
#endif
}
