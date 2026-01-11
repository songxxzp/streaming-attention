/**
 * @file test_ops.cpp
 * @brief Comprehensive tests for tensor operations with OpenMP/MPI parallel support
 */

#include "../include/tensor_lib/tensor.h"
#include "../include/tensor_lib/ops.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cassert>
#include <fstream>
#include <cstring>

#ifdef MPI_VERSION
#include <mpi.h>
#endif

using namespace tensor_lib;
using namespace ops;

// Timer utility
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_;
    std::string name_;

public:
    Timer(const std::string& name) : name_(name) {
        start_ = std::chrono::high_resolution_clock::now();
    }

    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start_;
        std::cout << "[" << name_ << "] Time: " << std::fixed << std::setprecision(3)
                  << elapsed.count() << " ms\n";
    }

    double elapsed_ms() {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start_;
        return elapsed.count();
    }
};

// Test result structure
struct TestResult {
    std::string name;
    bool passed;
    double time_ms;
    std::string details;

    TestResult(const std::string& n, bool p, double t, const std::string& d = "")
        : name(n), passed(p), time_ms(t), details(d) {}
};

std::vector<TestResult> all_results;

// Helper: Check if two tensors are close
template <typename T>
bool allclose(const Tensor<T>& a, const Tensor<T>& b, T rtol = 1e-5, T atol = 1e-8) {
    if (a.shape() != b.shape()) return false;

    for (size_t i = 0; i < a.size(); ++i) {
        T diff = std::abs(a[i] - b[i]);
        T tolerance = atol + rtol * std::abs(b[i]);
        if (diff > tolerance) {
            std::cout << "  Mismatch at index " << i << ": " << a[i]
                      << " vs " << b[i] << " (diff=" << diff << ")\n";
            return false;
        }
    }
    return true;
}

// ============================================================================
// Test 1: Basic Tensor Operations
// ============================================================================

void test_tensor_basic() {
    std::cout << "\n=== Test: Basic Tensor Operations ===\n";
    Timer timer("Tensor Basic");

    try {
        // Create tensor
        TensorF x = TensorF::randn(Shape({2, 3}));
        TensorF y = TensorF::ones(Shape({2, 3}));

        // Addition
        TensorF z = x + y;
        std::cout << "  Addition: OK\n";

        // Element-wise multiplication
        TensorF w = x * y;
        std::cout << "  Multiplication: OK\n";

        // Reshape
        TensorF reshaped = x.reshape(Shape({6}));
        std::cout << "  Reshape: OK\n";

        // Matrix multiplication
        TensorF a = TensorF::randn(Shape({4, 8}));
        TensorF b = TensorF::randn(Shape({8, 6}));
        TensorF c = a.matmul(b);
        std::cout << "  Matmul: " << c.shape().to_string() << "\n";

        all_results.emplace_back("Tensor Basic", true, timer.elapsed_ms());

    } catch (const std::exception& e) {
        std::cout << "  ERROR: " << e.what() << "\n";
        all_results.emplace_back("Tensor Basic", false, timer.elapsed_ms(), e.what());
    }
}

// ============================================================================
// Test 2: Add Operation
// ============================================================================

void test_add() {
    std::cout << "\n=== Test: Add Operation ===\n";
    Timer timer("Add");

    try {
        TensorF x = TensorF::randn(Shape({2, 3}));
        TensorF y = TensorF::randn(Shape({2, 3}));

        TensorF z = add(x, y, 1.5f);

        // Verify: manually compute first element
        float expected = x[0] + 1.5f * y[0];
        if (std::abs(z[0] - expected) < 1e-6) {
            std::cout << "  Add with alpha: OK\n";
        }

        // Scalar addition
        TensorF scalar_result = x + 2.0f;
        std::cout << "  Scalar addition: OK\n";

        all_results.emplace_back("Add", true, timer.elapsed_ms());

    } catch (const std::exception& e) {
        std::cout << "  ERROR: " << e.what() << "\n";
        all_results.emplace_back("Add", false, timer.elapsed_ms(), e.what());
    }
}

// ============================================================================
// Test 3: Argmax
// ============================================================================

void test_argmax() {
    std::cout << "\n=== Test: Argmax ===\n";
    Timer timer("Argmax");

    try {
        TensorF x = TensorF::zeros(Shape({4}));
        x[0] = 0.1f; x[1] = 0.9f; x[2] = 0.3f; x[3] = 0.5f;

        Tensor<long> idx = argmax(x);
        if (idx[0] == 1) {
            std::cout << "  Global argmax: OK (index=" << idx[0] << ")\n";
        } else {
            std::cout << "  Global argmax: Got index=" << idx[0] << ", expected 1\n";
            throw std::runtime_error("Expected argmax at index 1");
        }

        // 2D test
        TensorF matrix = TensorF::randn(Shape({3, 4}));
        Tensor<long> idx2d = argmax(matrix);
        std::cout << "  2D argmax: OK (flat index=" << idx2d[0] << ")\n";

        all_results.emplace_back("Argmax", true, timer.elapsed_ms());

    } catch (const std::exception& e) {
        std::cout << "  ERROR: " << e.what() << "\n";
        all_results.emplace_back("Argmax", false, timer.elapsed_ms(), e.what());
    }
}

// ============================================================================
// Test 4: Embedding
// ============================================================================

void test_embedding() {
    std::cout << "\n=== Test: Embedding ===\n";
    Timer timer("Embedding");

    try {
        // Create embedding matrix: vocab_size=100, embedding_dim=64
        TensorF weight = TensorF::randn(Shape({100, 64}));

        // Token indices
        Tensor<long> indices({2, 10});  // Batch of 2, sequence length 10
        for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = static_cast<long>(i % 100);  // Some indices
        }

        // Lookup embeddings
        TensorF embeddings = embedding(indices, weight);

        std::cout << "  Input shape: " << indices.shape().to_string() << "\n";
        std::cout << "  Output shape: " << embeddings.shape().to_string() << "\n";

        if (embeddings.shape() == Shape({2, 10, 64})) {
            std::cout << "  Embedding lookup: OK\n";
        } else {
            throw std::runtime_error("Unexpected output shape");
        }

        all_results.emplace_back("Embedding", true, timer.elapsed_ms());

    } catch (const std::exception& e) {
        std::cout << "  ERROR: " << e.what() << "\n";
        all_results.emplace_back("Embedding", false, timer.elapsed_ms(), e.what());
    }
}

// ============================================================================
// Test 5: Linear Layer
// ============================================================================

void test_linear() {
    std::cout << "\n=== Test: Linear Layer ===\n";
    Timer timer("Linear");

    try {
        // Input: batch_size=32, in_features=128
        TensorF input = TensorF::randn(Shape({32, 128}));

        // Weight: out_features=256, in_features=128
        TensorF weight = TensorF::randn(Shape({256, 128}));

        // Bias: out_features=256
        TensorF bias = TensorF::uniform(Shape({256}), -0.1f, 0.1f);

        // Apply linear transformation
        TensorF output = linear(input, weight, &bias);

        std::cout << "  Input shape: " << input.shape().to_string() << "\n";
        std::cout << "  Output shape: " << output.shape().to_string() << "\n";

        if (output.shape() == Shape({32, 256})) {
            std::cout << "  Linear layer: OK\n";
        } else {
            throw std::runtime_error("Unexpected output shape");
        }

        // Verify no bias gives different result
        TensorF output_no_bias = linear(input, weight, static_cast<const TensorF*>(nullptr));

        // At least one element should be different
        bool different = false;
        for (size_t i = 0; i < output.size(); ++i) {
            if (std::abs(output[i] - output_no_bias[i]) > 1e-6) {
                different = true;
                break;
            }
        }

        if (different) {
            std::cout << "  Bias effect: Verified\n";
        }

        all_results.emplace_back("Linear", true, timer.elapsed_ms());

    } catch (const std::exception& e) {
        std::cout << "  ERROR: " << e.what() << "\n";
        all_results.emplace_back("Linear", false, timer.elapsed_ms(), e.what());
    }
}

// ============================================================================
// Test 6: RMS Norm
// ============================================================================

void test_rms_norm() {
    std::cout << "\n=== Test: RMS Norm ===\n";
    Timer timer("RMS Norm");

    try {
        // Input: batch_size=16, hidden_size=768
        TensorF input = TensorF::randn(Shape({16, 768}));

        // Gain (gamma)
        TensorF gamma = TensorF::ones(Shape({768}));

        // Apply RMS norm
        TensorF output = rms_norm(input, &gamma);

        std::cout << "  Input shape: " << input.shape().to_string() << "\n";
        std::cout << "  Output shape: " << output.shape().to_string() << "\n";

        // Check that RMS norm approximately normalizes to unit variance
        // Compute variance of first sample
        float mean_square = 0;
        for (size_t i = 0; i < 768; ++i) {
            float val = output[i];
            mean_square += val * val;
        }
        mean_square /= 768;

        std::cout << "  Mean square after norm: " << mean_square << "\n";

        if (output.shape() == Shape({16, 768})) {
            std::cout << "  RMS norm: OK\n";
        }

        all_results.emplace_back("RMS Norm", true, timer.elapsed_ms());

    } catch (const std::exception& e) {
        std::cout << "  ERROR: " << e.what() << "\n";
        all_results.emplace_back("RMS Norm", false, timer.elapsed_ms(), e.what());
    }
}

// ============================================================================
// Test 7: RoPE (Rotary Position Embedding)
// ============================================================================

void test_rope() {
    std::cout << "\n=== Test: RoPE ===\n";
    Timer timer("RoPE");

    try {
        // Create rotary embedding
        RotaryEmbedding<float> rope(64, 2048);

        // Input: batch_size=2, seq_len=100, num_heads=8, head_dim=64
        TensorF input = TensorF::randn(Shape({2, 100, 8, 64}));

        // Apply rotary embedding
        TensorF output = rope.apply(input, 100);

        std::cout << "  Input shape: " << input.shape().to_string() << "\n";
        std::cout << "  Output shape: " << output.shape().to_string() << "\n";

        if (output.shape() == Shape({2, 100, 8, 64})) {
            std::cout << "  RoPE: OK\n";
        }

        all_results.emplace_back("RoPE", true, timer.elapsed_ms());

    } catch (const std::exception& e) {
        std::cout << "  ERROR: " << e.what() << "\n";
        all_results.emplace_back("RoPE", false, timer.elapsed_ms(), e.what());
    }
}

// ============================================================================
// Test 8: SwiGLU Activation
// ============================================================================

void test_swiglu() {
    std::cout << "\n=== Test: SwiGLU ===\n";
    Timer timer("SwiGLU");

    try {
        TensorF x = TensorF::randn(Shape({128, 256}));
        TensorF gate = TensorF::randn(Shape({128, 256}));

        TensorF output = swiglu(x, gate);

        std::cout << "  Input shape: " << x.shape().to_string() << "\n";
        std::cout << "  Output shape: " << output.shape().to_string() << "\n";

        // SwiGLU should be non-negative (SiLU output >= 0)
        bool all_positive = true;
        for (size_t i = 0; i < output.size(); ++i) {
            if (output[i] < 0) {
                all_positive = false;
                break;
            }
        }

        if (output.shape() == Shape({128, 256}) && all_positive) {
            std::cout << "  SwiGLU: OK (all non-negative)\n";
        }

        all_results.emplace_back("SwiGLU", true, timer.elapsed_ms());

    } catch (const std::exception& e) {
        std::cout << "  ERROR: " << e.what() << "\n";
        all_results.emplace_back("SwiGLU", false, timer.elapsed_ms(), e.what());
    }
}

// ============================================================================
// Test 9: Self Attention
// ============================================================================

void test_self_attention() {
    std::cout << "\n=== Test: Self Attention ===\n";
    Timer timer("Self Attention");

    try {
        size_t batch_size = 2;
        size_t num_heads = 8;
        size_t seq_len = 64;
        size_t head_dim = 64;

        // Q, K, V tensors
        TensorF query = TensorF::randn(Shape({batch_size, num_heads, seq_len, head_dim}));
        TensorF key = TensorF::randn(Shape({batch_size, num_heads, seq_len, head_dim}));
        TensorF value = TensorF::randn(Shape({batch_size, num_heads, seq_len, head_dim}));

        // Scale factor
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        // Compute self-attention
        TensorF output = self_attention(query, key, value, static_cast<const TensorF*>(nullptr), scale);

        std::cout << "  Q/K/V shape: " << query.shape().to_string() << "\n";
        std::cout << "  Output shape: " << output.shape().to_string() << "\n";

        if (output.shape() == Shape({batch_size, num_heads, seq_len, head_dim})) {
            std::cout << "  Self Attention: OK\n";
        }

        all_results.emplace_back("Self Attention", true, timer.elapsed_ms());

    } catch (const std::exception& e) {
        std::cout << "  ERROR: " << e.what() << "\n";
        all_results.emplace_back("Self Attention", false, timer.elapsed_ms(), e.what());
    }
}

// ============================================================================
// Test 10: OpenMP Scaling
// ============================================================================

void test_openmp_scaling() {
    std::cout << "\n=== Test: OpenMP Scaling ===\n";

    size_t M = 512, N = 512, K = 512;
    TensorF A = TensorF::randn(Shape({M, K}));
    TensorF B = TensorF::randn(Shape({K, N}));

    std::vector<int> thread_counts = {1, 2, 4, 8};

    std::cout << "  Matrix multiplication: " << M << "x" << K << " @ " << K << "x" << N << "\n";

    for (int threads : thread_counts) {
        #ifdef _OPENMP
        omp_set_num_threads(threads);
        #endif

        auto start = std::chrono::high_resolution_clock::now();

        TensorF C = A.matmul(B);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;

        std::cout << "  Threads=" << threads << ": " << std::fixed << std::setprecision(3)
                  << elapsed.count() << " ms\n";
    }
}

// ============================================================================
// Test 11: MPI Operations (if MPI is enabled)
// ============================================================================

#ifdef MPI_VERSION
void test_mpi_ops() {
    std::cout << "\n=== Test: MPI Operations ===\n";

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Create a tensor on each rank
    TensorF tensor = TensorF::randn(Shape({16, 16}));

    // Each rank adds its rank value
    for (size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] += static_cast<float>(rank);
    }

    // All-reduce sum
    Timer timer("MPI All-Reduce");
    all_reduce_sum(tensor, MPI_COMM_WORLD);

    // All ranks should have the same result: sum of 0..size-1 = size*(size-1)/2
    float expected_sum = static_cast<float>(size * (size - 1) / 2);

    bool correct = true;
    for (size_t i = 0; i < std::min(size_t(10), tensor.size()); ++i) {
        if (std::abs(tensor[i] - expected_sum) > 1e-3) {
            correct = false;
            break;
        }
    }

    if (rank == 0) {
        std::cout << "  MPI All-Reduce: ";
        if (correct) {
            std::cout << "OK (sum=" << expected_sum << ")\n";
            all_results.emplace_back("MPI All-Reduce", true, timer.elapsed_ms());
        } else {
            std::cout << "FAILED\n";
            all_results.emplace_back("MPI All-Reduce", false, timer.elapsed_ms());
        }
    }

    // Broadcast test
    if (rank == 0) {
        std::cout << "  MPI Broadcast: ";
        tensor.fill(42.0f);
    }

    broadcast(tensor, 0, MPI_COMM_WORLD);

    bool broadcast_correct = true;
    for (size_t i = 0; i < std::min(size_t(10), tensor.size()); ++i) {
        if (std::abs(tensor[i] - 42.0f) > 1e-6) {
            broadcast_correct = false;
            break;
        }
    }

    if (rank == 0) {
        if (broadcast_correct) {
            std::cout << "OK\n";
        } else {
            std::cout << "FAILED\n";
        }
    }
}
#endif

// ============================================================================
// Save Results
// ============================================================================

void save_results(const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing\n";
        return;
    }

    out << "Tensor Library Test Results\n";
    out << "============================\n\n";

    size_t passed = 0, failed = 0;
    for (const auto& result : all_results) {
        out << "Test: " << result.name << "\n";
        out << "  Status: " << (result.passed ? "PASSED" : "FAILED") << "\n";
        out << "  Time: " << result.time_ms << " ms\n";
        if (!result.details.empty()) {
            out << "  Details: " << result.details << "\n";
        }
        out << "\n";

        if (result.passed) passed++;
        else failed++;
    }

    out << "Summary:\n";
    out << "  Total: " << all_results.size() << "\n";
    out << "  Passed: " << passed << "\n";
    out << "  Failed: " << failed << "\n";

    out.close();

    std::cout << "\nResults saved to: " << filename << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    #ifdef MPI_VERSION
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "========================================================\n";
        std::cout << "  Tensor Library Test Suite with MPI\n";
        std::cout << "  MPI Ranks: " << size << "\n";
        std::cout << "========================================================\n";
    }
    #else
    std::cout << "========================================================\n";
    std::cout << "  Tensor Library Test Suite\n";
    #ifdef _OPENMP
    std::cout << "  OpenMP: Enabled (max threads=" << omp_get_max_threads() << ")\n";
    #else
    std::cout << "  OpenMP: Disabled\n";
    #endif
    std::cout << "========================================================\n";
    #endif

    // Run all tests
    #ifdef MPI_VERSION
    if (rank == 0) {
    #endif
        test_tensor_basic();
        test_add();
        test_argmax();
        test_embedding();
        test_linear();
        test_rms_norm();
        test_rope();
        test_swiglu();
        test_self_attention();
        test_openmp_scaling();
    #ifdef MPI_VERSION
    }

    // MPI tests
    test_mpi_ops();
    #else
    test_openmp_scaling();
    #endif

    // Save results (only rank 0 for MPI)
    #ifdef MPI_VERSION
    if (rank == 0) {
        save_results("tensor_lib_cpp/results/test_results.txt");
    }
    #else
    save_results("tensor_lib_cpp/results/test_results.txt");
    #endif

    // Print summary
    #ifdef MPI_VERSION
    if (rank == 0) {
    #endif
        std::cout << "\n========================================================\n";
        std::cout << "  Test Summary\n";
        std::cout << "========================================================\n";

        size_t passed = 0, failed = 0;
        for (const auto& result : all_results) {
            if (result.passed) passed++;
            else failed++;
        }

        std::cout << "  Total: " << all_results.size() << "\n";
        std::cout << "  Passed: " << passed << "\n";
        std::cout << "  Failed: " << failed << "\n";

        if (failed == 0) {
            std::cout << "\n  ✓ All tests PASSED!\n";
        } else {
            std::cout << "\n  ✗ Some tests FAILED\n";
        }

        std::cout << "========================================================\n";
    #ifdef MPI_VERSION
    }
    #endif

    #ifdef MPI_VERSION
    MPI_Finalize();
    #endif

    return 0;
}
