/**
 * @file test_mpi_ops.cpp
 * @brief Comprehensive test for MPI+OpenMP parallelized operators
 */

#include <mpi.h>
#include "tensor_cpp/ops_mpi.h"
#include "tensor_cpp/ops.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>

using namespace tensor_cpp;
using namespace tensor_cpp::ops;

// Helper function to compute L2 error
float compute_l2_error(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        return std::numeric_limits<float>::infinity();
    }

    float sum_sq = 0.0f;
    float sum_ref = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum_sq += diff * diff;
        sum_ref += b[i] * b[i];
    }

    return std::sqrt(sum_sq / (sum_ref + 1e-10f));
}

// Test 1: Matrix Multiplication
void test_matmul_mpi_omp(MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        std::cout << "\n========================================\n";
        std::cout << "Test 1: Matrix Multiplication (MPI+OpenMP)\n";
        std::cout << "========================================\n";
    }

    int M = 128, N = 64, K = 32;

    // Initialize data
    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(N * K, 2.0f);
    std::vector<float> C_mpi(M * N, 0.0f);  // Initialize to zero

    // Compute with MPI+OpenMP
    auto start = std::chrono::high_resolution_clock::now();
    mpi::matmul_mpi_omp(A.data(), B.data(), C_mpi.data(), M, N, K, comm);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;

    // Print some debug info
    if (rank == 0) {
        std::cout << "  First few C_mpi values: ";
        for (int i = 0; i < std::min(10, static_cast<int>(C_mpi.size())); ++i) {
            std::cout << C_mpi[i] << " ";
        }
        std::cout << "\n";
    }

    if (rank == 0) {
        // Compute reference
        std::vector<float> C_ref(M * N);
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[j * K + k];
                }
                C_ref[i * N + j] = sum;
            }
        }

        // Print reference values
        std::cout << "  First few C_ref values: ";
        for (int i = 0; i < std::min(10, static_cast<int>(C_ref.size())); ++i) {
            std::cout << C_ref[i] << " ";
        }
        std::cout << "\n";

        // Verify correctness
        float max_error = 0.0f;
        size_t max_error_idx = 0;
        for (size_t i = 0; i < C_mpi.size(); ++i) {
            float err = std::abs(C_mpi[i] - C_ref[i]);
            if (err > max_error) {
                max_error = err;
                max_error_idx = i;
            }
        }

        std::cout << "  Max error: " << max_error << " at index " << max_error_idx;
        std::cout << " (C_mpi=" << C_mpi[max_error_idx] << ", C_ref=" << C_ref[max_error_idx] << ")\n";

        std::cout << "  Matrix size: " << M << "x" << N << "x" << K << "\n";
        std::cout << "  MPI processes: " << size << "\n";
        std::cout << "  Time: " << diff.count() << " ms\n";
        std::cout << "  Max error: " << max_error << "\n";
        std::cout << "  Status: " << (max_error < 1e-4 ? "PASSED" : "FAILED") << "\n";
    }
}

// Test 2: Element-wise Add
void test_add_mpi_omp(MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        std::cout << "\n========================================\n";
        std::cout << "Test 2: Element-wise Add (MPI+OpenMP)\n";
        std::cout << "========================================\n";
    }

    // Create test tensors
    std::vector<float> data1(1000, 1.0f);
    std::vector<float> data2(1000, 2.0f);
    Tensor input1(std::vector<float>(data1), Shape({10, 10, 10}));
    Tensor input2(std::vector<float>(data2), Shape({10, 10, 10}));

    // Compute with MPI
    Tensor result = mpi::add_mpi_omp(input1, input2, 1.0f, comm);

    // Compute reference
    Tensor ref = add(input1, input2, 1.0f);

    // Check correctness
    float error = compute_l2_error(result, ref);

    if (rank == 0) {
        std::cout << "  Tensor shape: [10, 10, 10]\n";
        std::cout << "  MPI processes: " << size << "\n";
        std::cout << "  L2 error: " << error << "\n";
        std::cout << "  Status: " << (error < 1e-6 ? "PASSED" : "FAILED") << "\n";
    }
}

// Test 3: RMSNorm
void test_rms_norm_mpi_omp(MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        std::cout << "\n========================================\n";
        std::cout << "Test 3: RMSNorm (MPI+OpenMP)\n";
        std::cout << "========================================\n";
    }

    // Create test tensor
    std::vector<float> data(1024 * 768);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i) / 1000.0f;
    }

    Tensor input(std::move(data), Shape({2, 4, 128, 768}));

    // Compute with MPI
    Tensor result = mpi::rms_norm_mpi_omp(input, nullptr, 1e-6f, comm);

    // Compute reference
    Tensor ref = rms_norm(input, nullptr, 1e-6f);

    // Check correctness
    float error = compute_l2_error(result, ref);

    if (rank == 0) {
        std::cout << "  Tensor shape: [2, 4, 128, 768]\n";
        std::cout << "  MPI processes: " << size << "\n";
        std::cout << "  L2 error: " << error << "\n";
        std::cout << "  Status: " << (error < 1e-6 ? "PASSED" : "FAILED") << "\n";
    }
}

// Test 4: RoPE
void test_rope_mpi_omp(MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        std::cout << "\n========================================\n";
        std::cout << "Test 4: RoPE (MPI+OpenMP)\n";
        std::cout << "========================================\n";
    }

    // Create test tensors
    size_t batch = 2, heads = 4, seq_len = 16, head_dim = 128;
    std::vector<float> query_data(batch * heads * seq_len * head_dim, 1.0f);

    // Create cos/sin
    std::vector<float> cos_data(seq_len * head_dim / 2);
    std::vector<float> sin_data(seq_len * head_dim / 2);
    for (size_t i = 0; i < seq_len * head_dim / 2; ++i) {
        cos_data[i] = std::cos(static_cast<float>(i) / 100.0f);
        sin_data[i] = std::sin(static_cast<float>(i) / 100.0f);
    }

    Tensor query(std::move(query_data), Shape({static_cast<long>(batch), static_cast<long>(heads),
                                               static_cast<long>(seq_len), static_cast<long>(head_dim)}));
    Tensor cos(std::move(cos_data), Shape({static_cast<long>(seq_len), static_cast<long>(head_dim / 2)}));
    Tensor sin(std::move(sin_data), Shape({static_cast<long>(seq_len), static_cast<long>(head_dim / 2)}));

    // Compute with MPI
    Tensor result = mpi::rope_mpi_omp(query, cos, sin, comm);

    // Check shape
    bool shape_ok = result.shape() == query.shape();

    if (rank == 0) {
        std::cout << "  Query shape: [2, 4, 16, 128]\n";
        std::cout << "  MPI processes: " << size << "\n";
        std::cout << "  Output shape correct: " << (shape_ok ? "YES" : "NO") << "\n";
        std::cout << "  Status: " << (shape_ok ? "PASSED" : "FAILED") << "\n";
    }
}

// Test 5: SwiGLU
void test_swiglu_mpi_omp(MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        std::cout << "\n========================================\n";
        std::cout << "Test 5: SwiGLU (MPI+OpenMP)\n";
        std::cout << "========================================\n";
    }

    // Create test tensors
    std::vector<float> x_data(1000);
    std::vector<float> gate_data(1000);
    for (size_t i = 0; i < 1000; ++i) {
        x_data[i] = static_cast<float>(i);
        gate_data[i] = static_cast<float>(i) * 0.1f;
    }

    Tensor x(std::move(x_data), Shape({10, 10, 10}));
    Tensor gate(std::move(gate_data), Shape({10, 10, 10}));

    // Compute with MPI
    Tensor result = mpi::swiglu_mpi_omp(x, gate, comm);

    // Compute reference
    Tensor ref = swiglu(x, gate);

    // Check correctness
    float error = compute_l2_error(result, ref);

    if (rank == 0) {
        std::cout << "  Tensor shape: [10, 10, 10]\n";
        std::cout << "  MPI processes: " << size << "\n";
        std::cout << "  L2 error: " << error << "\n";
        std::cout << "  Status: " << (error < 1e-6 ? "PASSED" : "FAILED") << "\n";
    }
}

// Test 6: Self-Attention
void test_self_attention_mpi_omp(MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        std::cout << "\n========================================\n";
        std::cout << "Test 6: Self-Attention (MPI+OpenMP)\n";
        std::cout << "========================================\n";
    }

    // Create test tensors
    size_t batch = 2, num_heads = 16, seq_len = 32, head_dim = 128;
    size_t num_kv_heads = 8;

    std::vector<float> q_data(batch * num_heads * seq_len * head_dim);
    std::vector<float> k_data(batch * num_kv_heads * seq_len * head_dim);
    std::vector<float> v_data(batch * num_kv_heads * seq_len * head_dim);

    for (size_t i = 0; i < q_data.size(); ++i) q_data[i] = static_cast<float>(i) / 1000.0f;
    for (size_t i = 0; i < k_data.size(); ++i) k_data[i] = static_cast<float>(i) / 2000.0f;
    for (size_t i = 0; i < v_data.size(); ++i) v_data[i] = static_cast<float>(i) / 3000.0f;

    Tensor query(std::move(q_data), Shape({static_cast<long>(batch), static_cast<long>(num_heads),
                                           static_cast<long>(seq_len), static_cast<long>(head_dim)}));
    Tensor key(std::move(k_data), Shape({static_cast<long>(batch), static_cast<long>(num_kv_heads),
                                          static_cast<long>(seq_len), static_cast<long>(head_dim)}));
    Tensor value(std::move(v_data), Shape({static_cast<long>(batch), static_cast<long>(num_kv_heads),
                                            static_cast<long>(seq_len), static_cast<long>(head_dim)}));

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Compute with MPI
    Tensor result = mpi::self_attention_mpi_omp(query, key, value, nullptr, scale,
                                                 num_heads, num_kv_heads, comm);

    // Check shape
    bool shape_ok = result.shape() == query.shape();

    if (rank == 0) {
        std::cout << "  Query shape: [2, 16, 32, 128]\n";
        std::cout << "  Key/Value shape: [2, 8, 32, 128]\n";
        std::cout << "  MPI processes: " << size << "\n";
        std::cout << "  Output shape correct: " << (shape_ok ? "YES" : "NO") << "\n";
        std::cout << "  Status: " << (shape_ok ? "PASSED" : "FAILED") << "\n";
    }
}

// Test 6b: Self-Attention Streaming
void test_self_attention_mpi_streaming_omp(MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        std::cout << "\n========================================\n";
        std::cout << "Test 6b: Self-Attention STREAMING (MPI+OpenMP)\n";
        std::cout << "========================================\n";
    }

    // Create test tensors
    size_t batch = 2, num_heads = 16, seq_len = 32, head_dim = 128;
    size_t num_kv_heads = 8;

    std::vector<float> q_data(batch * num_heads * seq_len * head_dim);
    std::vector<float> k_data(batch * num_kv_heads * seq_len * head_dim);
    std::vector<float> v_data(batch * num_kv_heads * seq_len * head_dim);

    for (size_t i = 0; i < q_data.size(); ++i) q_data[i] = static_cast<float>(i) / 1000.0f;
    for (size_t i = 0; i < k_data.size(); ++i) k_data[i] = static_cast<float>(i) / 2000.0f;
    for (size_t i = 0; i < v_data.size(); ++i) v_data[i] = static_cast<float>(i) / 3000.0f;

    Tensor query(std::move(q_data), Shape({static_cast<long>(batch), static_cast<long>(num_heads),
                                           static_cast<long>(seq_len), static_cast<long>(head_dim)}));
    Tensor key(std::move(k_data), Shape({static_cast<long>(batch), static_cast<long>(num_kv_heads),
                                          static_cast<long>(seq_len), static_cast<long>(head_dim)}));
    Tensor value(std::move(v_data), Shape({static_cast<long>(batch), static_cast<long>(num_kv_heads),
                                            static_cast<long>(seq_len), static_cast<long>(head_dim)}));

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Compute with MPI streaming
    Tensor result = mpi::self_attention_mpi_streaming_omp(query, key, value, nullptr, scale,
                                                           num_heads, num_kv_heads, comm);

    // Check shape
    bool shape_ok = result.shape() == query.shape();

    if (rank == 0) {
        std::cout << "  Query shape: [2, 16, 32, 128]\n";
        std::cout << "  Key/Value shape: [2, 8, 32, 128]\n";
        std::cout << "  MPI processes: " << size << "\n";
        std::cout << "  Output shape correct: " << (shape_ok ? "YES" : "NO") << "\n";
        std::cout << "  Status: " << (shape_ok ? "PASSED" : "FAILED") << "\n";
    }
}

// Test 7: Linear Layer
void test_linear_mpi_omp(MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        std::cout << "\n========================================\n";
        std::cout << "Test 7: Linear Layer (MPI+OpenMP)\n";
        std::cout << "========================================\n";
    }

    // Create test tensors
    size_t seq_len = 128, in_features = 256, out_features = 512;

    std::vector<float> input_data(seq_len * in_features);
    std::vector<float> weight_data(out_features * in_features);

    for (size_t i = 0; i < input_data.size(); ++i) input_data[i] = static_cast<float>(i) / 1000.0f;
    for (size_t i = 0; i < weight_data.size(); ++i) weight_data[i] = static_cast<float>(i) / 10000.0f;

    Tensor input(std::move(input_data), Shape({static_cast<long>(seq_len), static_cast<long>(in_features)}));
    Tensor weight(std::move(weight_data), Shape({static_cast<long>(out_features), static_cast<long>(in_features)}));

    // Compute with MPI
    Tensor result = mpi::linear_mpi_omp(input, weight, nullptr, comm);

    // Check shape
    bool shape_ok = result.shape()[0] == static_cast<long>(seq_len) &&
                    result.shape()[1] == static_cast<long>(out_features);

    if (rank == 0) {
        std::cout << "  Input shape: [128, 256]\n";
        std::cout << "  Weight shape: [512, 256]\n";
        std::cout << "  MPI processes: " << size << "\n";
        std::cout << "  Output shape: [" << result.shape()[0] << ", " << result.shape()[1] << "]\n";
        std::cout << "  Output shape correct: " << (shape_ok ? "YES" : "NO") << "\n";
        std::cout << "  Status: " << (shape_ok ? "PASSED" : "FAILED") << "\n";
    }
}

// Main
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "============================================================\n";
        std::cout << "     MPI+OpenMP Operators Test Suite\n";
        std::cout << "============================================================\n";
        std::cout << "MPI processes: " << size << "\n";

        #ifdef _OPENMP
        std::cout << "OpenMP: Enabled (max threads = " << omp_get_max_threads() << ")\n";
        #else
        std::cout << "OpenMP: Disabled\n";
        #endif
        std::cout << "============================================================\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Run tests
    test_matmul_mpi_omp(MPI_COMM_WORLD);
    test_add_mpi_omp(MPI_COMM_WORLD);
    test_rms_norm_mpi_omp(MPI_COMM_WORLD);
    test_rope_mpi_omp(MPI_COMM_WORLD);
    test_swiglu_mpi_omp(MPI_COMM_WORLD);
    test_self_attention_mpi_omp(MPI_COMM_WORLD);
    test_self_attention_mpi_streaming_omp(MPI_COMM_WORLD);
    test_linear_mpi_omp(MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n============================================================\n";
        std::cout << "All tests completed!\n";
        std::cout << "============================================================\n";
    }

    MPI_Finalize();
    return 0;
}
