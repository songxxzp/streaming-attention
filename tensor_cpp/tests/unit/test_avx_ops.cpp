/**
 * @file test_avx_ops.cpp
 * @brief Test and benchmark AVX SIMD optimized operators
 */

#include "tensor_cpp/ops_avx.h"
#include "tensor_cpp/ops.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

using namespace tensor_cpp;
using namespace tensor_cpp::ops;

// Helper to compute max error
float compute_max_error(const std::vector<float>& a, const std::vector<float>& b) {
    float max_err = 0.0f;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        max_err = std::max(max_err, std::abs(a[i] - b[i]));
    }
    return max_err;
}

// Test 1: AVX Matrix Multiplication
void test_matmul_avx() {
    std::cout << "\n========================================\n";
    std::cout << "Test 1: AVX Matrix Multiplication\n";
    std::cout << "========================================\n";

    int M = 256, N = 256, K = 256;

    // Initialize data with smaller values to reduce floating point errors
    std::vector<float> A(M * K);
    std::vector<float> B(N * K);
    for (size_t i = 0; i < A.size(); ++i) A[i] = static_cast<float>(i % 100) / 100.0f;
    for (size_t i = 0; i < B.size(); ++i) B[i] = static_cast<float>(i % 100) / 100.0f;

    // Check AVX support
    std::cout << "  AVX2 supported: " << (avx::is_avx2_supported() ? "YES" : "NO") << "\n";
    std::cout << "  AVX-512 supported: " << (avx::is_avx512_supported() ? "YES" : "NO") << "\n";

    // Scalar version
    std::vector<float> C_scalar(M * N);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[j * K + k];
            }
            C_scalar[i * N + j] = sum;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> scalar_time = end - start;

    // AVX version
    auto C_avx = avx::matmul_avx(A.data(), B.data(), M, N, K);
    start = std::chrono::high_resolution_clock::now();
    C_avx = avx::matmul_avx(A.data(), B.data(), M, N, K);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> avx_time = end - start;

    // Verify correctness with relative error
    float max_rel_error = 0.0f;
    for (size_t i = 0; i < C_scalar.size(); ++i) {
        float ref_val = std::abs(C_scalar[i]);
        float abs_err = std::abs(C_scalar[i] - C_avx[i]);
        float rel_err = ref_val > 1e-6f ? abs_err / ref_val : abs_err;
        max_rel_error = std::max(max_rel_error, rel_err);
    }

    std::cout << "  Matrix size: " << M << "x" << N << "x" << K << "\n";
    std::cout << "  Scalar time: " << scalar_time.count() << " ms\n";
    std::cout << "  AVX time: " << avx_time.count() << " ms\n";
    std::cout << "  Speedup: " << scalar_time.count() / avx_time.count() << "x\n";
    std::cout << "  Max relative error: " << max_rel_error << "\n";
    std::cout << "  Status: " << (max_rel_error < 1e-3 ? "PASSED" : "FAILED") << "\n";
}

// Test 2: AVX Dot Product
void test_dot_avx() {
    std::cout << "\n========================================\n";
    std::cout << "Test 2: AVX Dot Product\n";
    std::cout << "========================================\n";

    int size = 10000;

    std::vector<float> x(size), y(size);
    for (int i = 0; i < size; ++i) {
        x[i] = static_cast<float>(i % 100) / 100.0f;
        y[i] = static_cast<float>(i % 100) / 200.0f;
    }

    // Scalar version
    float dot_scalar = 0.0f;
    for (int i = 0; i < size; ++i) {
        dot_scalar += x[i] * y[i];
    }

    // AVX version
    float dot_avx = avx::dot_avx(x.data(), y.data(), size);

    float abs_error = std::abs(dot_scalar - dot_avx);
    float rel_error = std::abs(dot_scalar) > 1e-6f ? abs_error / std::abs(dot_scalar) : abs_error;

    std::cout << "  Vector size: " << size << "\n";
    std::cout << "  Scalar result: " << dot_scalar << "\n";
    std::cout << "  AVX result: " << dot_avx << "\n";
    std::cout << "  Absolute error: " << abs_error << "\n";
    std::cout << "  Relative error: " << rel_error << "\n";
    std::cout << "  Status: " << (rel_error < 1e-4 ? "PASSED" : "FAILED") << "\n";
}

// Test 3: AVX Element-wise Addition
void test_add_avx() {
    std::cout << "\n========================================\n";
    std::cout << "Test 3: AVX Element-wise Addition\n";
    std::cout << "========================================\n";

    int size = 10000;
    float alpha = 2.5f;

    std::vector<float> x(size), y(size);
    for (int i = 0; i < size; ++i) {
        x[i] = static_cast<float>(i);
        y[i] = static_cast<float>(i) * 0.1f;
    }

    // Scalar version
    std::vector<float> result_scalar(size);
    for (int i = 0; i < size; ++i) {
        result_scalar[i] = x[i] + alpha * y[i];
    }

    // AVX version
    std::vector<float> result_avx(size);
    avx::add_avx(x.data(), y.data(), alpha, size, result_avx.data());

    float max_error = compute_max_error(result_scalar, result_avx);

    std::cout << "  Vector size: " << size << "\n";
    std::cout << "  Alpha: " << alpha << "\n";
    std::cout << "  Max error: " << max_error << "\n";
    std::cout << "  Status: " << (max_error < 1e-6 ? "PASSED" : "FAILED") << "\n";
}

// Test 4: AVX RMSNorm
void test_rms_norm_avx() {
    std::cout << "\n========================================\n";
    std::cout << "Test 4: AVX RMSNorm\n";
    std::cout << "========================================\n";

    int batch_size = 16;
    int hidden_size = 768;
    float eps = 1e-6f;

    std::vector<float> input(batch_size * hidden_size);
    std::vector<float> weight(hidden_size);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i) / 1000.0f;
    }
    for (int i = 0; i < hidden_size; ++i) {
        weight[i] = static_cast<float>(i) / 100.0f;
    }

    // Scalar version (using ops::rms_norm)
    Tensor input_tensor(std::vector<float>(input), Shape({static_cast<long>(batch_size), static_cast<long>(hidden_size)}));
    Tensor weight_tensor(std::vector<float>(weight), Shape({static_cast<long>(hidden_size)}));
    Tensor result_tensor = rms_norm(input_tensor, &weight_tensor, eps);

    // AVX version
    std::vector<float> result_avx(batch_size * hidden_size);
    avx::rms_norm_avx(input.data(), weight.data(), batch_size, hidden_size, eps, result_avx.data());

    // Compute error
    float max_error = 0.0f;
    for (size_t i = 0; i < result_tensor.size(); ++i) {
        max_error = std::max(max_error, std::abs(result_tensor[i] - result_avx[i]));
    }

    std::cout << "  Batch size: " << batch_size << "\n";
    std::cout << "  Hidden size: " << hidden_size << "\n";
    std::cout << "  Max error: " << max_error << "\n";
    std::cout << "  Status: " << (max_error < 1e-5 ? "PASSED" : "FAILED") << "\n";
}

// Benchmark: Compare performance on large matrix multiplication
void benchmark_large_matmul() {
    std::cout << "\n========================================\n";
    std::cout << "Benchmark: Large Matrix Multiplication\n";
    std::cout << "========================================\n";

    int M = 1024, N = 1024, K = 1024;
    int iters = 5;

    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(N * K, 2.0f);

    // Warmup
    avx::matmul_avx(A.data(), B.data(), M, N, K);

    // Scalar benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iters; ++iter) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[j * K + k];
                }
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> scalar_time = end - start;

    // AVX benchmark
    start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iters; ++iter) {
        auto C = avx::matmul_avx(A.data(), B.data(), M, N, K);
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> avx_time = end - start;

    std::cout << "  Matrix size: " << M << "x" << N << "x" << K << "\n";
    std::cout << "  Iterations: " << iters << "\n";
    std::cout << "  Scalar time: " << scalar_time.count() << " ms\n";
    std::cout << "  AVX time: " << avx_time.count() << " ms\n";
    std::cout << "  Speedup: " << std::fixed << std::setprecision(2)
              << scalar_time.count() / avx_time.count() << "x\n";
}

// Main
int main() {
    std::cout << "============================================================\n";
    std::cout << "     AVX SIMD Optimization Test Suite\n";
    std::cout << "============================================================\n";

    // Check CPU features
    std::cout << "\nCPU Features:\n";
    std::cout << "  AVX2: " << (avx::is_avx2_supported() ? "Supported" : "Not supported") << "\n";
    std::cout << "  AVX-512: " << (avx::is_avx512_supported() ? "Supported" : "Not supported") << "\n";

    // Run tests
    test_matmul_avx();
    test_dot_avx();
    test_add_avx();
    test_rms_norm_avx();

    // Run benchmark
    benchmark_large_matmul();

    std::cout << "\n============================================================\n";
    std::cout << "All tests completed!\n";
    std::cout << "============================================================\n";

    return 0;
}
