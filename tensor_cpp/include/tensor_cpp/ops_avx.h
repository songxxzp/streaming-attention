/**
 * @file ops_avx.h
 * @brief AVX SIMD optimized deep learning operators
 *
 * Provides AVX2/AVX-512 accelerated implementations for key operations
 */

#ifndef TENSOR_CPP_OPS_AVX_H
#define TENSOR_CPP_OPS_AVX_H

#include "tensor_cpp/tensor.h"
#include <vector>

namespace tensor_cpp {
namespace ops {
namespace avx {

// ============================================================================
// AVX SIMD Accelerated Matrix Operations
// ============================================================================

/**
 * @brief AVX-accelerated matrix multiplication: C = A @ B^T
 *
 * Uses AVX2 intrinsics for vectorized computation
 * Falls back to scalar implementation if AVX is not available
 *
 * @param A Input matrix [M, K]
 * @param B Weight matrix [N, K] (transposed in memory)
 * @param M Rows of A
 * @param N Rows of B
 * @param K Columns of A / rows of B
 * @return Output matrix [M, N]
 */
std::vector<float> matmul_avx(
    const float* A,
    const float* B,
    int M,
    int N,
    int K
);

/**
 * @brief AVX-accelerated vector dot product
 *
 * @param x Vector 1 [size]
 * @param y Vector 2 [size]
 * @param size Vector size
 * @return Dot product
 */
float dot_avx(const float* x, const float* y, int size);

/**
 * @brief AVX-accelerated element-wise vector addition
 *
 * @param x Input vector [size]
 * @param y Input vector [size]
 * @param scale Scaling factor for y
 * @param size Vector size
 * @param result Output vector [size]
 */
void add_avx(const float* x, const float* y, float scale, int size, float* result);

/**
 * @brief AVX-accelerated RMSNorm
 *
 * @param input Input [batch_size, hidden_size]
 * @param weight Optional weight [hidden_size]
 * @param batch_size Number of vectors
 * @param hidden_size Vector dimension
 * @param eps Epsilon for numerical stability
 * @param result Output [batch_size, hidden_size]
 */
void rms_norm_avx(
    const float* input,
    const float* weight,
    int batch_size,
    int hidden_size,
    float eps,
    float* result
);

/**
 * @brief Check if AVX2 is supported at runtime
 */
bool is_avx2_supported();

/**
 * @brief Check if AVX-512 is supported at runtime
 */
bool is_avx512_supported();

} // namespace avx
} // namespace ops
} // namespace tensor_cpp

#endif // TENSOR_CPP_OPS_AVX_H
