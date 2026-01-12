/**
 * @file avx2_helpers.h
 * @brief Common AVX2 helper functions for Qwen3 implementation
 *
 * This header provides optimized AVX2 utility functions used across
 * multiple Qwen3 implementations (AVX, AVX2, MPI+AVX2, etc.)
 */

#ifndef TENSOR_CPP_AVX2_HELPERS_H
#define TENSOR_CPP_AVX2_HELPERS_H

#include <immintrin.h>
#include <cmath>

namespace tensor_cpp {
namespace avx2_helpers {

// ============================================================================
// AVX2 Horizontal Sum (Optimized)
// ============================================================================

/**
 * @brief Compute horizontal sum of AVX2 vector (faster than hadd)
 *
 * Uses shuffle-based approach instead of _mm256_hadd_ps for better performance.
 * Extracts 8 floats from __m256 and returns their sum.
 *
 * @param v AVX2 vector to sum
 * @return Sum of all 8 floats in the vector
 */
inline float hsum_avx2(__m256 v) {
    // Extract high and low 128-bit lanes
    __m128 hi_quad = _mm256_extractf128_ps(v, 1);
    __m128 lo_quad = _mm256_castps256_ps128(v);

    // Add corresponding elements from both lanes
    __m128 sum_quad = _mm_add_ps(lo_quad, hi_quad);

    // Shuffle to add upper and lower halves
    __m128 lo_dual = sum_quad;
    __m128 hi_dual = _mm_movehl_ps(sum_quad, sum_quad);
    __m128 sum_dual = _mm_add_ps(lo_dual, hi_dual);

    // Final shuffle to get the sum
    __m128 lo = sum_dual;
    __m128 hi = _mm_shuffle_ps(sum_dual, sum_dual, 1);
    __m128 sum_128 = _mm_add_ss(lo, hi);

    return _mm_cvtss_f32(sum_128);
}

// ============================================================================
// AVX2 Fast Sigmoid Approximation
// ============================================================================

/**
 * @brief Fast sigmoid approximation using AVX2
 *
 * Approximates sigmoid(x) = 1 / (1 + exp(-x))
 * Using the approximation: x / (1 + |x|)
 * This is much faster than computing exp() but less accurate.
 *
 * @param x AVX2 vector of input values
 * @return AVX2 vector of sigmoid outputs
 */
inline __m256 sigmoid_fast_avx2(__m256 x) {
    // Compute |x| using bitwise operations
    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    __m256 abs_x = _mm256_andnot_ps(sign_mask, x);

    // Compute x / (1 + |x|)
    __m256 ones = _mm256_set1_ps(1.0f);
    return _mm256_div_ps(x, _mm256_add_ps(abs_x, ones));
}

// ============================================================================
// AVX2 Vector Operations
// ============================================================================

/**
 * @brief Vectorized RMS normalization
 *
 * @param input Input data
 * @param weight Normalization weights (can be nullptr)
 * @param size Number of elements
 * @param eps Epsilon for numerical stability
 * @param output Output buffer (must be pre-allocated)
 */
inline void rms_norm_avx2(
    const float* input,
    const float* weight,
    size_t size,
    float eps,
    float* output
) {
    // Compute sum of squares using AVX2
    __m256 sum_sq_vec = _mm256_setzero_ps();
    size_t i = 0;

    // Process 8 floats at a time
    for (; i + 8 <= size; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 x_sq = _mm256_mul_ps(x, x);
        sum_sq_vec = _mm256_add_ps(sum_sq_vec, x_sq);
    }

    // Horizontal sum to get total
    float sum_sq = hsum_avx2(sum_sq_vec);

    // Handle remaining elements
    for (; i < size; ++i) {
        sum_sq += input[i] * input[i];
    }

    // Compute RMS
    float rms = std::sqrt(sum_sq / size + eps);
    float inv_rms = 1.0f / rms;

    // Normalize and apply weight
    __m256 inv_rms_vec = _mm256_set1_ps(inv_rms);
    i = 0;

    for (; i + 8 <= size; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 w = weight ? _mm256_loadu_ps(&weight[i]) : _mm256_set1_ps(1.0f);
        __m256 result = _mm256_mul_ps(_mm256_mul_ps(x, inv_rms_vec), w);
        _mm256_storeu_ps(&output[i], result);
    }

    // Handle remaining elements
    for (; i < size; ++i) {
        output[i] = (input[i] * inv_rms) * (weight ? weight[i] : 1.0f);
    }
}

// ============================================================================
// AVX2 RoPE (Rotary Position Embedding)
// ============================================================================

/**
 * @brief Apply rotary position embedding to a single vector
 *
 * @param x Input vector (head_dim elements)
 * @param cos Cosine values for rotation
 * @param sin Sine values for rotation
 * @param head_dim Dimension of attention head
 * @param output Output buffer
 */
inline void apply_rope_single_avx2(
    const float* x,
    const float* cos,
    const float* sin,
    size_t head_dim,
    float* output
) {
    for (size_t i = 0; i + 8 <= head_dim; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(&x[i]);
        __m256 cos_vec = _mm256_loadu_ps(&cos[i]);
        __m256 sin_vec = _mm256_loadu_ps(&sin[i]);

        // For RoPE, we need to pair up elements: (x0, x1), (x2, x3), etc.
        // Apply rotation: x' = x*cos + rotate(x)*sin
        // where rotate swaps pairs: (x0, x1) -> (-x1, x0)

        // Load and swap pairs for the sin term
        __m256 x_swapped = _mm256_permute_ps(x_vec, 0xB1);  // Swap pairs
        __m256 sign_mask = _mm256_set1_ps(-0.0f);
        // Apply sign to make it: (-x1, x0, -x3, x2, ...)
        __m256 x_neg_swapped = _mm256_xor_ps(x_swapped, _mm256_and_ps(sign_mask, _mm256_set1_ps(1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f)));

        // x' = x*cos + x_swapped*sin
        __m256 result = _mm256_fmadd_ps(x_vec, cos_vec, _mm256_mul_ps(x_neg_swapped, sin_vec));

        _mm256_storeu_ps(&output[i], result);
    }

    // Handle remaining elements
    for (size_t i = (head_dim / 8) * 8; i < head_dim; i += 2) {
        float x0 = x[i];
        float x1 = x[i + 1];
        output[i] = x0 * cos[i] - x1 * sin[i];
        output[i + 1] = x1 * cos[i] + x0 * sin[i];
    }
}

} // namespace avx2_helpers
} // namespace tensor_cpp

#endif // TENSOR_CPP_AVX2_HELPERS_H
