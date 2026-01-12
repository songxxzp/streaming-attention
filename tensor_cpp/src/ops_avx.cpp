/**
 * @file ops_avx.cpp
 * @brief Implementation of AVX SIMD optimized operators
 */

#include "tensor_cpp/ops_avx.h"
#include <cmath>
#include <algorithm>

// AVX intrinsics
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #ifdef __AVX2__
    #include <immintrin.h>
    #define HAS_AVX2
    #endif

    #ifdef __AVX512F__
    #include <immintrin.h>
    #define HAS_AVX512
    #endif
#endif

namespace tensor_cpp {
namespace ops {
namespace avx {

// ============================================================================
// CPU Feature Detection
// ============================================================================

bool is_avx2_supported() {
#ifdef HAS_AVX2
    return true;
#else
    return false;
#endif
}

bool is_avx512_supported() {
#ifdef HAS_AVX512
    return true;
#else
    return false;
#endif
}

// ============================================================================
// AVX Matrix Multiplication
// ============================================================================

std::vector<float> matmul_avx(
    const float* A,
    const float* B,
    int M,
    int N,
    int K
) {
    std::vector<float> C(M * N, 0.0f);

#ifdef HAS_AVX512
    // AVX-512 implementation: 16 floats per vector
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            __m512 sum = _mm512_setzero_ps();

            int k = 0;
            // Process 16 elements at a time
            for (; k + 15 < K; k += 16) {
                __m512 a = _mm512_loadu_ps(A + i * K + k);
                __m512 b = _mm512_loadu_ps(B + j * K + k);
                sum = _mm512_fmadd_ps(a, b, sum);  // sum += a * b
            }

            // Horizontal sum using _mm512_reduce_add_ps
            float result = _mm512_reduce_add_ps(sum);

            // Handle remaining elements
            for (; k < K; ++k) {
                result += A[i * K + k] * B[j * K + k];
            }

            C[i * N + j] = result;
        }
    }
#elif defined(HAS_AVX2)
    // AVX2 implementation: 8 floats per vector
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            __m256 sum0 = _mm256_setzero_ps();
            __m256 sum1 = _mm256_setzero_ps();

            int k = 0;
            // Process 16 elements at a time (2 AVX vectors)
            for (; k + 15 < K; k += 16) {
                __m256 a0 = _mm256_loadu_ps(A + i * K + k);
                __m256 b0 = _mm256_loadu_ps(B + j * K + k);
                sum0 = _mm256_fmadd_ps(a0, b0, sum0);

                __m256 a1 = _mm256_loadu_ps(A + i * K + k + 8);
                __m256 b1 = _mm256_loadu_ps(B + j * K + k + 8);
                sum1 = _mm256_fmadd_ps(a1, b1, sum1);
            }

            // Process 8 elements at a time
            for (; k + 7 < K; k += 8) {
                __m256 a = _mm256_loadu_ps(A + i * K + k);
                __m256 b = _mm256_loadu_ps(B + j * K + k);
                sum0 = _mm256_fmadd_ps(a, b, sum0);
            }

            // Horizontal sum - use more accurate method
            // Shuffle to add pairs
            sum0 = _mm256_add_ps(sum0, sum1);
            sum0 = _mm256_hadd_ps(sum0, sum0);
            sum0 = _mm256_hadd_ps(sum0, sum0);

            // Extract the result
            float temp[8];
            _mm256_storeu_ps(temp, sum0);
            float result = temp[0] + temp[4];

            // Handle remaining elements
            for (; k < K; ++k) {
                result += A[i * K + k] * B[j * K + k];
            }

            C[i * N + j] = result;
        }
    }
#else
    // Scalar fallback
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = sum;
        }
    }
#endif

    return C;
}

// ============================================================================
// AVX Dot Product
// ============================================================================

float dot_avx(const float* x, const float* y, int size) {
    float result = 0.0f;

#ifdef HAS_AVX512
    __m512 sum = _mm512_setzero_ps();
    int i = 0;

    // Process 16 elements at a time
    for (; i + 15 < size; i += 16) {
        __m512 vx = _mm512_loadu_ps(x + i);
        __m512 vy = _mm512_loadu_ps(y + i);
        sum = _mm512_fmadd_ps(vx, vy, sum);
    }

    result = _mm512_reduce_add_ps(sum);

#elif defined(HAS_AVX2)
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    int i = 0;

    // Process 16 elements at a time
    for (; i + 15 < size; i += 16) {
        __m256 vx0 = _mm256_loadu_ps(x + i);
        __m256 vy0 = _mm256_loadu_ps(y + i);
        sum0 = _mm256_fmadd_ps(vx0, vy0, sum0);

        __m256 vx1 = _mm256_loadu_ps(x + i + 8);
        __m256 vy1 = _mm256_loadu_ps(y + i + 8);
        sum1 = _mm256_fmadd_ps(vx1, vy1, sum1);
    }

    // Process 8 elements at a time
    for (; i + 7 < size; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        sum0 = _mm256_fmadd_ps(vx, vy, sum0);
    }

    // Horizontal sum - use more accurate method
    sum0 = _mm256_add_ps(sum0, sum1);
    sum0 = _mm256_hadd_ps(sum0, sum0);
    sum0 = _mm256_hadd_ps(sum0, sum0);

    // Extract the result
    float temp[8];
    _mm256_storeu_ps(temp, sum0);
    result = temp[0] + temp[4];
#else
    int i = 0;  // Declare i for non-AVX2 path
#endif

    // Handle remaining elements
    for (; i < size; ++i) {
        result += x[i] * y[i];
    }

    return result;
}

// ============================================================================
// AVX Element-wise Addition
// ============================================================================

void add_avx(const float* x, const float* y, float scale, int size, float* result) {
    int i = 0;

#ifdef HAS_AVX512
    // Process 16 elements at a time
    __m512 v_scale = _mm512_set1_ps(scale);
    for (; i + 15 < size; i += 16) {
        __m512 vx = _mm512_loadu_ps(x + i);
        __m512 vy = _mm512_loadu_ps(y + i);
        __m512 v_scaled_y = _mm512_mul_ps(vy, v_scale);
        __m512 vr = _mm512_add_ps(vx, v_scaled_y);
        _mm512_storeu_ps(result + i, vr);
    }
#elif defined(HAS_AVX2)
    // Process 8 elements at a time
    __m256 v_scale = _mm256_set1_ps(scale);
    for (; i + 7 < size; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        __m256 v_scaled_y = _mm256_mul_ps(vy, v_scale);
        __m256 vr = _mm256_add_ps(vx, v_scaled_y);
        _mm256_storeu_ps(result + i, vr);
    }
#endif

    // Handle remaining elements
    for (; i < size; ++i) {
        result[i] = x[i] + scale * y[i];
    }
}

// ============================================================================
// AVX RMSNorm
// ============================================================================

void rms_norm_avx(
    const float* input,
    const float* weight,
    int batch_size,
    int hidden_size,
    float eps,
    float* result
) {
    for (int b = 0; b < batch_size; ++b) {
        const float* x = input + b * hidden_size;
        float* r = result + b * hidden_size;

        // Compute sum of squares
        float sum_sq = 0.0f;
        int i = 0;

#ifdef HAS_AVX512
        __m512 v_sum_sq = _mm512_setzero_ps();
        for (; i + 15 < hidden_size; i += 16) {
            __m512 vx = _mm512_loadu_ps(x + i);
            __m256 v_sq_low = _mm256_mul_ps(_mm512_castps512_ps256(vx), _mm512_castps512_ps256(vx));
            __m256 v_sq_high = _mm256_mul_ps(_mm512_extractf32x8_ps(vx, 1), _mm512_extractf32x8_ps(vx, 1));
            v_sum_sq = _mm512_add_ps(_mm512_castps256_ps512(v_sq_low),
                                     _mm512_add_ps(_mm512_castps256_ps512(v_sq_high), v_sum_sq));
        }
        sum_sq = _mm512_reduce_add_ps(v_sum_sq);
#elif defined(HAS_AVX2)
        __m256 v_sum_sq = _mm256_setzero_ps();
        for (; i + 7 < hidden_size; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 v_sq = _mm256_mul_ps(vx, vx);
            v_sum_sq = _mm256_add_ps(v_sum_sq, v_sq);
        }
        float temp[8];
        _mm256_storeu_ps(temp, v_sum_sq);
        for (int j = 0; j < 8; ++j) sum_sq += temp[j];
#endif

        // Handle remaining elements
        for (; i < hidden_size; ++i) {
            sum_sq += x[i] * x[i];
        }

        // Compute RMS
        float rms = std::sqrt(sum_sq / hidden_size + eps);

        // Normalize and apply weight
        i = 0;
#ifdef HAS_AVX512
        __m512 v_rms = _mm512_set1_ps(1.0f / rms);
        for (; i + 15 < hidden_size; i += 16) {
            __m512 vx = _mm512_loadu_ps(x + i);
            __m512 vw = weight ? _mm512_loadu_ps(weight + i) : _mm512_set1_ps(1.0f);
            __m512 vr = _mm512_mul_ps(_mm512_mul_ps(vx, v_rms), vw);
            _mm512_storeu_ps(r + i, vr);
        }
#elif defined(HAS_AVX2)
        __m256 v_rms = _mm256_set1_ps(1.0f / rms);
        for (; i + 7 < hidden_size; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vw = weight ? _mm256_loadu_ps(weight + i) : _mm256_set1_ps(1.0f);
            __m256 vr = _mm256_mul_ps(_mm256_mul_ps(vx, v_rms), vw);
            _mm256_storeu_ps(r + i, vr);
        }
#endif

        // Handle remaining elements
        for (; i < hidden_size; ++i) {
            r[i] = (x[i] / rms) * (weight ? weight[i] : 1.0f);
        }
    }
}

} // namespace avx
} // namespace ops
} // namespace tensor_cpp
