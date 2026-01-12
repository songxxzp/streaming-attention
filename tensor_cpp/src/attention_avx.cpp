/**
 * @file attention_avx.cpp
 * @brief AVX2-optimized attention operators
 */

#include "tensor_cpp/attention_avx.h"
#include "tensor_cpp/ops.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>

using namespace tensor_cpp::ops;

namespace tensor_cpp {
namespace ops {
namespace avx2 {

// ============================================================================
// AVX2 Self-Attention
// ============================================================================

Tensor self_attention_avx2(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor* causal_mask,
    float scale
) {
    // q, k, v shapes: [batch, num_heads, seq_len, head_dim]
    const Shape& q_shape = q.shape();
    size_t batch = q_shape[0];
    size_t num_heads = q_shape[1];
    size_t seq_len = q_shape[2];
    size_t head_dim = q_shape[3];

    std::cerr << "DEBUG self_attention: batch=" << batch << ", heads=" << num_heads
              << ", seq_len=" << seq_len << ", head_dim=" << head_dim << std::endl;

    // Output: [batch, num_heads, seq_len, head_dim]
    size_t output_size = batch * num_heads * seq_len * head_dim;
    std::cerr << "DEBUG: Allocating output_data, size=" << output_size << std::endl;
    std::vector<float> output_data(output_size);

    #pragma omp parallel for if(batch * num_heads * seq_len > 10)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            // Process this head
            size_t head_offset = (b * num_heads + h) * seq_len * head_dim;

            for (size_t i = 0; i < seq_len; ++i) {
                // Compute attention scores for position i
                // scores[i, j] = scale * sum(q[b, h, i, :] * k[b, h, j, :])

                size_t out_offset = head_offset + i * head_dim;

                // First, compute all attention scores
                std::vector<float> scores(seq_len);

                for (size_t j = 0; j < seq_len; ++j) {
                    // Dot product: q[i, :] and k[j, :]
                    size_t q_offset = ((b * num_heads + h) * seq_len + i) * head_dim;
                    size_t k_offset = ((b * num_heads + h) * seq_len + j) * head_dim;

                    float sum = 0.0f;
                    size_t d = 0;

                    // AVX2 dot product
                    __m256 sum_vec = _mm256_setzero_ps();
                    for (; d + 8 <= head_dim; d += 8) {
                        __m256 q_vec = _mm256_loadu_ps(&q[q_offset + d]);
                        __m256 k_vec = _mm256_loadu_ps(&k[k_offset + d]);
                        sum_vec = _mm256_fmadd_ps(q_vec, k_vec, sum_vec);
                    }

                    // Horizontal sum
                    __m128 hi_quad = _mm256_extractf128_ps(sum_vec, 1);
                    __m128 lo_quad = _mm256_castps256_ps128(sum_vec);
                    __m128 sum_quad = _mm_add_ps(lo_quad, hi_quad);
                    __m128 lo_dual = sum_quad;
                    __m128 hi_dual = _mm_movehl_ps(sum_quad, sum_quad);
                    __m128 sum_dual = _mm_add_ps(lo_dual, hi_dual);
                    __m128 lo = sum_dual;
                    __m128 hi = _mm_shuffle_ps(sum_dual, sum_dual, 1);
                    __m128 sum_128 = _mm_add_ss(lo, hi);
                    sum = _mm_cvtss_f32(sum_128);

                    // Remaining elements
                    for (; d < head_dim; ++d) {
                        sum += q[q_offset + d] * k[k_offset + d];
                    }

                    scores[j] = sum * scale;
                }

                // Apply causal mask if provided
                if (causal_mask != nullptr) {
                    const float* mask_data = causal_mask->data();
                    for (size_t j = 0; j < seq_len; ++j) {
                        float mask_val = mask_data[i * seq_len + j];
                        if (mask_val == -std::numeric_limits<float>::infinity()) {
                            scores[j] = mask_val;
                        }
                    }
                }

                // Softmax: exp(score) / sum(exp(scores))
                // Compute max for numerical stability
                float max_score = -std::numeric_limits<float>::infinity();
                for (size_t j = 0; j < seq_len; ++j) {
                    max_score = std::max(max_score, scores[j]);
                }

                // Compute exp and sum
                std::vector<float> exp_scores(seq_len);
                float sum_exp = 0.0f;
                for (size_t j = 0; j < seq_len; ++j) {
                    float diff = scores[j] - max_score;
                    if (diff >= -20.0f) {  // Avoid underflow
                        exp_scores[j] = std::exp(diff);
                        sum_exp += exp_scores[j];
                    } else {
                        exp_scores[j] = 0.0f;
                    }
                }

                // Normalize and compute weighted sum of values
                for (size_t d = 0; d < head_dim; ++d) {
                    float weighted_sum = 0.0f;

                    for (size_t j = 0; j < seq_len; ++j) {
                        if (exp_scores[j] > 0.0f) {
                            float weight = exp_scores[j] / sum_exp;
                            size_t v_offset = ((b * num_heads + h) * seq_len + j) * head_dim;
                            weighted_sum += weight * v[v_offset + d];
                        }
                    }

                    output_data[out_offset + d] = weighted_sum;
                }
            }
        }
    }

    return Tensor(std::move(output_data), q_shape);
}

// ============================================================================
// AVX2 Linear (for QKV projections)
// ============================================================================

Tensor linear_avx2(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias
) {
    if (input.shape().ndim() < 2) {
        throw std::invalid_argument("Linear requires at least 2D input");
    }

    size_t ndim = input.shape().ndim();
    size_t in_features = input.shape()[ndim - 1];
    size_t out_features = weight.shape()[0];

    if (weight.shape()[1] != in_features) {
        throw std::invalid_argument("Weight shape mismatch in linear layer");
    }

    // Compute total number of "samples" (all dimensions except last)
    size_t num_samples = 1;
    for (size_t i = 0; i < ndim - 1; ++i) {
        num_samples *= input.shape()[i];
    }

    std::cerr << "DEBUG linear_avx2: num_samples=" << num_samples
              << ", in_features=" << in_features
              << ", out_features=" << out_features
              << ", output_size=" << (num_samples * out_features) << std::endl;

    std::vector<float> output(num_samples * out_features, 0.0f);

    #pragma omp parallel for if(num_samples * out_features > 100)
    for (size_t s = 0; s < num_samples; ++s) {
        for (size_t o = 0; o < out_features; ++o) {
            float sum = 0.0f;
            size_t weight_offset = o * in_features;
            size_t j = 0;

            // AVX2 dot product
            __m256 sum_vec = _mm256_setzero_ps();
            for (; j + 8 <= in_features; j += 8) {
                __m256 input_vec = _mm256_loadu_ps(&input[s * in_features + j]);
                __m256 weight_vec = _mm256_loadu_ps(&weight[weight_offset + j]);
                sum_vec = _mm256_fmadd_ps(input_vec, weight_vec, sum_vec);
            }

            // Horizontal sum
            __m128 hi_quad = _mm256_extractf128_ps(sum_vec, 1);
            __m128 lo_quad = _mm256_castps256_ps128(sum_vec);
            __m128 sum_quad = _mm_add_ps(lo_quad, hi_quad);
            __m128 lo_dual = sum_quad;
            __m128 hi_dual = _mm_movehl_ps(sum_quad, sum_quad);
            __m128 sum_dual = _mm_add_ps(lo_dual, hi_dual);
            __m128 lo = sum_dual;
            __m128 hi = _mm_shuffle_ps(sum_dual, sum_dual, 1);
            __m128 sum_128 = _mm_add_ss(lo, hi);
            sum = _mm_cvtss_f32(sum_128);

            // Remaining elements
            for (; j < in_features; ++j) {
                sum += input[s * in_features + j] * weight[weight_offset + j];
            }

            output[s * out_features + o] = sum;
        }
    }

    if (bias != nullptr) {
        #pragma omp parallel for if(num_samples * out_features > 100)
        for (size_t s = 0; s < num_samples; ++s) {
            for (size_t o = 0; o < out_features; ++o) {
                output[s * out_features + o] += (*bias)[o];
            }
        }
    }

    // Build output shape
    std::vector<size_t> out_shape_dims;
    for (size_t i = 0; i < ndim - 1; ++i) {
        out_shape_dims.push_back(input.shape()[i]);
    }
    out_shape_dims.push_back(out_features);

    return Tensor(std::move(output), Shape(out_shape_dims));
}

} // namespace avx2
} // namespace ops
} // namespace tensor_cpp
