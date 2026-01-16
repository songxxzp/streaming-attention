/**
 * @file ops.cpp
 * @brief Implementation of deep learning operators (non-template, float-only)
 *
 * Contains:
 * - Basic operators (add, argmax, embedding, linear, rms_norm, rope, swiglu)
 * - Standard attention (self_attention, cross_attention)
 * - Streaming attention (naive, streaming_serial, streaming_omp)
 * - MPI functions
 */

#include "tensor_cpp/ops.h"
#include <algorithm>
#include <limits>
#include <complex>
#include <random>
#include <cstring>

// AVX2 intrinsics
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #ifdef __AVX2__
    #include <immintrin.h>
    #define HAS_AVX2_OPS
    #endif
#endif

namespace tensor_cpp {
namespace ops {

// ============================================================================
// Online Softmax State (for Streaming Attention)
// ============================================================================

/**
 * Online Softmax Accumulator for Streaming Attention
 */
class OnlineSoftmaxState {
public:
    OnlineSoftmaxState() : m(-std::numeric_limits<float>::infinity()), l(1.0f) {}

    float m;  // Running maximum
    float l;  // Running normalizer
};

/**
 * Process a single block in streaming attention
 */
inline void process_streaming_block(
    const float* scores,
    const float* values,
    OnlineSoftmaxState& state,
    float* output,
    int block_size,
    int d
) {
    // Find max in current block and global max
    float m_new = state.m;
    for (int i = 0; i < block_size; ++i) {
        m_new = std::max(m_new, scores[i]);
    }

    // Compute sum of exp(scores - m_new) for this block
    float sum_exp_block = 0.0f;
    for (int i = 0; i < block_size; ++i) {
        sum_exp_block += std::exp(scores[i] - m_new);
    }

    // Update normalizer
    float scale_old = std::exp(state.m - m_new);
    float l_new = state.l * scale_old + sum_exp_block;

    // Update output
    float alpha_old = (state.l * scale_old) / l_new;
    float alpha_new = 1.0f / l_new;

    // Scale old output
    for (int j = 0; j < d; ++j) {
        output[j] *= alpha_old;
    }

    // Add weighted sum of values from this block
    for (int i = 0; i < block_size; ++i) {
        float weight = alpha_new * std::exp(scores[i] - m_new);
        const float* v_row = values + i * d;
        for (int j = 0; j < d; ++j) {
            output[j] += weight * v_row[j];
        }
    }

    // Update state
    state.m = m_new;
    state.l = l_new;
}

#ifdef HAS_AVX2_OPS
/**
 * AVX2-optimized version of process_streaming_block
 */
inline void process_streaming_block_avx2(
    const float* scores,
    const float* values,
    OnlineSoftmaxState& state,
    float* output,
    int block_size,
    int d
) {
    // Find max in current block and global max (AVX2)
    float m_new = state.m;
    int i = 0;

    #ifdef __AVX2__
    __m256 v_m_new = _mm256_set1_ps(m_new);
    for (; i + 7 < block_size; i += 8) {
        __m256 v_scores = _mm256_loadu_ps(scores + i);
        v_m_new = _mm256_max_ps(v_m_new, v_scores);
    }
    // Horizontal max
    float temp[8];
    _mm256_storeu_ps(temp, v_m_new);
    for (int j = 0; j < 8; ++j) {
        m_new = std::max(m_new, temp[j]);
    }
    #endif

    // Handle remaining elements
    for (; i < block_size; ++i) {
        m_new = std::max(m_new, scores[i]);
    }

    // Compute sum of exp(scores - m_new) for this block
    // Note: exp is scalar-only, AVX2 doesn't have exp intrinsic
    float sum_exp_block = 0.0f;
    for (int i = 0; i < block_size; ++i) {
        sum_exp_block += std::exp(scores[i] - m_new);
    }

    // Update normalizer
    float scale_old = std::exp(state.m - m_new);
    float l_new = state.l * scale_old + sum_exp_block;

    // Update output
    float alpha_old = (state.l * scale_old) / l_new;
    float alpha_new = 1.0f / l_new;

    // Scale old output (AVX2)
    int j = 0;
    #ifdef __AVX2__
    __m256 v_alpha_old = _mm256_set1_ps(alpha_old);
    for (; j + 7 < d; j += 8) {
        __m256 v_out = _mm256_loadu_ps(output + j);
        v_out = _mm256_mul_ps(v_out, v_alpha_old);
        _mm256_storeu_ps(output + j, v_out);
    }
    #endif

    // Handle remaining elements
    for (; j < d; ++j) {
        output[j] *= alpha_old;
    }

    // Add weighted sum of values from this block (AVX2)
    for (int i = 0; i < block_size; ++i) {
        float weight = alpha_new * std::exp(scores[i] - m_new);
        const float* v_row = values + i * d;

        int k = 0;
        #ifdef __AVX2__
        __m256 v_weight = _mm256_set1_ps(weight);
        for (; k + 7 < d; k += 8) {
            __m256 v_out = _mm256_loadu_ps(output + k);
            __m256 v_val = _mm256_loadu_ps(v_row + k);
            __m256 v_weighted = _mm256_mul_ps(v_weight, v_val);
            v_out = _mm256_add_ps(v_out, v_weighted);
            _mm256_storeu_ps(output + k, v_out);
        }
        #endif

        // Handle remaining elements
        for (; k < d; ++k) {
            output[k] += weight * v_row[k];
        }
    }

    // Update state
    state.m = m_new;
    state.l = l_new;
}
#endif

// ============================================================================
// Element-wise Addition
// ============================================================================

Tensor add(const Tensor& input, const Tensor& other, float alpha) {
    if (input.shape() != other.shape()) {
        throw std::invalid_argument("Shape mismatch for addition");
    }

    std::vector<float> result(input.size());

    #pragma omp parallel for if(input.size() > 1000)
    for (size_t i = 0; i < input.size(); ++i) {
        result[i] = input[i] + alpha * other[i];
    }

    return Tensor(std::move(result), input.shape());
}

// ============================================================================
// Argmax
// ============================================================================

TensorL argmax(const Tensor& input, int dim, bool keepdim) {
    if (dim == -1) {
        // Global argmax
        size_t max_idx = 0;
        float max_val = input[0];

        for (size_t i = 1; i < input.size(); ++i) {
            if (input[i] > max_val) {
                max_val = input[i];
                max_idx = i;
            }
        }

        std::vector<long> result_data = {static_cast<long>(max_idx)};
        return TensorL(result_data, Shape({1}));
    } else if (dim == static_cast<int>(input.shape().ndim()) - 1) {
        // Argmax along last dimension (e.g., [batch, vocab_size] -> [batch])
        size_t ndim = input.shape().ndim();
        if (ndim < 2) {
            throw std::invalid_argument("Argmax along last dim requires at least 2D input");
        }

        size_t last_dim = input.shape()[ndim - 1];
        size_t num_samples = 1;
        for (size_t i = 0; i < ndim - 1; ++i) {
            num_samples *= input.shape()[i];
        }

        std::vector<long> result_data(num_samples);

        #pragma omp parallel for if(num_samples > 100)
        for (size_t s = 0; s < num_samples; ++s) {
            size_t max_idx = 0;
            float max_val = input[s * last_dim];

            for (size_t i = 1; i < last_dim; ++i) {
                if (input[s * last_dim + i] > max_val) {
                    max_val = input[s * last_dim + i];
                    max_idx = i;
                }
            }
            result_data[s] = static_cast<long>(max_idx);
        }

        // Build output shape (same as input except last dimension removed)
        std::vector<size_t> out_shape_dims;
        for (size_t i = 0; i < ndim - 1; ++i) {
            if (keepdim || i < ndim - 2) {
                out_shape_dims.push_back(input.shape()[i]);
            }
        }
        if (out_shape_dims.empty()) {
            out_shape_dims.push_back(1);
        }

        return TensorL(result_data, Shape(out_shape_dims));
    } else {
        throw std::runtime_error("Argmax along dimensions other than -1 or last is not yet implemented");
    }
}

// ============================================================================
// Embedding Lookup
// ============================================================================

Tensor embedding(const TensorL& indices, const Tensor& weight,
                long padding_idx) {
    size_t num_embeddings = weight.shape()[0];
    size_t embedding_dim = weight.shape()[1];
    size_t num_indices = indices.size();

    std::vector<float> result(num_indices * embedding_dim);

    #pragma omp parallel for if(num_indices * embedding_dim > 1000)
    for (size_t i = 0; i < num_indices; ++i) {
        long idx = indices[i];

        if (idx < 0 || idx >= static_cast<long>(num_embeddings)) {
            if (padding_idx >= 0 && idx == padding_idx) {
                for (size_t j = 0; j < embedding_dim; ++j) {
                    result[i * embedding_dim + j] = 0.0f;
                }
            } else {
                throw std::out_of_range("Embedding index out of range");
            }
        } else {
            for (size_t j = 0; j < embedding_dim; ++j) {
                result[i * embedding_dim + j] = weight[idx * embedding_dim + j];
            }
        }
    }

    Shape out_shape = indices.shape();
    out_shape.dims.push_back(embedding_dim);

    return Tensor(std::move(result), out_shape);
}

// ============================================================================
// Linear Layer
// ============================================================================

Tensor linear(const Tensor& input, const Tensor& weight,
              const Tensor* bias) {
    if (input.shape().ndim() < 2) {
        throw std::invalid_argument("Linear requires at least 2D input");
    }

    size_t ndim = input.shape().ndim();
    size_t in_features = input.shape()[ndim - 1];  // Last dimension
    size_t out_features = weight.shape()[0];

    if (weight.shape()[1] != in_features) {
        throw std::invalid_argument("Weight shape mismatch in linear layer");
    }

    // Compute total number of "samples" (all dimensions except last)
    size_t num_samples = 1;
    for (size_t i = 0; i < ndim - 1; ++i) {
        num_samples *= input.shape()[i];
    }

    std::vector<float> output(num_samples * out_features, 0.0f);

    #pragma omp parallel for if(num_samples * out_features * in_features > 1000)
    for (size_t s = 0; s < num_samples; ++s) {
        for (size_t o = 0; o < out_features; ++o) {
            float sum = 0.0f;
            for (size_t i = 0; i < in_features; ++i) {
                // PyTorch linear: output = input @ weight.T
                // weight shape: [out_features, in_features]
                // For output[o], we need: sum over i of input[i] * weight[o, i]
                // In row-major: weight[o, i] = weight[o * in_features + i]
                sum += input[s * in_features + i] * weight[o * in_features + i];
            }
            output[s * out_features + o] = sum;
        }
    }

    if (bias != nullptr) {
        #pragma omp parallel for if(num_samples * out_features > 1000)
        for (size_t s = 0; s < num_samples; ++s) {
            for (size_t o = 0; o < out_features; ++o) {
                output[s * out_features + o] += (*bias)[o];
            }
        }
    }

    // Build output shape (same as input except last dimension)
    std::vector<size_t> out_shape_dims;
    for (size_t i = 0; i < ndim - 1; ++i) {
        out_shape_dims.push_back(input.shape()[i]);
    }
    out_shape_dims.push_back(out_features);

    return Tensor(std::move(output), Shape(out_shape_dims));
}

// ============================================================================
// RMS Normalization
// ============================================================================

Tensor rms_norm(const Tensor& input, const Tensor* weight,
                float eps, int dim) {
    if (input.shape().ndim() < 2) {
        throw std::invalid_argument("RMS norm requires at least 2D input");
    }

    // Get dimensions
    size_t ndim = input.shape().ndim();
    size_t hidden_size = input.shape()[ndim - 1];  // Last dimension

    // Compute total number of "samples" (all dimensions except last)
    size_t num_samples = 1;
    for (size_t i = 0; i < ndim - 1; ++i) {
        num_samples *= input.shape()[i];
    }

    std::vector<float> output(input.size());

    #pragma omp parallel for if(num_samples * hidden_size > 1000)
    for (size_t s = 0; s < num_samples; ++s) {
        // Compute RMS for this sample
        float mean_square = 0.0f;
        for (size_t i = 0; i < hidden_size; ++i) {
            float val = input[s * hidden_size + i];
            mean_square += val * val;
        }
        mean_square /= static_cast<float>(hidden_size);
        float rms = std::sqrt(mean_square + eps);

        // Normalize
        for (size_t i = 0; i < hidden_size; ++i) {
            output[s * hidden_size + i] = input[s * hidden_size + i] / rms;
        }

        // Apply gain if provided
        if (weight != nullptr) {
            for (size_t i = 0; i < hidden_size; ++i) {
                output[s * hidden_size + i] *= (*weight)[i];
            }
        }
    }

    return Tensor(std::move(output), input.shape());
}

// ============================================================================
// Rotary Position Embedding (RoPE)
// ============================================================================

RotaryEmbedding::RotaryEmbedding(size_t dim, size_t max_seq_len, float theta)
    : dim_(dim), max_seq_len_(max_seq_len), theta_(theta) {
    if (dim % 2 != 0) {
        throw std::invalid_argument("Rotary embedding dimension must be even");
    }
    precompute_freqs_cis();
}

void RotaryEmbedding::precompute_freqs_cis() {
    std::vector<float> freqs(dim_ / 2);
    for (size_t i = 0; i < dim_ / 2; ++i) {
        freqs[i] = 1.0f / std::pow(theta_, static_cast<float>(i) / static_cast<float>(dim_));
    }

    freqs_cis_.resize(max_seq_len_ * (dim_ / 2));

    #pragma omp parallel for if(max_seq_len_ * dim_ / 2 > 1000)
    for (size_t t = 0; t < max_seq_len_; ++t) {
        for (size_t i = 0; i < dim_ / 2; ++i) {
            float angle = static_cast<float>(t) * freqs[i];
            freqs_cis_[t * (dim_ / 2) + i] = std::complex<float>(std::cos(angle), std::sin(angle));
        }
    }
}

Tensor RotaryEmbedding::apply(const Tensor& input, size_t seq_len) {
    if (input.shape().ndim() != 4) {
        throw std::invalid_argument("RoPE requires 4D input: (batch, seq, heads, head_dim)");
    }

    size_t batch_size = input.shape()[0];
    size_t num_heads = input.shape()[2];
    size_t head_dim = input.shape()[3];

    if (head_dim != dim_) {
        throw std::invalid_argument("Head dimension must match rotary embedding dimension");
    }

    std::vector<float> output(input.size());

    #pragma omp parallel for if(batch_size * seq_len * num_heads * head_dim > 1000)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            for (size_t h = 0; h < num_heads; ++h) {
                const std::complex<float>* cis = &freqs_cis_[s * (dim_ / 2)];

                for (size_t i = 0; i < dim_ / 2; ++i) {
                    size_t base_idx = ((b * input.shape()[1] + s) * num_heads + h) * head_dim;

                    float x_real = input[base_idx + i];
                    float x_imag = input[base_idx + i + dim_ / 2];

                    std::complex<float> x_complex(x_real, x_imag);
                    std::complex<float> rotated = x_complex * cis[i];

                    output[base_idx + i] = rotated.real();
                    output[base_idx + i + dim_ / 2] = rotated.imag();
                }
            }
        }
    }

    return Tensor(std::move(output), input.shape());
}

// ============================================================================
// SwiGLU Activation Function
// ============================================================================

Tensor swiglu(const Tensor& x, const Tensor& gate) {
    if (x.shape() != gate.shape()) {
        throw std::invalid_argument("Shape mismatch for SwiGLU");
    }

    std::vector<float> result(x.size());

    #pragma omp parallel for if(x.size() > 1000)
    for (size_t i = 0; i < x.size(); ++i) {
        float sigmoid_gate = 1.0f / (1.0f + std::exp(-gate[i]));
        float silu = gate[i] * sigmoid_gate;
        result[i] = x[i] * silu;
    }

    return Tensor(std::move(result), x.shape());
}

// ============================================================================
// Self Attention (Standard)
// ============================================================================

Tensor self_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor* mask,
    float scale
) {
    size_t batch_size = query.shape()[0];
    size_t num_heads = query.shape()[1];
    size_t q_seq_len = query.shape()[2];
    size_t k_seq_len = key.shape()[2];  // Key sequence length may differ from query!
    size_t head_dim = query.shape()[3];

    // Compute attention scores: Q @ K^T
    // Output shape: [batch, num_heads, q_seq_len, k_seq_len]
    std::vector<float> scores(batch_size * num_heads * q_seq_len * k_seq_len);

    #pragma omp parallel for if(batch_size * num_heads * q_seq_len * k_seq_len > 1000)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < q_seq_len; ++i) {
                for (size_t j = 0; j < k_seq_len; ++j) {
                    float sum = 0.0f;
                    for (size_t d = 0; d < head_dim; ++d) {
                        size_t q_idx = ((b * num_heads + h) * q_seq_len + i) * head_dim + d;
                        size_t k_idx = ((b * num_heads + h) * k_seq_len + j) * head_dim + d;
                        sum += query[q_idx] * key[k_idx];
                    }

                    size_t score_idx = ((b * num_heads + h) * q_seq_len + i) * k_seq_len + j;
                    scores[score_idx] = sum * scale;
                }
            }
        }
    }

    // Apply mask if provided (CRITICAL: must be done before softmax!)
    if (mask != nullptr) {
        const Shape& mask_shape = mask->shape();
        // mask shape should be [1, 1, k_seq_len, k_seq_len] or [k_seq_len, k_seq_len]
        if (mask_shape.ndim() == 4) {
            // mask: [1, 1, k_seq_len, k_seq_len]
            #pragma omp parallel for if(batch_size * num_heads * q_seq_len * k_seq_len > 1000)
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t h = 0; h < num_heads; ++h) {
                    for (size_t i = 0; i < q_seq_len; ++i) {
                        for (size_t j = 0; j < k_seq_len; ++j) {
                            size_t score_idx = ((b * num_heads + h) * q_seq_len + i) * k_seq_len + j;
                            size_t mask_idx = i * k_seq_len + j;  // mask[i, j] for causal mask
                            scores[score_idx] += (*mask)[mask_idx];
                        }
                    }
                }
            }
        } else if (mask_shape.ndim() == 2) {
            // mask: [k_seq_len, k_seq_len]
            #pragma omp parallel for if(batch_size * num_heads * q_seq_len * k_seq_len > 1000)
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t h = 0; h < num_heads; ++h) {
                    for (size_t i = 0; i < q_seq_len; ++i) {
                        for (size_t j = 0; j < k_seq_len; ++j) {
                            size_t score_idx = ((b * num_heads + h) * q_seq_len + i) * k_seq_len + j;
                            size_t mask_idx = i * k_seq_len + j;
                            scores[score_idx] += (*mask)[mask_idx];
                        }
                    }
                }
            }
        }
    }

    // Apply softmax
    std::vector<float> attn_weights(batch_size * num_heads * q_seq_len * k_seq_len);

    #pragma omp parallel for if(batch_size * num_heads * q_seq_len > 100)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < q_seq_len; ++i) {
                // Find max for this row
                float max_score = -std::numeric_limits<float>::infinity();
                for (size_t j = 0; j < k_seq_len; ++j) {
                    size_t score_idx = ((b * num_heads + h) * q_seq_len + i) * k_seq_len + j;
                    max_score = std::max(max_score, scores[score_idx]);
                }

                // Compute exp and sum
                float sum_exp = 0.0f;
                for (size_t j = 0; j < k_seq_len; ++j) {
                    size_t score_idx = ((b * num_heads + h) * q_seq_len + i) * k_seq_len + j;
                    float exp_val = std::exp(scores[score_idx] - max_score);
                    attn_weights[score_idx] = exp_val;
                    sum_exp += exp_val;
                }

                // Normalize
                for (size_t j = 0; j < k_seq_len; ++j) {
                    size_t score_idx = ((b * num_heads + h) * q_seq_len + i) * k_seq_len + j;
                    attn_weights[score_idx] /= sum_exp;
                }
            }
        }
    }

    // Compute output: attn_weights @ value
    // value shape: [batch, num_heads, k_seq_len, head_dim]
    // output shape: [batch, num_heads, q_seq_len, head_dim]
    std::vector<float> output(batch_size * num_heads * q_seq_len * head_dim);

    #pragma omp parallel for if(batch_size * num_heads * q_seq_len * head_dim > 1000)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < q_seq_len; ++i) {
                for (size_t d = 0; d < head_dim; ++d) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < k_seq_len; ++j) {
                        size_t weight_idx = ((b * num_heads + h) * q_seq_len + i) * k_seq_len + j;
                        size_t value_idx = ((b * num_heads + h) * k_seq_len + j) * head_dim + d;
                        sum += attn_weights[weight_idx] * value[value_idx];
                    }
                    size_t out_idx = ((b * num_heads + h) * q_seq_len + i) * head_dim + d;
                    output[out_idx] = sum;
                }
            }
        }
    }

    return Tensor(std::move(output), query.shape());
}

// ============================================================================
// Cross Attention (Standard)
// ============================================================================

Tensor cross_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor* mask,
    float scale
) {
    size_t batch_size = query.shape()[0];
    size_t num_heads = query.shape()[1];
    size_t query_len = query.shape()[2];
    size_t kv_len = key.shape()[2];
    size_t head_dim = query.shape()[3];

    // Compute attention scores: Q @ K^T
    std::vector<float> scores(batch_size * num_heads * query_len * kv_len);

    #pragma omp parallel for if(batch_size * num_heads * query_len * kv_len > 1000)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < query_len; ++i) {
                for (size_t j = 0; j < kv_len; ++j) {
                    float sum = 0.0f;
                    for (size_t d = 0; d < head_dim; ++d) {
                        size_t q_idx = ((b * num_heads + h) * query_len + i) * head_dim + d;
                        size_t k_idx = ((b * num_heads + h) * kv_len + j) * head_dim + d;
                        sum += query[q_idx] * key[k_idx];
                    }

                    size_t score_idx = ((b * num_heads + h) * query_len + i) * kv_len + j;
                    scores[score_idx] = sum * scale;
                }
            }
        }
    }

    // Apply softmax
    std::vector<float> attn_weights(batch_size * num_heads * query_len * kv_len);

    #pragma omp parallel for if(batch_size * num_heads * query_len > 100)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < query_len; ++i) {
                // Find max for this row
                float max_score = -std::numeric_limits<float>::infinity();
                for (size_t j = 0; j < kv_len; ++j) {
                    size_t score_idx = ((b * num_heads + h) * query_len + i) * kv_len + j;
                    max_score = std::max(max_score, scores[score_idx]);
                }

                // Compute exp and sum
                float sum_exp = 0.0f;
                for (size_t j = 0; j < kv_len; ++j) {
                    size_t score_idx = ((b * num_heads + h) * query_len + i) * kv_len + j;
                    float exp_val = std::exp(scores[score_idx] - max_score);
                    attn_weights[score_idx] = exp_val;
                    sum_exp += exp_val;
                }

                // Normalize
                for (size_t j = 0; j < kv_len; ++j) {
                    size_t score_idx = ((b * num_heads + h) * query_len + i) * kv_len + j;
                    attn_weights[score_idx] /= sum_exp;
                }
            }
        }
    }

    // Compute output: attn_weights @ value
    std::vector<float> output(batch_size * num_heads * query_len * head_dim);

    #pragma omp parallel for if(batch_size * num_heads * query_len * head_dim > 1000)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < query_len; ++i) {
                for (size_t d = 0; d < head_dim; ++d) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < kv_len; ++j) {
                        size_t weight_idx = ((b * num_heads + h) * query_len + i) * kv_len + j;
                        size_t value_idx = ((b * num_heads + h) * kv_len + j) * head_dim + d;
                        sum += attn_weights[weight_idx] * value[value_idx];
                    }
                    size_t out_idx = ((b * num_heads + h) * query_len + i) * head_dim + d;
                    output[out_idx] = sum;
                }
            }
        }
    }

    return Tensor(std::move(output), query.shape());
}

// ============================================================================
// Streaming Attention - Naive (Serial Baseline)
// ============================================================================

std::vector<float> naive_attention_serial(
    const float* Q,
    const float* K,
    const float* V,
    int T,
    int d
) {
    // Step 1: Compute scores S = Q @ K^T
    std::vector<float> scores(T);

    for (int i = 0; i < T; ++i) {
        float sum = 0.0f;
        const float* K_row = K + i * d;
        for (int j = 0; j < d; ++j) {
            sum += Q[j] * K_row[j];
        }
        scores[i] = sum;
    }

    // Step 2: Compute softmax(S)
    float max_score = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < T; ++i) {
        max_score = std::max(max_score, scores[i]);
    }

    std::vector<float> exp_scores(T);
    float sum_exp = 0.0f;
    for (int i = 0; i < T; ++i) {
        exp_scores[i] = std::exp(scores[i] - max_score);
        sum_exp += exp_scores[i];
    }

    for (int i = 0; i < T; ++i) {
        exp_scores[i] /= sum_exp;
    }

    // Step 3: Compute output O = softmax(S) @ V
    std::vector<float> output(d, 0.0f);

    for (int i = 0; i < T; ++i) {
        float weight = exp_scores[i];
        const float* V_row = V + i * d;
        for (int j = 0; j < d; ++j) {
            output[j] += weight * V_row[j];
        }
    }

    return output;
}

// ============================================================================
// Streaming Attention - Serial (Online Softmax)
// ============================================================================

std::vector<float> streaming_attention_serial(
    const float* Q,
    const float* K,
    const float* V,
    int T,
    int d,
    int block_size
) {
    std::vector<float> output(d, 0.0f);
    OnlineSoftmaxState state;

    int num_blocks = (T + block_size - 1) / block_size;

    for (int block = 0; block < num_blocks; ++block) {
        int block_start = block * block_size;
        int block_end = std::min(block_start + block_size, T);
        int current_block_size = block_end - block_start;

        // Compute scores for this block
        std::vector<float> scores(current_block_size);

        for (int i = 0; i < current_block_size; ++i) {
            int global_idx = block_start + i;
            float sum = 0.0f;
            const float* K_row = K + global_idx * d;

            for (int j = 0; j < d; ++j) {
                sum += Q[j] * K_row[j];
            }
            scores[i] = sum;
        }

        const float* V_block = V + block_start * d;

        // Process this block using online softmax
        process_streaming_block(
            scores.data(),
            V_block,
            state,
            output.data(),
            current_block_size,
            d
        );
    }

    return output;
}

// ============================================================================
// Streaming Attention - OpenMP
// ============================================================================

struct PartialResult : public OnlineSoftmaxState {
    std::vector<float> O;

    PartialResult(int d) : OnlineSoftmaxState(), O(d, 0.0f) {}
};

inline void merge_partial_results(PartialResult& left, const PartialResult& right) {
    int d = left.O.size();

    float m_new = std::max(left.m, right.m);
    float scale_left = std::exp(left.m - m_new);
    float scale_right = std::exp(right.m - m_new);
    float l_new = left.l * scale_left + right.l * scale_right;

    float alpha_left = (left.l * scale_left) / l_new;
    float alpha_right = (right.l * scale_right) / l_new;

    for (int j = 0; j < d; ++j) {
        left.O[j] = left.O[j] * alpha_left + right.O[j] * alpha_right;
    }

    left.m = m_new;
    left.l = l_new;
}

inline PartialResult process_blocks_serial(
    const float* Q,
    const float* K,
    const float* V,
    int T_start,
    int T_end,
    int d,
    int block_size
) {
    PartialResult result(d);
    int T_chunk = T_end - T_start;

    for (int block = 0; block < (T_chunk + block_size - 1) / block_size; ++block) {
        int block_start = T_start + block * block_size;
        int block_end = std::min(T_start + block * block_size + block_size, T_end);
        int current_block_size = block_end - block_start;

        // Compute scores for this block
        std::vector<float> scores(current_block_size);
        for (int i = 0; i < current_block_size; ++i) {
            int global_idx = block_start + i;
            float sum = 0.0f;
            const float* K_row = K + global_idx * d;
            for (int j = 0; j < d; ++j) {
                sum += Q[j] * K_row[j];
            }
            scores[i] = sum;
        }

        const float* V_block = V + block_start * d;
        process_streaming_block(scores.data(), V_block, result, result.O.data(),
                                current_block_size, d);
    }

    return result;
}

std::vector<float> streaming_attention_omp(
    const float* Q,
    const float* K,
    const float* V,
    int T,
    int d,
    int block_size,
    int num_threads
) {
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    int n_threads = omp_get_max_threads();
    int min_blocks_per_thread = 2;
    int n_chunks = std::min(n_threads, (T + block_size * min_blocks_per_thread - 1) /
                             (block_size * min_blocks_per_thread));
    n_chunks = std::max(1, n_chunks);

    std::vector<PartialResult> partials;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        #pragma omp single
        {
            partials.resize(n_chunks, PartialResult(d));
        }

        #pragma omp barrier

        if (tid < n_chunks) {
            int chunk_size = (T + n_chunks - 1) / n_chunks;
            int T_start = tid * chunk_size;
            int T_end = std::min(T_start + chunk_size, T);

            partials[tid] = process_blocks_serial(Q, K, V, T_start, T_end, d, block_size);
        }
    }

    // Sequential tree reduction
    PartialResult merged(d);
    for (int i = 0; i < n_chunks; ++i) {
        merge_partial_results(merged, partials[i]);
    }

    return merged.O;
}

// ============================================================================
// Multi-head Streaming Attention (Tensor wrapper)
// ============================================================================

Tensor self_attention_streaming(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    float scale,
    int block_size
) {
    // query: [batch, num_heads, 1, head_dim]
    // key:   [batch, num_heads, kv_seq_len, head_dim]
    // value: [batch, num_heads, kv_seq_len, head_dim]
    const Shape& q_shape = query.shape();
    size_t batch = q_shape[0];
    size_t num_heads = q_shape[1];
    size_t q_seq_len = q_shape[2];
    size_t head_dim = q_shape[3];

    const Shape& k_shape = key.shape();
    size_t kv_seq_len = k_shape[2];

    // Output: [batch, num_heads, q_seq_len, head_dim]
    size_t output_size = batch * num_heads * q_seq_len * head_dim;
    std::vector<float> output_data(output_size);

    #pragma omp parallel for if(batch * num_heads > 1)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            // Process each head independently
            for (size_t i = 0; i < q_seq_len; ++i) {
                // Get query vector for this position: [head_dim]
                size_t q_offset = ((b * num_heads + h) * q_seq_len + i) * head_dim;
                const float* Q = query.data() + q_offset;

                // Get key and value matrices: [kv_seq_len, head_dim]
                size_t kv_offset = (b * num_heads + h) * kv_seq_len * head_dim;
                const float* K = key.data() + kv_offset;
                const float* V = value.data() + kv_offset;

                // Call streaming attention for this head
                // Note: scale is applied inside streaming_attention_omp
                std::vector<float> result = streaming_attention_omp(
                    Q, K, V, kv_seq_len, head_dim, block_size, 0
                );

                // Copy result to output
                size_t out_offset = ((b * num_heads + h) * q_seq_len + i) * head_dim;
                for (size_t d = 0; d < head_dim; ++d) {
                    output_data[out_offset + d] = result[d] * scale;
                }
            }
        }
    }

    return Tensor(std::move(output_data), query.shape());
}

// ============================================================================
// Block-wise Streaming Attention (for Prefill Phase)
// ============================================================================

/**
 * Process a Q block with causal constraint
 * Each query position in the block maintains its own online softmax state
 */
inline void process_q_block_causal(
    const float* Q_block,          // [q_block_size, head_dim]
    const float* K_all,            // [kv_seq_len, head_dim]
    const float* V_all,            // [kv_seq_len, head_dim]
    float* output_block,           // [q_block_size, head_dim]
    int q_block_start,             // Starting position of this Q block
    int q_block_size,              // Number of queries in this block
    int kv_seq_len,                // Total KV sequence length
    int head_dim,                  // Head dimension
    int kv_block_size,             // KV block size
    float scale                    // Scaling factor
) {
    // Each query position in the block has its own state
    std::vector<OnlineSoftmaxState> states(q_block_size);
    std::vector<std::vector<float>> outputs(q_block_size, std::vector<float>(head_dim, 0.0f));

    // Process KV blocks sequentially
    int num_kv_blocks = (kv_seq_len + kv_block_size - 1) / kv_block_size;

    for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++kv_block_idx) {
        int kv_start = kv_block_idx * kv_block_size;
        int kv_end = std::min(kv_start + kv_block_size, kv_seq_len);
        int current_kv_size = kv_end - kv_start;

        const float* K_block = K_all + kv_start * head_dim;
        const float* V_block = V_all + kv_start * head_dim;

        // For each query in the Q block
        for (int q_local = 0; q_local < q_block_size; ++q_local) {
            int q_global = q_block_start + q_local;

            // Check if this KV block is relevant (causal constraint)
            if (kv_start >= q_global + 1) {
                // This KV block is entirely in the future, skip
                continue;
            }

            // Compute effective KV range for this query (causal)
            int effective_kv_start = kv_start;
            int effective_kv_end = std::min(kv_end, q_global + 1);

            if (effective_kv_start >= effective_kv_end) {
                continue;  // No valid positions in this block
            }

            int effective_size = effective_kv_end - effective_kv_start;

            // Compute scores for this query against the effective KV range
            std::vector<float> scores(effective_size);

            const float* q_vec = Q_block + q_local * head_dim;

            for (int kv_local = 0; kv_local < effective_size; ++kv_local) {
                int kv_global = effective_kv_start + kv_local;
                const float* k_vec = K_all + kv_global * head_dim;

                // Dot product
                float sum = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    sum += q_vec[d] * k_vec[d];
                }
                scores[kv_local] = sum * scale;
            }

            // Get corresponding V values
            const float* V_effective = V_all + effective_kv_start * head_dim;

            // Update online softmax for this query
            process_streaming_block(
                scores.data(),
                V_effective,
                states[q_local],
                outputs[q_local].data(),
                effective_size,
                head_dim
            );
        }
    }

    // Copy outputs to output block
    for (int q_local = 0; q_local < q_block_size; ++q_local) {
        std::memcpy(output_block + q_local * head_dim,
                   outputs[q_local].data(),
                   head_dim * sizeof(float));
    }
}

#ifdef HAS_AVX2_OPS
/**
 * AVX2-optimized version of process_q_block_causal
 */
inline void process_q_block_causal_avx2(
    const float* Q_block,          // [q_block_size, head_dim]
    const float* K_all,            // [kv_seq_len, head_dim]
    const float* V_all,            // [kv_seq_len, head_dim]
    float* output_block,           // [q_block_size, head_dim]
    int q_block_start,             // Starting position of this Q block
    int q_block_size,              // Number of queries in this block
    int kv_seq_len,                // Total KV sequence length
    int head_dim,                  // Head dimension
    int kv_block_size,             // KV block size
    float scale                    // Scaling factor
) {
    // OPTIMIZATION 3: Add alignment hints for AVX2 loads
    // Assumes 32-byte alignment for better AVX2 performance (vmovaps vs vmovups)
    const float* Q_block_aligned = (const float*)__builtin_assume_aligned(Q_block, 32);
    const float* K_all_aligned = (const float*)__builtin_assume_aligned(K_all, 32);
    const float* V_all_aligned = (const float*)__builtin_assume_aligned(V_all, 32);
    float* output_aligned = (float*)__builtin_assume_aligned(output_block, 32);

    // Each query position in the block has its own state
    std::vector<OnlineSoftmaxState> states(q_block_size);
    std::vector<std::vector<float>> outputs(q_block_size, std::vector<float>(head_dim, 0.0f));

    // Process KV blocks sequentially
    int num_kv_blocks = (kv_seq_len + kv_block_size - 1) / kv_block_size;

    for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++kv_block_idx) {
        int kv_start = kv_block_idx * kv_block_size;
        int kv_end = std::min(kv_start + kv_block_size, kv_seq_len);
        int current_kv_size = kv_end - kv_start;

        // Use aligned pointers
        const float* K_block = K_all_aligned + kv_start * head_dim;
        const float* V_block = V_all_aligned + kv_start * head_dim;

        // For each query in the Q block
        for (int q_local = 0; q_local < q_block_size; ++q_local) {
            int q_global = q_block_start + q_local;

            // Check if this KV block is relevant (causal constraint)
            if (kv_start >= q_global + 1) {
                // This KV block is entirely in the future, skip
                continue;
            }

            // Compute effective KV range for this query (causal)
            int effective_kv_start = kv_start;
            int effective_kv_end = std::min(kv_end, q_global + 1);

            if (effective_kv_start >= effective_kv_end) {
                continue;  // No valid positions in this block
            }

            int effective_size = effective_kv_end - effective_kv_start;

            // Compute scores for this query against the effective KV range
            std::vector<float> scores(effective_size);

            const float* q_vec = Q_block_aligned + q_local * head_dim;

            // AVX2-optimized dot product
            for (int kv_local = 0; kv_local < effective_size; ++kv_local) {
                int kv_global = effective_kv_start + kv_local;
                const float* k_vec = K_all_aligned + kv_global * head_dim;

                // AVX2 dot product
                float sum = 0.0f;
                int d = 0;

                #ifdef __AVX2__
                __m256 v_sum0 = _mm256_setzero_ps();
                __m256 v_sum1 = _mm256_setzero_ps();

                // Process 16 elements at a time
                for (; d + 15 < head_dim; d += 16) {
                    __m256 v_q0 = _mm256_loadu_ps(q_vec + d);
                    __m256 v_k0 = _mm256_loadu_ps(k_vec + d);
                    v_sum0 = _mm256_fmadd_ps(v_q0, v_k0, v_sum0);

                    __m256 v_q1 = _mm256_loadu_ps(q_vec + d + 8);
                    __m256 v_k1 = _mm256_loadu_ps(k_vec + d + 8);
                    v_sum1 = _mm256_fmadd_ps(v_q1, v_k1, v_sum1);
                }

                // Process 8 elements at a time
                for (; d + 7 < head_dim; d += 8) {
                    __m256 v_q = _mm256_loadu_ps(q_vec + d);
                    __m256 v_k = _mm256_loadu_ps(k_vec + d);
                    v_sum0 = _mm256_fmadd_ps(v_q, v_k, v_sum0);
                }

                // Horizontal sum
                v_sum0 = _mm256_add_ps(v_sum0, v_sum1);
                v_sum0 = _mm256_hadd_ps(v_sum0, v_sum0);
                v_sum0 = _mm256_hadd_ps(v_sum0, v_sum0);

                float temp[8];
                _mm256_storeu_ps(temp, v_sum0);
                sum = temp[0] + temp[4];
                #endif

                // Handle remaining elements
                for (; d < head_dim; ++d) {
                    sum += q_vec[d] * k_vec[d];
                }

                scores[kv_local] = sum * scale;
            }

            // Get corresponding V values
            const float* V_effective = V_all_aligned + effective_kv_start * head_dim;

            // Update online softmax for this query (AVX2 version)
            process_streaming_block_avx2(
                scores.data(),
                V_effective,
                states[q_local],
                outputs[q_local].data(),
                effective_size,
                head_dim
            );
        }
    }

    // Copy outputs to output block (AVX2)
    for (int q_local = 0; q_local < q_block_size; ++q_local) {
        std::memcpy(output_aligned + q_local * head_dim,
                   outputs[q_local].data(),
                   head_dim * sizeof(float));
    }
}

/**
 * OPTIMIZATION 2: Cache-optimized version with KV blocks in outer loop
 * This improves L2/L3 cache hit rate by reusing K/V blocks across multiple queries
 *
 * Key insight: Instead of each query scanning all K/V independently,
 * we iterate through KV blocks once and update all relevant queries.
 * This is similar to FlashAttention's tiling strategy.
 */
inline void process_q_block_causal_avx2_cache_optimized(
    const float* Q_block,          // [q_block_size, head_dim]
    const float* K_all,            // [kv_seq_len, head_dim]
    const float* V_all,            // [kv_seq_len, head_dim]
    float* output_block,           // [q_block_size, head_dim]
    int q_block_start,             // Starting position of this Q block
    int q_block_size,              // Number of queries in this block
    int kv_seq_len,                // Total KV sequence length
    int head_dim,                  // Head dimension
    int kv_block_size,             // KV block size
    float scale                    // Scaling factor
) {
    // Each query position maintains its own online softmax state
    std::vector<OnlineSoftmaxState> states(q_block_size);
    std::vector<std::vector<float>> outputs(q_block_size, std::vector<float>(head_dim, 0.0f));

    // OPTIMIZATION 2: KV blocks in OUTER loop (better cache reuse)
    int num_kv_blocks = (kv_seq_len + kv_block_size - 1) / kv_block_size;

    for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++kv_block_idx) {
        int kv_start = kv_block_idx * kv_block_size;
        int kv_end = std::min(kv_start + kv_block_size, kv_seq_len);

        // Pre-load this KV block (will be reused by multiple queries)
        const float* K_block = K_all + kv_start * head_dim;
        const float* V_block = V_all + kv_start * head_dim;

        // For each query in this Q block
        for (int q_local = 0; q_local < q_block_size; ++q_local) {
            int q_global = q_block_start + q_local;

            // Check if this KV block is relevant (causal constraint)
            if (kv_start >= q_global + 1) {
                continue;  // This KV block is in the future for this query
            }

            // Compute effective KV range for this query (causal mask)
            int effective_kv_start = kv_start;
            int effective_kv_end = std::min(kv_end, q_global + 1);

            if (effective_kv_start >= effective_kv_end) {
                continue;  // No valid positions in this block for this query
            }

            int effective_size = effective_kv_end - effective_kv_start;

            // Compute scores for this query against the effective KV range
            std::vector<float> scores(effective_size);
            const float* q_vec = Q_block + q_local * head_dim;

            // AVX2-optimized dot product (reuses cached K_block)
            for (int kv_local = 0; kv_local < effective_size; ++kv_local) {
                int kv_global = effective_kv_start + kv_local;
                const float* k_vec = K_all + kv_global * head_dim;

                // AVX2 dot product
                float sum = 0.0f;
                int d = 0;

                #ifdef __AVX2__
                __m256 v_sum0 = _mm256_setzero_ps();
                __m256 v_sum1 = _mm256_setzero_ps();

                // Process 16 elements at a time
                for (; d + 15 < head_dim; d += 16) {
                    __m256 v_q0 = _mm256_loadu_ps(q_vec + d);
                    __m256 v_k0 = _mm256_loadu_ps(k_vec + d);
                    v_sum0 = _mm256_fmadd_ps(v_q0, v_k0, v_sum0);

                    __m256 v_q1 = _mm256_loadu_ps(q_vec + d + 8);
                    __m256 v_k1 = _mm256_loadu_ps(k_vec + d + 8);
                    v_sum1 = _mm256_fmadd_ps(v_q1, v_k1, v_sum1);
                }

                // Process 8 elements at a time
                for (; d + 7 < head_dim; d += 8) {
                    __m256 v_q = _mm256_loadu_ps(q_vec + d);
                    __m256 v_k = _mm256_loadu_ps(k_vec + d);
                    v_sum0 = _mm256_fmadd_ps(v_q, v_k, v_sum0);
                }

                // Horizontal sum
                v_sum0 = _mm256_add_ps(v_sum0, v_sum1);
                v_sum0 = _mm256_hadd_ps(v_sum0, v_sum0);
                v_sum0 = _mm256_hadd_ps(v_sum0, v_sum0);

                float temp[8];
                _mm256_storeu_ps(temp, v_sum0);
                sum = temp[0] + temp[4];
                #endif

                // Handle remaining elements
                for (; d < head_dim; ++d) {
                    sum += q_vec[d] * k_vec[d];
                }

                scores[kv_local] = sum * scale;
            }

            // Get corresponding V values (reuses cached V_block)
            const float* V_effective = V_all + effective_kv_start * head_dim;

            // Update online softmax for this query (AVX2 version)
            process_streaming_block_avx2(
                scores.data(),
                V_effective,
                states[q_local],
                outputs[q_local].data(),
                effective_size,
                head_dim
            );
        }
        // K_block and V_block stay in cache for next query (better reuse!)
    }

    // Copy outputs to output block
    for (int q_local = 0; q_local < q_block_size; ++q_local) {
        std::memcpy(output_block + q_local * head_dim,
                   outputs[q_local].data(),
                   head_dim * sizeof(float));
    }
}

#endif

Tensor self_attention_streaming_blockwise(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    float scale,
    int q_block_size,
    int kv_block_size
) {
    // query: [batch, num_heads, q_seq_len, head_dim]
    // key:   [batch, num_heads, kv_seq_len, head_dim]
    // value: [batch, num_heads, kv_seq_len, head_dim]
    const Shape& q_shape = query.shape();
    size_t batch = q_shape[0];
    size_t num_heads = q_shape[1];
    size_t q_seq_len = q_shape[2];
    size_t head_dim = q_shape[3];

    const Shape& k_shape = key.shape();
    size_t kv_seq_len = k_shape[2];

    // Output: [batch, num_heads, q_seq_len, head_dim]
    size_t output_size = batch * num_heads * q_seq_len * head_dim;
    std::vector<float> output_data(output_size);

    // Calculate number of Q blocks
    int num_q_blocks = (q_seq_len + q_block_size - 1) / q_block_size;

    #pragma omp parallel for if(batch * num_heads > 1)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            // Get base pointers for this batch and head
            size_t base_offset = (b * num_heads + h) * q_seq_len * head_dim;
            const float* Q = query.data() + base_offset;

            size_t kv_base_offset = (b * num_heads + h) * kv_seq_len * head_dim;
            const float* K = key.data() + kv_base_offset;
            const float* V = value.data() + kv_base_offset;

            float* output = output_data.data() + base_offset;

            // Process Q blocks
            for (int q_block_idx = 0; q_block_idx < num_q_blocks; ++q_block_idx) {
                int q_start = q_block_idx * q_block_size;
                int q_end = std::min(q_start + q_block_size, static_cast<int>(q_seq_len));
                int current_q_size = q_end - q_start;

                const float* Q_block = Q + q_start * head_dim;
                float* output_block = output + q_start * head_dim;

                // Process this Q block with causal constraint
                process_q_block_causal(
                    Q_block,
                    K,
                    V,
                    output_block,
                    q_start,
                    current_q_size,
                    kv_seq_len,
                    head_dim,
                    kv_block_size,
                    scale
                );
            }
        }
    }

    return Tensor(std::move(output_data), query.shape());
}

#ifdef HAS_AVX2_OPS
Tensor self_attention_streaming_blockwise_avx2(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    float scale,
    int q_block_size,
    int kv_block_size
) {
    // query: [batch, num_heads, q_seq_len, head_dim]
    // key:   [batch, num_heads, kv_seq_len, head_dim]
    // value: [batch, num_heads, kv_seq_len, head_dim]
    const Shape& q_shape = query.shape();
    size_t batch = q_shape[0];
    size_t num_heads = q_shape[1];
    size_t q_seq_len = q_shape[2];
    size_t head_dim = q_shape[3];

    const Shape& k_shape = key.shape();
    size_t kv_seq_len = k_shape[2];

    // Output: [batch, num_heads, q_seq_len, head_dim]
    size_t output_size = batch * num_heads * q_seq_len * head_dim;
    std::vector<float> output_data(output_size);

    // Calculate number of Q blocks
    int num_q_blocks = (q_seq_len + q_block_size - 1) / q_block_size;

    // OPTIMIZATION 1: Collapse all three loops for finer-grained parallelism
    // This creates (batch * num_heads * num_q_blocks) tasks instead of just (batch * num_heads)
    // For batch=1, heads=16, q_blocks=4: 64 tasks instead of 16
    // Better utilization of 26-52 CPU cores
    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (int q_block_idx = 0; q_block_idx < num_q_blocks; ++q_block_idx) {
                // Get base pointers for this batch and head
                size_t base_offset = (b * num_heads + h) * q_seq_len * head_dim;
                const float* Q = query.data() + base_offset;

                size_t kv_base_offset = (b * num_heads + h) * kv_seq_len * head_dim;
                const float* K = key.data() + kv_base_offset;
                const float* V = value.data() + kv_base_offset;

                float* output = output_data.data() + base_offset;

                // Process this Q block
                int q_start = q_block_idx * q_block_size;
                int q_end = std::min(q_start + q_block_size, static_cast<int>(q_seq_len));
                int current_q_size = q_end - q_start;

                const float* Q_block = Q + q_start * head_dim;
                float* output_block = output + q_start * head_dim;

                // Process this Q block with causal constraint (AVX2 version)
                process_q_block_causal_avx2(
                    Q_block,
                    K,
                    V,
                    output_block,
                    q_start,
                    current_q_size,
                    kv_seq_len,
                    head_dim,
                    kv_block_size,
                    scale
                );
            }
        }
    }

    return Tensor(std::move(output_data), query.shape());
}

/**
 * OPTIMIZATION 2: Cache-optimized version of AVX2 streaming attention
 * Uses KV blocks in outer loop for better L2/L3 cache utilization
 */
Tensor self_attention_streaming_blockwise_avx2_cache_optimized(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    float scale,
    int q_block_size,
    int kv_block_size
) {
    // query: [batch, num_heads, q_seq_len, head_dim]
    // key:   [batch, num_heads, kv_seq_len, head_dim]
    // value: [batch, num_heads, kv_seq_len, head_dim]
    const Shape& q_shape = query.shape();
    size_t batch = q_shape[0];
    size_t num_heads = q_shape[1];
    size_t q_seq_len = q_shape[2];
    size_t head_dim = q_shape[3];

    const Shape& k_shape = key.shape();
    size_t kv_seq_len = k_shape[2];

    // Output: [batch, num_heads, q_seq_len, head_dim]
    size_t output_size = batch * num_heads * q_seq_len * head_dim;
    std::vector<float> output_data(output_size);

    // Calculate number of Q blocks
    int num_q_blocks = (q_seq_len + q_block_size - 1) / q_block_size;

    // OPTIMIZATION 1: Collapse all three loops for finer-grained parallelism
    // OPTIMIZATION 2: Use cache-optimized processing function
    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (int q_block_idx = 0; q_block_idx < num_q_blocks; ++q_block_idx) {
                // Get base pointers for this batch and head
                size_t base_offset = (b * num_heads + h) * q_seq_len * head_dim;
                const float* Q = query.data() + base_offset;

                size_t kv_base_offset = (b * num_heads + h) * kv_seq_len * head_dim;
                const float* K = key.data() + kv_base_offset;
                const float* V = value.data() + kv_base_offset;

                float* output = output_data.data() + base_offset;

                // Process this Q block
                int q_start = q_block_idx * q_block_size;
                int q_end = std::min(q_start + q_block_size, static_cast<int>(q_seq_len));
                int current_q_size = q_end - q_start;

                const float* Q_block = Q + q_start * head_dim;
                float* output_block = output + q_start * head_dim;

                // Use cache-optimized version (OPTIMIZATION 2)
                process_q_block_causal_avx2_cache_optimized(
                    Q_block,
                    K,
                    V,
                    output_block,
                    q_start,
                    current_q_size,
                    kv_seq_len,
                    head_dim,
                    kv_block_size,
                    scale
                );
            }
        }
    }

    return Tensor(std::move(output_data), query.shape());
}

#endif

// ============================================================================
// MPI Functions
// ============================================================================

#ifdef MPI_VERSION

void all_reduce_sum(Tensor& tensor, MPI_Comm comm) {
    float* data = tensor.data();
    size_t count = tensor.size();

    MPI_Allreduce(MPI_IN_PLACE, data, count, MPI_FLOAT, MPI_SUM, comm);
}

void broadcast(Tensor& tensor, int root, MPI_Comm comm) {
    float* data = tensor.data();
    size_t count = tensor.size();

    MPI_Bcast(data, count, MPI_FLOAT, root, comm);
}

#endif // MPI_VERSION

} // namespace ops
} // namespace tensor_cpp

// Global block size configuration for streaming attention
namespace tensor_cpp {
    int g_q_block_size = 32;   // Default query block size
    int g_kv_block_size = 64;  // Default key/value block size
}
