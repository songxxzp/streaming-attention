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
    } else {
        throw std::runtime_error("Argmax along specific dimension not yet implemented");
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
    if (input.shape().ndim() != 2) {
        throw std::invalid_argument("Linear only supports 2D input for now");
    }

    size_t batch_size = input.shape()[0];
    size_t in_features = input.shape()[1];
    size_t out_features = weight.shape()[0];

    if (weight.shape()[1] != in_features) {
        throw std::invalid_argument("Weight shape mismatch in linear layer");
    }

    std::vector<float> output(batch_size * out_features, 0.0f);

    #pragma omp parallel for if(batch_size * out_features * in_features > 1000)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t o = 0; o < out_features; ++o) {
            float sum = 0.0f;
            for (size_t i = 0; i < in_features; ++i) {
                sum += input[b * in_features + i] * weight[o * in_features + i];
            }
            output[b * out_features + o] = sum;
        }
    }

    if (bias != nullptr) {
        #pragma omp parallel for if(batch_size * out_features > 1000)
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t o = 0; o < out_features; ++o) {
                output[b * out_features + o] += (*bias)[o];
            }
        }
    }

    return Tensor(std::move(output), Shape({batch_size, out_features}));
}

// ============================================================================
// RMS Normalization
// ============================================================================

Tensor rms_norm(const Tensor& input, const Tensor* weight,
                float eps, int dim) {
    if (input.shape().ndim() != 2) {
        throw std::invalid_argument("RMS norm only supports 2D input for now");
    }

    size_t batch_size = input.shape()[0];
    size_t hidden_size = input.shape()[1];

    std::vector<float> output(input.size());

    #pragma omp parallel for if(batch_size * hidden_size > 1000)
    for (size_t b = 0; b < batch_size; ++b) {
        // Compute RMS for this sample
        float mean_square = 0.0f;
        for (size_t i = 0; i < hidden_size; ++i) {
            float val = input[b * hidden_size + i];
            mean_square += val * val;
        }
        mean_square /= static_cast<float>(hidden_size);
        float rms = std::sqrt(mean_square + eps);

        // Normalize
        for (size_t i = 0; i < hidden_size; ++i) {
            output[b * hidden_size + i] = input[b * hidden_size + i] / rms;
        }

        // Apply gain if provided
        if (weight != nullptr) {
            for (size_t i = 0; i < hidden_size; ++i) {
                output[b * hidden_size + i] *= (*weight)[i];
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
    size_t seq_len = query.shape()[2];
    size_t head_dim = query.shape()[3];

    // Compute attention scores: Q @ K^T
    std::vector<float> scores(batch_size * num_heads * seq_len * seq_len);

    #pragma omp parallel for if(batch_size * num_heads * seq_len * seq_len > 1000)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j < seq_len; ++j) {
                    float sum = 0.0f;
                    for (size_t d = 0; d < head_dim; ++d) {
                        size_t q_idx = ((b * num_heads + h) * seq_len + i) * head_dim + d;
                        size_t k_idx = ((b * num_heads + h) * seq_len + j) * head_dim + d;
                        sum += query[q_idx] * key[k_idx];
                    }

                    size_t score_idx = ((b * num_heads + h) * seq_len + i) * seq_len + j;
                    scores[score_idx] = sum * scale;
                }
            }
        }
    }

    // Apply softmax
    std::vector<float> attn_weights(batch_size * num_heads * seq_len * seq_len);

    #pragma omp parallel for if(batch_size * num_heads * seq_len > 100)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < seq_len; ++i) {
                // Find max for this row
                float max_score = -std::numeric_limits<float>::infinity();
                for (size_t j = 0; j < seq_len; ++j) {
                    size_t score_idx = ((b * num_heads + h) * seq_len + i) * seq_len + j;
                    max_score = std::max(max_score, scores[score_idx]);
                }

                // Compute exp and sum
                float sum_exp = 0.0f;
                for (size_t j = 0; j < seq_len; ++j) {
                    size_t score_idx = ((b * num_heads + h) * seq_len + i) * seq_len + j;
                    float exp_val = std::exp(scores[score_idx] - max_score);
                    attn_weights[score_idx] = exp_val;
                    sum_exp += exp_val;
                }

                // Normalize
                for (size_t j = 0; j < seq_len; ++j) {
                    size_t score_idx = ((b * num_heads + h) * seq_len + i) * seq_len + j;
                    attn_weights[score_idx] /= sum_exp;
                }
            }
        }
    }

    // Compute output: attn_weights @ value
    std::vector<float> output(batch_size * num_heads * seq_len * head_dim);

    #pragma omp parallel for if(batch_size * num_heads * seq_len * head_dim > 1000)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t d = 0; d < head_dim; ++d) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < seq_len; ++j) {
                        size_t weight_idx = ((b * num_heads + h) * seq_len + i) * seq_len + j;
                        size_t value_idx = ((b * num_heads + h) * seq_len + j) * head_dim + d;
                        sum += attn_weights[weight_idx] * value[value_idx];
                    }
                    size_t out_idx = ((b * num_heads + h) * seq_len + i) * head_dim + d;
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
