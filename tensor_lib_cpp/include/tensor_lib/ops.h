/**
 * @file ops.h
 * @brief Common deep learning operators with OpenMP/MPI parallel support
 */

#ifndef TENSOR_LIB_OPS_H
#define TENSOR_LIB_OPS_H

#include "tensor.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <complex>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef MPI_VERSION
#include <mpi.h>
#endif

namespace tensor_lib {
namespace ops {

// ============================================================================
// Element-wise Addition (with broadcasting)
// ============================================================================

template <typename T>
Tensor<T> add(const Tensor<T>& input, const Tensor<T>& other, T alpha = 1.0) {
    // Simple implementation: same shape only
    if (input.shape() != other.shape()) {
        throw std::invalid_argument("Shape mismatch for addition");
    }

    std::vector<T> result(input.size());

    #pragma omp parallel for if(input.size() > 1000)
    for (size_t i = 0; i < input.size(); ++i) {
        result[i] = input[i] + alpha * other[i];
    }

    return Tensor<T>(std::move(result), input.shape());
}

// ============================================================================
// Argmax
// ============================================================================

template <typename T>
Tensor<long> argmax(const Tensor<T>& input, int dim = -1, bool keepdim = false) {
    if (dim == -1) {
        // Global argmax
        size_t max_idx = 0;
        T max_val = input[0];

        // Find maximum without OpenMP for correctness first
        for (size_t i = 1; i < input.size(); ++i) {
            if (input[i] > max_val) {
                max_val = input[i];
                max_idx = i;
            }
        }

        // Create a 1-element tensor with the result
        std::vector<long> result_data = {static_cast<long>(max_idx)};
        return Tensor<long>(result_data, Shape({1}));
    } else {
        // Argmax along dimension
        throw std::runtime_error("Argmax along specific dimension not yet implemented");
    }
}

// ============================================================================
// Embedding Lookup
// ============================================================================

template <typename T>
Tensor<T> embedding(const Tensor<long>& indices, const Tensor<T>& weight,
                    long padding_idx = -1) {
    // indices: (batch_size, seq_len) or (seq_len,)
    // weight: (num_embeddings, embedding_dim)
    // Output: (*indices.shape(), embedding_dim)

    size_t num_embeddings = weight.shape()[0];
    size_t embedding_dim = weight.shape()[1];

    // Flatten indices for easy indexing
    size_t num_indices = indices.size();

    std::vector<T> result(num_indices * embedding_dim);

    #pragma omp parallel for if(num_indices * embedding_dim > 1000)
    for (size_t i = 0; i < num_indices; ++i) {
        long idx = indices[i];

        // Handle out-of-range indices
        if (idx < 0 || idx >= static_cast<long>(num_embeddings)) {
            if (padding_idx >= 0 && idx == padding_idx) {
                // Use zeros for padding indices
                for (size_t j = 0; j < embedding_dim; ++j) {
                    result[i * embedding_dim + j] = static_cast<T>(0);
                }
            } else {
                throw std::out_of_range("Embedding index out of range");
            }
        } else {
            // Lookup embedding
            for (size_t j = 0; j < embedding_dim; ++j) {
                result[i * embedding_dim + j] = weight[idx * embedding_dim + j];
            }
        }
    }

    // Determine output shape
    Shape out_shape = indices.shape();
    out_shape.dims.push_back(embedding_dim);

    return Tensor<T>(std::move(result), out_shape);
}

// ============================================================================
// Linear (Fully Connected) Layer: y = xA^T + b
// ============================================================================

template <typename T>
Tensor<T> linear(const Tensor<T>& input, const Tensor<T>& weight,
                const Tensor<T>* bias = nullptr) {
    // input: (*, in_features)
    // weight: (out_features, in_features)
    // bias: (out_features,)
    // output: (*, out_features)

    if (input.shape().ndim() != 2) {
        throw std::invalid_argument("Linear only supports 2D input for now");
    }

    size_t batch_size = input.shape()[0];
    size_t in_features = input.shape()[1];
    size_t out_features = weight.shape()[0];

    if (weight.shape()[1] != in_features) {
        throw std::invalid_argument("Weight shape mismatch in linear layer");
    }

    // Output = input @ weight.T
    // Shape: (batch_size, out_features)

    std::vector<T> output(batch_size * out_features, static_cast<T>(0));

    #pragma omp parallel for if(batch_size * out_features * in_features > 1000)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t o = 0; o < out_features; ++o) {
            T sum = static_cast<T>(0);
            for (size_t i = 0; i < in_features; ++i) {
                sum += input[b * in_features + i] * weight[o * in_features + i];
            }
            output[b * out_features + o] = sum;
        }
    }

    // Add bias if provided
    if (bias != nullptr) {
        #pragma omp parallel for if(batch_size * out_features > 1000)
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t o = 0; o < out_features; ++o) {
                output[b * out_features + o] += (*bias)[o];
            }
        }
    }

    return Tensor<T>(std::move(output), Shape({batch_size, out_features}));
}

// ============================================================================
// RMS Normalization
// ============================================================================

template <typename T>
Tensor<T> rms_norm(const Tensor<T>& input, const Tensor<T>* weight = nullptr,
                   T eps = 1e-8, int dim = -1) {
    // RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
    // Normalize along last dimension by default

    if (input.shape().ndim() != 2) {
        throw std::invalid_argument("RMS norm only supports 2D input for now");
    }

    size_t batch_size = input.shape()[0];
    size_t hidden_size = input.shape()[1];

    std::vector<T> output(input.size());

    #pragma omp parallel for if(batch_size * hidden_size > 1000)
    for (size_t b = 0; b < batch_size; ++b) {
        // Compute RMS for this sample
        T mean_square = static_cast<T>(0);
        for (size_t i = 0; i < hidden_size; ++i) {
            T val = input[b * hidden_size + i];
            mean_square += val * val;
        }
        mean_square /= static_cast<T>(hidden_size);
        T rms = std::sqrt(mean_square + eps);

        // Normalize
        for (size_t i = 0; i < hidden_size; ++i) {
            output[b * hidden_size + i] = input[b * hidden_size + i] / rms;
        }

        // Apply gain (gamma) if provided
        if (weight != nullptr) {
            for (size_t i = 0; i < hidden_size; ++i) {
                output[b * hidden_size + i] *= (*weight)[i];
            }
        }
    }

    return Tensor<T>(std::move(output), input.shape());
}

// ============================================================================
// Rotary Position Embedding (RoPE)
// ============================================================================

template <typename T>
class RotaryEmbedding {
private:
    size_t dim_;
    size_t max_seq_len_;
    T theta_;
    std::vector<std::complex<T>> freqs_cis_;

    void precompute_freqs_cis() {
        // Compute inverse frequencies
        std::vector<T> freqs(dim_ / 2);
        for (size_t i = 0; i < dim_ / 2; ++i) {
            freqs[i] = static_cast<T>(1.0) / std::pow(theta_, static_cast<T>(i) / static_cast<T>(dim_));
        }

        // Generate sequence positions and compute freqs_cis
        freqs_cis_.resize(max_seq_len_ * (dim_ / 2));

        #pragma omp parallel for if(max_seq_len_ * dim_ / 2 > 1000)
        for (size_t t = 0; t < max_seq_len_; ++t) {
            for (size_t i = 0; i < dim_ / 2; ++i) {
                T angle = static_cast<T>(t) * freqs[i];
                freqs_cis_[t * (dim_ / 2) + i] = std::complex<T>(std::cos(angle), std::sin(angle));
            }
        }
    }

public:
    RotaryEmbedding(size_t dim, size_t max_seq_len = 2048, T theta = 10000.0)
        : dim_(dim), max_seq_len_(max_seq_len), theta_(theta) {
        if (dim % 2 != 0) {
            throw std::invalid_argument("Rotary embedding dimension must be even");
        }
        precompute_freqs_cis();
    }

    // Apply rotary embedding to a tensor
    // Input shape: (batch_size, seq_len, num_heads, head_dim)
    Tensor<T> apply(const Tensor<T>& input, size_t seq_len) {
        if (input.shape().ndim() != 4) {
            throw std::invalid_argument("RoPE requires 4D input: (batch, seq, heads, head_dim)");
        }

        size_t batch_size = input.shape()[0];
        size_t num_heads = input.shape()[2];
        size_t head_dim = input.shape()[3];

        if (head_dim != dim_) {
            throw std::invalid_argument("Head dimension must match rotary embedding dimension");
        }

        std::vector<T> output(input.size());

        #pragma omp parallel for if(batch_size * seq_len * num_heads * head_dim > 1000)
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                for (size_t h = 0; h < num_heads; ++h) {
                    // Get precomputed freq for this position
                    const std::complex<T>* cis = &freqs_cis_[s * (dim_ / 2)];

                    for (size_t i = 0; i < dim_ / 2; ++i) {
                        size_t base_idx = ((b * input.shape()[1] + s) * num_heads + h) * head_dim;

                        // Split into real and imaginary parts
                        T x_real = input[base_idx + i];
                        T x_imag = input[base_idx + i + dim_ / 2];

                        // Apply rotation
                        std::complex<T> x_complex(x_real, x_imag);
                        std::complex<T> rotated = x_complex * cis[i];

                        output[base_idx + i] = rotated.real();
                        output[base_idx + i + dim_ / 2] = rotated.imag();
                    }
                }
            }
        }

        return Tensor<T>(std::move(output), input.shape());
    }
};

// ============================================================================
// SwiGLU Activation Function
// ============================================================================

template <typename T>
Tensor<T> swiglu(const Tensor<T>& x, const Tensor<T>& gate) {
    // SwiGLU(x, gate) = Swish(x) * gate = (x * sigmoid(x)) * gate
    // Simplified: x * sigmoid(gate) * gate
    // More commonly: SiLU(gate) * x where SiLU(x) = x * sigmoid(x)

    if (x.shape() != gate.shape()) {
        throw std::invalid_argument("Shape mismatch for SwiGLU");
    }

    std::vector<T> result(x.size());

    #pragma omp parallel for if(x.size() > 1000)
    for (size_t i = 0; i < x.size(); ++i) {
        // SiLU(gate) = gate * sigmoid(gate)
        T sigmoid_gate = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-gate[i]));
        T silu = gate[i] * sigmoid_gate;
        result[i] = x[i] * silu;
    }

    return Tensor<T>(std::move(result), x.shape());
}

// ============================================================================
// Self Attention
// ============================================================================

template <typename T>
Tensor<T> self_attention(
    const Tensor<T>& query,      // (batch_size, num_heads, seq_len, head_dim)
    const Tensor<T>& key,        // (batch_size, num_heads, seq_len, head_dim)
    const Tensor<T>& value,      // (batch_size, num_heads, seq_len, head_dim)
    const Tensor<T>* mask = nullptr,  // Optional: (seq_len, seq_len) or (batch_size, num_heads, seq_len, seq_len)
    float scale = 1.0f
) {
    // Self-attention: softmax(Q @ K^T / sqrt(d)) @ V

    size_t batch_size = query.shape()[0];
    size_t num_heads = query.shape()[1];
    size_t seq_len = query.shape()[2];
    size_t head_dim = query.shape()[3];

    // Compute attention scores: Q @ K^T
    // Output shape: (batch_size, num_heads, seq_len, seq_len)

    std::vector<T> scores(batch_size * num_heads * seq_len * seq_len);

    #pragma omp parallel for if(batch_size * num_heads * seq_len * seq_len > 1000)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j < seq_len; ++j) {
                    // Dot product of query[i] and key[j]
                    T sum = static_cast<T>(0);
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

    // Apply softmax (numerically stable: exp(x - max(x)) / sum(exp(x - max(x))))
    std::vector<T> attn_weights(batch_size * num_heads * seq_len * seq_len);

    #pragma omp parallel for if(batch_size * num_heads * seq_len > 100)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < seq_len; ++i) {
                // Find max for this row
                T max_score = -std::numeric_limits<T>::infinity();
                for (size_t j = 0; j < seq_len; ++j) {
                    size_t score_idx = ((b * num_heads + h) * seq_len + i) * seq_len + j;
                    max_score = std::max(max_score, scores[score_idx]);
                }

                // Compute exp and sum
                T sum_exp = static_cast<T>(0);
                for (size_t j = 0; j < seq_len; ++j) {
                    size_t score_idx = ((b * num_heads + h) * seq_len + i) * seq_len + j;
                    T exp_val = std::exp(scores[score_idx] - max_score);
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
    // Output shape: (batch_size, num_heads, seq_len, head_dim)

    std::vector<T> output(batch_size * num_heads * seq_len * head_dim);

    #pragma omp parallel for if(batch_size * num_heads * seq_len * head_dim > 1000)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t d = 0; d < head_dim; ++d) {
                    T sum = static_cast<T>(0);
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

    return Tensor<T>(std::move(output), query.shape());
}

// ============================================================================
// MPI All-Reduce for distributed training
// ============================================================================

#ifdef MPI_VERSION

template <typename T>
void all_reduce_sum(Tensor<T>& tensor, MPI_Comm comm = MPI_COMM_WORLD) {
    T* data = tensor.data();
    size_t count = tensor.size();

    MPI_Datatype mpi_type;
    if (std::is_same<T, float>::value) {
        mpi_type = MPI_FLOAT;
    } else if (std::is_same<T, double>::value) {
        mpi_type = MPI_DOUBLE;
    } else if (std::is_same<T, int>::value) {
        mpi_type = MPI_INT;
    } else if (std::is_same<T, long>::value) {
        mpi_type = MPI_LONG;
    } else {
        throw std::runtime_error("Unsupported MPI datatype");
    }

    MPI_Allreduce(MPI_IN_PLACE, data, count, mpi_type, MPI_SUM, comm);
}

template <typename T>
void broadcast(Tensor<T>& tensor, int root, MPI_Comm comm = MPI_COMM_WORLD) {
    T* data = tensor.data();
    size_t count = tensor.size();

    MPI_Datatype mpi_type;
    if (std::is_same<T, float>::value) {
        mpi_type = MPI_FLOAT;
    } else if (std::is_same<T, double>::value) {
        mpi_type = MPI_DOUBLE;
    } else if (std::is_same<T, int>::value) {
        mpi_type = MPI_INT;
    } else if (std::is_same<T, long>::value) {
        mpi_type = MPI_LONG;
    } else {
        throw std::runtime_error("Unsupported MPI datatype");
    }

    MPI_Bcast(data, count, mpi_type, root, comm);
}

#endif // MPI_VERSION

} // namespace ops
} // namespace tensor_lib

#endif // TENSOR_LIB_OPS_H
