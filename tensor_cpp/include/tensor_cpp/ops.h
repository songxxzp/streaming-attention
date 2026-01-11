/**
 * @file ops.h
 * @brief Common deep learning operators (non-template, float-only)
 *
 * Supports:
 * - Standard operators (add, argmax, embedding, linear, rms_norm, rope, swiglu)
 * - Standard attention (self_attention, cross_attention)
 * - Streaming attention (streaming_attention_serial, streaming_attention_omp)
 */

#ifndef TENSOR_CPP_OPS_H
#define TENSOR_CPP_OPS_H

#include "tensor_cpp/tensor.h"
#include <vector>
#include <cmath>
#include <complex>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// Integer Tensor for indices (argmax, embedding, etc.)
// ============================================================================

class TensorL {
private:
    std::vector<long> data_;
    tensor_cpp::Shape shape_;

public:
    TensorL() = default;
    TensorL(const std::vector<long>& data, const tensor_cpp::Shape& shape)
        : data_(data), shape_(shape) {
        if (data.size() != shape.total()) {
            throw std::invalid_argument("Data size does not match shape");
        }
    }

    const tensor_cpp::Shape& shape() const { return shape_; }
    size_t size() const { return data_.size(); }
    const long* data() const { return data_.data(); }
    long operator[](size_t i) const { return data_[i]; }
};

#ifdef MPI_VERSION
#include <mpi.h>
#endif

namespace tensor_cpp {
namespace ops {

// ============================================================================
// Element-wise Addition (with broadcasting)
// ============================================================================

Tensor add(const Tensor& input, const Tensor& other, float alpha = 1.0f);

// ============================================================================
// Argmax
// ============================================================================

TensorL argmax(const Tensor& input, int dim = -1, bool keepdim = false);

// ============================================================================
// Embedding Lookup
// ============================================================================

Tensor embedding(const TensorL& indices, const Tensor& weight,
                long padding_idx = -1);

// ============================================================================
// Linear (Fully Connected) Layer: y = xA^T + b
// ============================================================================

Tensor linear(const Tensor& input, const Tensor& weight,
              const Tensor* bias = nullptr);

// ============================================================================
// RMS Normalization
// ============================================================================

Tensor rms_norm(const Tensor& input, const Tensor* weight = nullptr,
                float eps = 1e-8f, int dim = -1);

// ============================================================================
// Rotary Position Embedding (RoPE)
// ============================================================================

class RotaryEmbedding {
private:
    size_t dim_;
    size_t max_seq_len_;
    float theta_;
    std::vector<std::complex<float>> freqs_cis_;

    void precompute_freqs_cis();

public:
    RotaryEmbedding(size_t dim, size_t max_seq_len = 2048, float theta = 10000.0f);
    Tensor apply(const Tensor& input, size_t seq_len);
};

// ============================================================================
// SwiGLU Activation Function
// ============================================================================

Tensor swiglu(const Tensor& x, const Tensor& gate);

// ============================================================================
// Self Attention (Standard Transformer)
// ============================================================================

/**
 * Standard self-attention: softmax(Q @ K^T / sqrt(d)) @ V
 *
 * @param query   (batch_size, num_heads, seq_len, head_dim)
 * @param key     (batch_size, num_heads, seq_len, head_dim)
 * @param value   (batch_size, num_heads, seq_len, head_dim)
 * @param mask    Optional attention mask
 * @param scale   Scaling factor (typically 1/sqrt(head_dim))
 * @return        (batch_size, num_heads, seq_len, head_dim)
 */
Tensor self_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor* mask = nullptr,
    float scale = 1.0f
);

// ============================================================================
// Cross Attention (Standard Transformer)
// ============================================================================

/**
 * Standard cross-attention: softmax(Q @ K^T / sqrt(d)) @ V
 *
 * @param query   (batch_size, num_heads, query_len, head_dim)
 * @param key     (batch_size, num_heads, kv_len, head_dim)
 * @param value   (batch_size, num_heads, kv_len, head_dim)
 * @param mask    Optional attention mask
 * @param scale   Scaling factor (typically 1/sqrt(head_dim))
 * @return        (batch_size, num_heads, query_len, head_dim)
 */
Tensor cross_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor* mask = nullptr,
    float scale = 1.0f
);

// ============================================================================
// Streaming Attention (from attention/ directory)
// ============================================================================

/**
 * Naive Attention (Serial Baseline)
 * Computes: O = softmax(Q @ K^T) @ V
 *
 * @param Q Query vector [1 x d]
 * @param K Key cache [T x d]
 * @param V Value cache [T x d]
 * @param T Sequence length
 * @param d Hidden dimension
 * @return Output vector [1 x d]
 */
std::vector<float> naive_attention_serial(
    const float* Q,
    const float* K,
    const float* V,
    int T,
    int d
);

/**
 * Streaming Block Attention (Serial)
 * Uses online softmax to compute attention in blocks
 *
 * Mathematically equivalent to naive_attention_serial
 *
 * @param Q Query vector [1 x d]
 * @param K Key cache [T x d]
 * @param V Value cache [T x d]
 * @param T Sequence length
 * @param d Hidden dimension
 * @param block_size Size of each block for streaming computation
 * @return Output vector [1 x d]
 */
std::vector<float> streaming_attention_serial(
    const float* Q,
    const float* K,
    const float* V,
    int T,
    int d,
    int block_size
);

/**
 * Streaming Block Attention (OpenMP Parallelized)
 *
 * Parallelizes across KV blocks using OpenMP.
 * Uses tree reduction to merge partial results from each thread.
 *
 * @param Q Query vector [1 x d]
 * @param K Key cache [T x d]
 * @param V Value cache [T x d]
 * @param T Sequence length
 * @param d Hidden dimension
 * @param block_size Size of each block for streaming computation
 * @param num_threads Number of OpenMP threads (0 = use OMP_NUM_THREADS)
 * @return Output vector [1 x d]
 */
std::vector<float> streaming_attention_omp(
    const float* Q,
    const float* K,
    const float* V,
    int T,
    int d,
    int block_size,
    int num_threads = 0
);

// ============================================================================
// MPI Functions
// ============================================================================

#ifdef MPI_VERSION

void all_reduce_sum(Tensor& tensor, MPI_Comm comm = MPI_COMM_WORLD);
void broadcast(Tensor& tensor, int root, MPI_Comm comm = MPI_COMM_WORLD);

#endif // MPI_VERSION

} // namespace ops
} // namespace tensor_cpp

#endif // TENSOR_CPP_OPS_H
