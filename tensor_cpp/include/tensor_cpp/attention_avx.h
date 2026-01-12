/**
 * @file attention_avx.h
 * @brief AVX2-optimized attention operators
 */

#ifndef TENSOR_CPP_ATTENTION_AVX_H
#define TENSOR_CPP_ATTENTION_AVX_H

#include "tensor_cpp/tensor.h"
#include <immintrin.h>

namespace tensor_cpp {
namespace ops {
namespace avx2 {

/**
 * @brief AVX2-optimized self-attention
 *
 * Computes attention with AVX2-accelerated dot products:
 * - Q @ K^T with AVX2
 * - Softmax
 * - Attention weights @ V
 *
 * @param query Query tensor [batch, num_heads, q_seq_len, head_dim]
 * @param key Key tensor [batch, num_heads, k_seq_len, head_dim]
 * @param value Value tensor [batch, num_heads, k_seq_len, head_dim]
 * @param mask Optional attention mask [q_seq_len, k_seq_len] or [1, 1, q_seq_len, k_seq_len]
 * @param scale Scaling factor (typically 1/sqrt(head_dim))
 * @return Output tensor [batch, num_heads, q_seq_len, head_dim]
 */
Tensor self_attention_avx2(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor* mask,
    float scale
);

/**
 * @brief AVX2-optimized linear layer (for QKV projections)
 *
 * @param input Input tensor [..., in_features]
 * @param weight Weight matrix [out_features, in_features]
 * @param bias Optional bias [out_features]
 * @return Output tensor [..., out_features]
 */
Tensor linear_avx2(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias = nullptr
);

} // namespace avx2
} // namespace ops
} // namespace tensor_cpp

#endif // TENSOR_CPP_ATTENTION_AVX_H
