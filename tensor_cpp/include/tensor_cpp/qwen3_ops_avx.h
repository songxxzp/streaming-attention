/**
 * @file qwen3_ops_avx.h
 * @brief AVX2-optimized Qwen3 operators
 *
 * Implements AVX SIMD accelerated versions of Qwen3 operators
 */

#ifndef TENSOR_CPP_QWEN3_OPS_AVX_H
#define TENSOR_CPP_QWEN3_OPS_AVX_H

#include "tensor_cpp/tensor.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/ops_avx.h"
#include <vector>

namespace tensor_cpp {
namespace qwen3 {
namespace avx2 {

// ============================================================================
// Qwen3 MLP (SwiGLU) with AVX2
// ============================================================================

/**
 * @brief Qwen3 MLP layer with AVX2 optimization
 *
 * Optimized SwiGLU MLP using AVX2 SIMD instructions
 *
 * @param hidden_states Input [batch, seq_len, hidden_size]
 * @param gate_proj Gate projection weight [intermediate_size, hidden_size]
 * @param up_proj Up projection weight [intermediate_size, hidden_size]
 * @param down_proj Down projection weight [hidden_size, intermediate_size]
 * @return Output [batch, seq_len, hidden_size]
 */
Tensor qwen3_mlp_avx(
    const Tensor& hidden_states,
    const Tensor& gate_proj,
    const Tensor& up_proj,
    const Tensor& down_proj
);

// ============================================================================
// Qwen3 Decoder Layer with AVX2
// ============================================================================

/**
 * @brief Qwen3 decoder layer with AVX2 optimization
 *
 * Complete decoder layer with AVX2-optimized MLP
 *
 * @param hidden_states Input [batch, seq_len, hidden_size]
 * @param num_attention_heads Number of attention heads
 * @param num_key_value_heads Number of KV heads
 * @param head_dim Head dimension
 * @param rms_norm_eps RMS norm epsilon
 * @param input_layernorm_weight Input layer norm weight
 * @param qkv_projs Combined QKV projection weights
 * @param o_proj Output projection weight
 * @param q_norm_weight Q normalization weight
 * @param k_norm_weight K normalization weight
 * @param post_attention_layernorm_weight Post attention layer norm weight
 * @param gate_mlp MLP gate projection weight
 * @param up_mlp MLP up projection weight
 * @param down_mlp MLP down projection weight
 * @param cos RoPE cosine values
 * @param sin RoPE sine values
 * @return Output tensor [batch, seq_len, hidden_size]
 */
Tensor qwen3_decoder_layer_avx(
    const Tensor& hidden_states,
    size_t num_attention_heads,
    size_t num_key_value_heads,
    size_t head_dim,
    float rms_norm_eps,
    const Tensor& input_layernorm_weight,
    const Tensor& qkv_projs,
    const Tensor& o_proj,
    const Tensor& q_norm_weight,
    const Tensor& k_norm_weight,
    const Tensor& post_attention_layernorm_weight,
    const Tensor& gate_mlp,
    const Tensor& up_mlp,
    const Tensor& down_mlp,
    const Tensor& cos,
    const Tensor& sin
);

// ============================================================================
// Qwen3 Model (Full Forward Pass) with AVX2
// ============================================================================

/**
 * @brief Complete Qwen3 model forward pass with AVX2 optimization
 *
 * @param input_ids Input token IDs [batch_size, seq_len]
 * @param token_embedding Word embedding weight [vocab_size, hidden_size]
 * @param layers List of decoder layer weights
 * @param norm_weight Final layer norm weight
 * @param num_layers Number of layers
 * @param num_attention_heads Number of attention heads
 * @param num_key_value_heads Number of KV heads
 * @param head_dim Head dimension
 * @param rms_norm_eps RMS norm epsilon
 * @return Hidden states [batch_size, seq_len, hidden_size]
 */
Tensor qwen3_forward_avx(
    const TensorL& input_ids,
    const Tensor& token_embedding,
    const std::vector<Qwen3LayerWeights>& layers,
    const Tensor& norm_weight,
    size_t num_layers,
    size_t num_attention_heads,
    size_t num_key_value_heads,
    size_t head_dim,
    float rms_norm_eps
);

} // namespace avx2
} // namespace qwen3
} // namespace tensor_cpp

#endif // TENSOR_CPP_QWEN3_OPS_AVX_H
