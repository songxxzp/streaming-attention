/**
 * @file qwen3_ops_avx.h
 * @brief AVX2-optimized Qwen3 operators with pre-extracted QKV projections
 */

#ifndef TENSOR_CPP_QWEN3_OPS_AVX_H
#define TENSOR_CPP_QWEN3_OPS_AVX_H

#include "tensor_cpp/tensor.h"
#include "tensor_cpp/qwen3_ops.h"
#include <string>
#include <vector>

namespace tensor_cpp {
namespace qwen3 {
namespace avx2 {

/**
 * @brief AVX2-optimized Qwen3 attention with pre-extracted QKV projections
 *
 * Uses pre-extracted q_proj, k_proj, v_proj to avoid repeated matrix copying.
 * All dot products use AVX2 instructions.
 */
Tensor qwen3_attention_avx(
    const Tensor& hidden_states,
    size_t num_attention_heads,
    size_t num_key_value_heads,
    size_t head_dim,
    const Tensor& q_proj,
    const Tensor& k_proj,
    const Tensor& v_proj,
    const Tensor& o_proj,
    const Tensor& q_norm_weight,
    const Tensor& k_norm_weight,
    const Tensor& cos,
    const Tensor& sin
);

/**
 * @brief AVX2-optimized Qwen3 decoder layer with pre-extracted QKV
 */
Tensor qwen3_decoder_layer_avx(
    const Tensor& hidden_states,
    size_t num_attention_heads,
    size_t num_key_value_heads,
    size_t head_dim,
    float rms_norm_eps,
    const Tensor& input_layernorm_weight,
    const Tensor& q_proj,
    const Tensor& k_proj,
    const Tensor& v_proj,
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

/**
 * @brief Complete AVX2-optimized Qwen3 forward pass with pre-extracted QKV
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

/**
 * @brief AVX2-optimized MLP (SwiGLU)
 */
Tensor qwen3_mlp_avx(
    const Tensor& hidden_states,
    const Tensor& gate_proj,
    const Tensor& up_proj,
    const Tensor& down_proj
);

/**
 * @brief AVX2-optimized Qwen3 decoder layer with KV cache support
 */
Tensor qwen3_decoder_layer_avx_with_cache(
    const Tensor& hidden_states,
    KVCache* kv_cache,
    size_t layer_idx,
    size_t num_attention_heads,
    size_t num_key_value_heads,
    size_t head_dim,
    float rms_norm_eps,
    const Tensor& input_layernorm_weight,
    const Tensor& q_proj,
    const Tensor& k_proj,
    const Tensor& v_proj,
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

/**
 * @brief AVX2-optimized Qwen3 forward pass with KV cache support
 */
Tensor qwen3_forward_avx_with_cache(
    const TensorL& input_ids,
    KVCache* kv_cache,
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

#endif // TENSOR_CPP_QWEN3_OPS_AVX_V2_H
