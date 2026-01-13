/**
 * @file qwen3_ops_mpi_avx.h
 * @brief MPI+AVX2 hybrid Qwen3 operators
 *
 * Implements combined MPI and AVX2 SIMD optimization for Qwen3
 */

#ifndef TENSOR_CPP_QWEN3_OPS_MPI_AVX_H
#define TENSOR_CPP_QWEN3_OPS_MPI_AVX_H

#include "tensor_cpp/tensor.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/ops_avx.h"
#include <vector>

#ifdef MPI_VERSION
#include <mpi.h>
#endif

namespace tensor_cpp {
namespace qwen3 {
namespace mpi_avx {

#ifdef MPI_VERSION

// ============================================================================
// Qwen3 MLP (SwiGLU) with MPI+AVX2
// ============================================================================

/**
 * @brief Qwen3 MLP layer with MPI+AVX2 optimization
 *
 * Combined MPI distributed computation and AVX2 SIMD for SwiGLU MLP
 *
 * @param hidden_states Input [batch, seq_len, hidden_size]
 * @param gate_proj Gate projection weight [intermediate_size, hidden_size]
 * @param up_proj Up projection weight [intermediate_size, hidden_size]
 * @param down_proj Down projection weight [hidden_size, intermediate_size]
 * @param comm MPI communicator
 * @return Output [batch, seq_len, hidden_size]
 */
Tensor qwen3_mlp_mpi_avx(
    const Tensor& hidden_states,
    const Tensor& gate_proj,
    const Tensor& up_proj,
    const Tensor& down_proj,
    MPI_Comm comm = MPI_COMM_WORLD
);

// ============================================================================
// Qwen3 Decoder Layer with MPI+AVX2
// ============================================================================

/**
 * @brief Qwen3 decoder layer with MPI+AVX2 optimization
 *
 * Complete decoder layer with distributed attention and AVX2-optimized MLP
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
 * @param comm MPI communicator
 * @return Output tensor [batch, seq_len, hidden_size]
 */
Tensor qwen3_decoder_layer_mpi_avx(
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
    const Tensor& sin,
    MPI_Comm comm = MPI_COMM_WORLD
);

// ============================================================================
// Qwen3 Model (Full Forward Pass) with MPI+AVX2
// ============================================================================

/**
 * @brief Complete Qwen3 model forward pass with MPI+AVX2 optimization
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
 * @param comm MPI communicator
 * @return Hidden states [batch_size, seq_len, hidden_size]
 */
Tensor qwen3_forward_mpi_avx(
    const TensorL& input_ids,
    const Tensor& token_embedding,
    const std::vector<Qwen3LayerWeights>& layers,
    const Tensor& norm_weight,
    const Tensor& lm_head,
    size_t num_layers,
    size_t num_attention_heads,
    size_t num_key_value_heads,
    size_t head_dim,
    float rms_norm_eps,
    MPI_Comm comm = MPI_COMM_WORLD
);

/**
 * @brief MPI+AVX2 decoder layer with KV cache support
 */
Tensor qwen3_decoder_layer_mpi_avx_with_cache(
    const Tensor& hidden_states,
    KVCache* kv_cache,
    size_t layer_idx,
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
    const Tensor& sin,
    MPI_Comm comm = MPI_COMM_WORLD
);

/**
 * @brief MPI+AVX2 forward pass with KV cache support
 */
Tensor qwen3_forward_mpi_avx_with_cache(
    const TensorL& input_ids,
    KVCache* kv_cache,
    const Tensor& token_embedding,
    const std::vector<Qwen3LayerWeights>& layers,
    const Tensor& norm_weight,
    const Tensor& lm_head,
    size_t num_layers,
    size_t num_attention_heads,
    size_t num_key_value_heads,
    size_t head_dim,
    float rms_norm_eps,
    MPI_Comm comm = MPI_COMM_WORLD
);

#endif // MPI_VERSION

} // namespace mpi_avx
} // namespace qwen3
} // namespace tensor_cpp

#endif // TENSOR_CPP_QWEN3_OPS_MPI_AVX_H
