/**
 * @file qwen3_ops_mpi.h
 * @brief MPI+OpenMP parallelized Qwen3 operators
 *
 * Implements distributed Qwen3 inference using MPI for inter-node parallelization
 * and OpenMP for intra-node parallelization.
 */

#ifndef TENSOR_CPP_QWEN3_OPS_MPI_H
#define TENSOR_CPP_QWEN3_OPS_MPI_H

#include "tensor_cpp/tensor.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/ops_mpi.h"
#include <vector>

#ifdef MPI_VERSION
#include <mpi.h>
#endif

namespace tensor_cpp {
namespace qwen3 {
namespace mpi {

#ifdef MPI_VERSION

/**
 * @brief Attention type for MPI computation
 */
enum class MPIAttentionType {
    STANDARD,   ///< Standard attention (materializes QK^T matrix)
    STREAMING   ///< Streaming attention (block-wise, memory efficient)
};

/**
 * @brief Qwen3 MLP layer with MPI+OpenMP parallelization
 *
 * Distributes intermediate layer computation across MPI ranks
 *
 * @param hidden_states Input [batch, seq_len, hidden_size]
 * @param gate_proj Gate projection weight [intermediate_size, hidden_size]
 * @param up_proj Up projection weight [intermediate_size, hidden_size]
 * @param down_proj Down projection weight [hidden_size, intermediate_size]
 * @param comm MPI communicator
 * @return Output [batch, seq_len, hidden_size]
 */
Tensor qwen3_mlp_mpi_omp(
    const Tensor& hidden_states,
    const Tensor& gate_proj,
    const Tensor& up_proj,
    const Tensor& down_proj,
    MPI_Comm comm = MPI_COMM_WORLD
);

/**
 * @brief Qwen3 Attention with MPI+OpenMP parallelization
 *
 * Distributes attention heads across MPI ranks for efficient parallel computation
 *
 * @param hidden_states Input [batch, seq_len, hidden_size]
 * @param num_attention_heads Total number of attention heads
 * @param num_key_value_heads Total number of KV heads
 * @param head_dim Head dimension
 * @param qkv_projs Combined QKV projection weights
 * @param o_proj Output projection weight
 * @param q_norm_weight Q normalization weight
 * @param k_norm_weight K normalization weight
 * @param cos RoPE cosine values
 * @param sin RoPE sine values
 * @param comm MPI communicator
 * @param attention_type Type of attention (STANDARD or STREAMING)
 * @return Output tensor [batch, seq_len, hidden_size]
 */
Tensor qwen3_attention_mpi_omp(
    const Tensor& hidden_states,
    size_t num_attention_heads,
    size_t num_key_value_heads,
    size_t head_dim,
    const Tensor& qkv_projs,
    const Tensor& o_proj,
    const Tensor& q_norm_weight,
    const Tensor& k_norm_weight,
    const Tensor& cos,
    const Tensor& sin,
    MPI_Comm comm = MPI_COMM_WORLD,
    MPIAttentionType attention_type = MPIAttentionType::STANDARD
);

/**
 * @brief Qwen3 decoder layer with MPI+OpenMP parallelization
 *
 * Complete decoder layer with distributed attention and MLP computation
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
Tensor qwen3_decoder_layer_mpi_omp(
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

/**
 * @brief Complete Qwen3 forward pass with MPI+OpenMP parallelization
 *
 * @param input_ids Input token IDs [batch_size, seq_len]
 * @param token_embedding Word embedding weight
 * @param layers List of decoder layer weights
 * @param norm_weight Final layer norm weight
 * @param lm_head LM head projection weight
 * @param num_layers Number of layers
 * @param num_attention_heads Number of attention heads
 * @param num_key_value_heads Number of KV heads
 * @param head_dim Head dimension
 * @param rms_norm_eps RMS norm epsilon
 * @param comm MPI communicator
 * @param attention_type Attention type (STANDARD or STREAMING)
 * @return Hidden states [batch_size, seq_len, hidden_size]
 */
Tensor qwen3_forward_mpi_omp(
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
    MPI_Comm comm = MPI_COMM_WORLD,
    MPIAttentionType attention_type = MPIAttentionType::STANDARD
);

/**
 * @brief MPI+OpenMP decoder layer with KV cache support
 */
Tensor qwen3_decoder_layer_mpi_omp_with_cache(
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
    MPI_Comm comm = MPI_COMM_WORLD,
    MPIAttentionType attention_type = MPIAttentionType::STANDARD
);

/**
 * @brief MPI+OpenMP forward pass with KV cache support
 */
Tensor qwen3_forward_mpi_omp_with_cache(
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
    MPI_Comm comm = MPI_COMM_WORLD,
    MPIAttentionType attention_type = MPIAttentionType::STANDARD
);

#endif // MPI_VERSION

} // namespace mpi
} // namespace qwen3
} // namespace tensor_cpp

#endif // TENSOR_CPP_QWEN3_OPS_MPI_H
