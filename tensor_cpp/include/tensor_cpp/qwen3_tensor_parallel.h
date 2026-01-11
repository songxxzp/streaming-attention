/**
 * @file qwen3_tensor_parallel.h
 * @brief Qwen3 tensor parallelism with MPI
 *
 * Distributes model weights across multiple GPUs/nodes for large model inference
 */

#ifndef TENSOR_CPP_QWEN3_TENSOR_PARALLEL_H
#define TENSOR_CPP_QWEN3_TENSOR_PARALLEL_H

#include "tensor_cpp/tensor.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/qwen3_loader.h"
#include <vector>

#ifdef MPI_VERSION
#include <mpi.h>
#endif

namespace tensor_cpp {
namespace qwen3 {
namespace tensor_parallel {

#ifdef MPI_VERSION

/**
 * @brief Distribute Qwen3 weights across MPI ranks
 *
 * Strategy: Column parallelism for linear layers
 * - Each rank holds a subset of output features
 * - Allreduce to combine results
 *
 * @param weights Full model weights (only used on rank 0)
 * @param rank Current MPI rank
 * @param size Total number of MPI ranks
 * @return Distributed weights for this rank
 */
Qwen3Weights distribute_weights(
    const Qwen3Weights& weights,
    int rank,
    int size
);

/**
 * @brief Tensor parallel forward pass for Qwen3
 *
 * @param input_ids Input token IDs [batch_size, seq_len]
 * @param local_weights Local weights for this rank
 * @param num_layers Total layers
 * @param num_attention_heads Total attention heads
 * @param num_key_value_heads Total KV heads
 * @param head_dim Head dimension
 * @param rms_norm_eps RMS norm epsilon
 * @param rank Current MPI rank
 * @param size Total number of MPI ranks
 * @param comm MPI communicator
 * @return Hidden states [batch_size, seq_len, hidden_size]
 */
Tensor forward_tensor_parallel(
    const TensorL& input_ids,
    const Qwen3Weights& local_weights,
    size_t num_layers,
    size_t num_attention_heads,
    size_t num_key_value_heads,
    size_t head_dim,
    float rms_norm_eps,
    int rank,
    int size,
    MPI_Comm comm = MPI_COMM_WORLD
);

/**
 * @brief Tensor parallel linear layer
 *
 * Each rank computes a subset of output features
 *
 * @param input Input [seq_len, in_features]
 * @param weight Local weight [out_features/size, in_features]
 * @param bias Optional bias
 * @param comm MPI communicator
 * @return Output [seq_len, out_features] (allreduced across ranks)
 */
Tensor linear_tensor_parallel(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    MPI_Comm comm
);

/**
 * @brief Tensor parallel attention
 *
 * Distributes attention heads and output features across ranks
 *
 * @param hidden_states Input [batch, seq_len, hidden_size]
 * @param num_attention_heads Total attention heads
 * @param num_key_value_heads Total KV heads
 * @param head_dim Head dimension
 * @param qkv_projs Combined QKV weights
 * @param o_proj Output projection weight
 * @param q_norm_weight Q normalization weight
 * @param k_norm_weight K normalization weight
 * @param cos RoPE cosine values
 * @param sin RoPE sine values
 * @param rank Current MPI rank
 * @param size Total number of ranks
 * @param comm MPI communicator
 * @return Output [batch, seq_len, hidden_size]
 */
Tensor attention_tensor_parallel(
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
    int rank,
    int size,
    MPI_Comm comm
);

#endif // MPI_VERSION

} // namespace tensor_parallel
} // namespace qwen3
} // namespace tensor_cpp

#endif // TENSOR_CPP_QWEN3_TENSOR_PARALLEL_H
