/**
 * @file ops_mpi.h
 * @brief MPI+OpenMP parallelized deep learning operators
 *
 * Data parallelism strategy for LLM inference:
 * - Each MPI rank holds a complete copy of model weights
 * - Input data is distributed across ranks (by batch or sequence length)
 * - Results are aggregated using MPI_Allreduce
 * - OpenMP handles intra-node parallelization
 */

#ifndef TENSOR_CPP_OPS_MPI_H
#define TENSOR_CPP_OPS_MPI_H

#include "tensor_cpp/tensor.h"
#include "tensor_cpp/ops.h"  // For TensorL
#include <vector>

#ifdef MPI_VERSION
#include <mpi.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tensor_cpp {
namespace ops {
namespace mpi {

// ============================================================================
// MPI Communication Helpers
// ============================================================================

#ifdef MPI_VERSION

/**
 * @brief All-reduce sum for tensors (in-place)
 * @param tensor Tensor to all-reduce
 * @param comm MPI communicator (default: MPI_COMM_WORLD)
 */
void all_reduce_sum(Tensor& tensor, MPI_Comm comm = MPI_COMM_WORLD);

/**
 * @brief Broadcast a tensor from root to all ranks
 * @param tensor Tensor to broadcast
 * @param root Root rank
 * @param comm MPI communicator (default: MPI_COMM_WORLD)
 */
void broadcast(Tensor& tensor, int root, MPI_Comm comm = MPI_COMM_WORLD);

/**
 * @brief Get MPI rank and size
 */
std::pair<int, int> get_mpi_info(MPI_Comm comm = MPI_COMM_WORLD);

#endif // MPI_VERSION

// ============================================================================
// MPI+OpenMP Parallelized Operators
// ============================================================================

#ifdef MPI_VERSION

/**
 * @brief MPI+OpenMP parallelized matrix multiplication
 *
 * Strategy: Distribute rows of A across MPI ranks
 * - Each rank computes: C_local = A_local @ B^T
 * - Results are combined (if needed)
 *
 * @param A Input matrix [M, K]
 * @param B Weight matrix [N, K] (transposed)
 * @param C Output matrix [M, N]
 * @param M Rows of A
 * @param N Rows of B (columns of result)
 * @param K Columns of A / rows of B
 * @param comm MPI communicator
 */
void matmul_mpi_omp(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    MPI_Comm comm = MPI_COMM_WORLD
);

/**
 * @brief MPI+OpenMP parallelized element-wise add
 *
 * @param input Input tensor
 * @param other Other tensor (must have same shape on all ranks)
 * @param alpha Scaling factor for other
 * @param comm MPI communicator
 * @return Result tensor
 */
Tensor add_mpi_omp(
    const Tensor& input,
    const Tensor& other,
    float alpha = 1.0f,
    MPI_Comm comm = MPI_COMM_WORLD
);

/**
 * @brief MPI+OpenMP parallelized RMSNorm
 *
 * @param input Input tensor [batch, seq_len, hidden_size]
 * @param weight Optional weight tensor [hidden_size]
 * @param eps Epsilon for numerical stability
 * @param comm MPI communicator
 * @return Normalized tensor
 */
Tensor rms_norm_mpi_omp(
    const Tensor& input,
    const Tensor* weight = nullptr,
    float eps = 1e-6f,
    MPI_Comm comm = MPI_COMM_WORLD
);

/**
 * @brief MPI+OpenMP parallelized Rotary Position Embedding
 *
 * @param input Input tensor [batch, num_heads, seq_len, head_dim]
 * @param cos Cosine values [seq_len, head_dim//2]
 * @param sin Sine values [seq_len, head_dim//2]
 * @param comm MPI communicator
 * @return Rotated tensor
 */
Tensor rope_mpi_omp(
    const Tensor& input,
    const Tensor& cos,
    const Tensor& sin,
    MPI_Comm comm = MPI_COMM_WORLD
);

/**
 * @brief MPI+OpenMP parallelized SwiGLU activation
 *
 * @param x Gate tensor [batch, seq_len, intermediate_size]
 * @param gate Up tensor [batch, seq_len, intermediate_size]
 * @param comm MPI communicator
 * @return Result tensor
 */
Tensor swiglu_mpi_omp(
    const Tensor& x,
    const Tensor& gate,
    MPI_Comm comm = MPI_COMM_WORLD
);

/**
 * @brief MPI+OpenMP parallelized self-attention
 *
 * Strategy: Distribute attention heads across MPI ranks
 * - Each rank computes attention for its subset of heads
 * - Results are combined using allgather
 *
 * @param query Query tensor [batch, num_heads, q_seq_len, head_dim]
 * @param key Key tensor [batch, num_kv_heads, k_seq_len, head_dim]
 * @param value Value tensor [batch, num_kv_heads, k_seq_len, head_dim]
 * @param mask Optional attention mask
 * @param scale Scaling factor
 * @param num_attention_heads Total number of attention heads
 * @param num_key_value_heads Total number of KV heads
 * @param comm MPI communicator
 * @return Attention output [batch, num_heads, q_seq_len, head_dim]
 */
Tensor self_attention_mpi_omp(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor* mask,
    float scale,
    int num_attention_heads,
    int num_key_value_heads,
    MPI_Comm comm = MPI_COMM_WORLD
);

/**
 * @brief MPI+OpenMP parallelized linear layer
 *
 * @param input Input tensor [seq_len, in_features]
 * @param weight Weight tensor [out_features, in_features]
 * @param bias Optional bias [out_features]
 * @param comm MPI communicator
 * @return Output tensor [seq_len, out_features]
 */
Tensor linear_mpi_omp(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias = nullptr,
    MPI_Comm comm = MPI_COMM_WORLD
);

/**
 * @brief MPI+OpenMP parallelized embedding lookup
 *
 * @param indices Token indices [batch, seq_len]
 * @param weight Embedding matrix [vocab_size, hidden_size]
 * @param padding_idx Padding index
 * @param comm MPI communicator
 * @return Embedded tensor [batch, seq_len, hidden_size]
 */
Tensor embedding_mpi_omp(
    const TensorL& indices,
    const Tensor& weight,
    long padding_idx = -1,
    MPI_Comm comm = MPI_COMM_WORLD
);

#endif // MPI_VERSION

} // namespace mpi
} // namespace ops
} // namespace tensor_cpp

#endif // TENSOR_CPP_OPS_MPI_H
