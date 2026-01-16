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
 * @brief MPI+OpenMP parallelized streaming self-attention
 *
 * @deprecated Use attention_headwise_online_softmax() instead.
 * This function is kept for backward compatibility.
 *
 * Strategy: Distribute attention heads across MPI ranks (same as standard)
 * - Each rank computes streaming attention for its subset of heads
 * - Uses block-wise streaming attention to avoid materializing QK^T
 * - Results are combined using allgather
 *
 * @param query Query tensor [batch, num_heads, q_seq_len, head_dim]
 * @param key Key tensor [batch, num_kv_heads, k_seq_len, head_dim]
 * @param value Value tensor [batch, num_kv_heads, k_seq_len, head_dim]
 * @param mask Optional attention mask (not used in streaming)
 * @param scale Scaling factor
 * @param num_attention_heads Total number of attention heads
 * @param num_key_value_heads Total number of KV heads
 * @param comm MPI communicator
 * @return Attention output [batch, num_heads, q_seq_len, head_dim]
 */
Tensor self_attention_mpi_streaming_omp(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor* mask,
    float scale,
    int num_attention_heads,
    int num_key_value_heads,
    MPI_Comm comm = MPI_COMM_WORLD
);

// ============================================================================
// Head-wise Parallelism (按注意力头并行)
// ============================================================================

/**
 * @brief Head-wise parallel attention with standard algorithm
 *
 * 并行策略：按注意力头（Head维度）分配给不同MPI进程
 * - 每个rank负责计算 num_heads / size 个注意力头
 * - 每个rank独立计算，最后AllGather合并结果
 * - 通信量：O(batch × seq_len × d_model)
 *
 * 注意力算法：Standard Attention
 * - 显式计算 QK^T 矩阵 [seq_len, seq_len]
 * - 内存复杂度：O(seq_len²)
 *
 * @param query Query tensor [batch, num_heads, q_seq_len, head_dim]
 * @param key Key tensor [batch, num_kv_heads, k_seq_len, head_dim]
 * @param value Value tensor [batch, num_kv_heads, k_seq_len, head_dim]
 * @param mask Optional attention mask
 * @param scale Scaling factor (typically 1.0/sqrt(head_dim))
 * @param num_attention_heads Total number of attention heads
 * @param num_key_value_heads Total number of KV heads
 * @param comm MPI communicator
 * @return Attention output [batch, num_heads, q_seq_len, head_dim]
 */
Tensor attention_headwise_standard(
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
 * @brief Head-wise parallel attention with online softmax algorithm
 *
 * 并行策略：按注意力头（Head维度）分配给不同MPI进程
 * - 每个rank负责计算 num_heads / size 个注意力头
 * - 每个rank独立计算，最后AllGather合并结果
 * - 通信量：O(batch × seq_len × d_model)
 *
 * 注意力算法：Online Softmax (Streaming Attention)
 * - Block-wise处理，使用online softmax避免materialize完整QK^T
 * - 内存复杂度：O(seq_len) per head
 * - 适合长序列，cache-friendly
 *
 * @param query Query tensor [batch, num_heads, q_seq_len, head_dim]
 * @param key Key tensor [batch, num_kv_heads, k_seq_len, head_dim]
 * @param value Value tensor [batch, num_kv_heads, k_seq_len, head_dim]
 * @param mask Optional attention mask (not used in online softmax)
 * @param scale Scaling factor
 * @param num_attention_heads Total number of attention heads
 * @param num_key_value_heads Total number of KV heads
 * @param comm MPI communicator
 * @return Attention output [batch, num_heads, q_seq_len, head_dim]
 */
Tensor attention_headwise_online_softmax(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor* mask,
    float scale,
    int num_attention_heads,
    int num_key_value_heads,
    MPI_Comm comm = MPI_COMM_WORLD
);

// ============================================================================
// Sequence Parallelism (按序列维度并行)
// ============================================================================

/**
 * @brief Sequence-wise parallel attention with online softmax algorithm
 *
 * 并行策略：按序列长度（Sequence维度）分配给不同MPI进程
 * - 序列长度 L，MPI进程数 P
 * - Rank i 负责token block: [i * L/P, (i+1) * L/P)
 * - 每个rank只存储并访问本地token block的K/V
 *
 * 注意力算法：Online Softmax (Distributed)
 * - Step 1: 本地streaming计算
 *   * 维护 local_max, local_exp_sum, local_weighted_value
 *   * 禁止显式materialize QK^T或score matrix
 *
 * - Step 2: 跨rank归约 (MPI_Allreduce)
 *   * 归约 global_max, global_exp_sum
 *   * 通信对象：标量或小向量 O(d_head)
 *   * 禁止传输K/V、score、完整attention output
 *
 * - Step 3: 本地归一化修正
 *   * 使用global_max / global_exp_sum修正本地输出
 *
 * 通信复杂度：
 * - O(batch × d_head × P) per layer
 * - 相比head-wise的 O(batch × seq_len × d_model) 显著降低
 *
 * @param query Query tensor [batch, num_heads, q_seq_len, head_dim]
 *              q_seq_len should be local sequence length for this rank
 * @param key Key tensor [batch, num_kv_heads, k_seq_len, head_dim]
 *            k_seq_len should be local sequence length for this rank
 * @param value Value tensor [batch, num_kv_heads, v_seq_len, head_dim]
 *              v_seq_len should be local sequence length for this rank
 * @param mask Optional attention mask
 * @param scale Scaling factor
 * @param num_attention_heads Total number of attention heads
 * @param num_key_value_heads Total number of KV heads
 * @param global_seq_len Global sequence length (across all ranks)
 * @param comm MPI communicator
 * @return Attention output [batch, num_heads, local_q_seq_len, head_dim]
 */
Tensor attention_sequence_online_softmax(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor* mask,
    float scale,
    int num_attention_heads,
    int num_key_value_heads,
    size_t global_seq_len,
    MPI_Comm comm = MPI_COMM_WORLD
);

// AVX2-optimized version of sequence parallel attention
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #ifdef __AVX2__
Tensor attention_sequence_online_softmax_avx2(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor* mask,
    float scale,
    int num_attention_heads,
    int num_key_value_heads,
    size_t global_seq_len,
    MPI_Comm comm = MPI_COMM_WORLD
);
    #endif
#endif

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

// AVX2-optimized linear layer
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #ifdef __AVX2__
/**
 * @brief MPI+OpenMP parallelized linear layer with AVX2 optimization
 *
 * Same communication pattern as linear_mpi_omp, but uses AVX2 SIMD for matrix multiplication.
 * Distributes output features across ranks and Allgathers results.
 *
 * @param input Input tensor [seq_len, in_features]
 * @param weight Weight tensor [out_features, in_features]
 * @param bias Optional bias [out_features]
 * @param comm MPI communicator
 * @return Output tensor [seq_len, out_features]
 */
Tensor linear_mpi_omp_avx2(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias = nullptr,
    MPI_Comm comm = MPI_COMM_WORLD
);
    #endif
#endif

/**
 * @brief MPI+OpenMP parallelized linear layer WITHOUT Allgather (for sequence parallelism)
 *
 * Unlike linear_mpi_omp, this keeps the output distributed across ranks.
 * Each rank computes only its local portion of output features.
 * Used in sequence parallelism to maintain sequence dimension distribution.
 *
 * @param input Input tensor [seq_len, in_features]
 * @param weight Weight tensor [out_features, in_features]
 * @param bias Optional bias [out_features]
 * @param comm MPI communicator
 * @return Output tensor [seq_len, local_out_features] - DISTRIBUTED
 */
Tensor linear_mpi_omp_no_allgather(
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

/**
 * @brief Embedding layer WITHOUT Allgather (for true sequence parallelism)
 *
 * Distributes sequence positions across ranks and KEEPS them distributed.
 * Each rank computes embedding for its local sequence positions only.
 *
 * @param indices Token indices [batch, seq_len]
 * @param weight Embedding matrix [vocab_size, hidden_size]
 * @param padding_idx Padding index
 * @param comm MPI communicator
 * @return Embedded tensor [batch, local_seq_len, hidden_size] - DISTRIBUTED
 */
Tensor embedding_mpi_omp_no_allgather(
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
