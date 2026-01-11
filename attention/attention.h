#ifndef ATTENTION_H
#define ATTENTION_H

#include <vector>

// Forward declaration for MPI_Comm (needed before function declarations)
// Only provide forward declaration if MPI headers are not included
#if !defined(MPI_COMM_WORLD) && !defined(OMPI_MPI_H) && !defined(MPICH_NAME) && !defined(MPI_VERSION) && !defined(MPI_Comm)
typedef void* MPI_Comm;
#endif

// Attention computation interfaces
// All inputs use row-major ordering

/**
 * Naive Attention (Serial Baseline)
 *
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
 * Naive Attention with OpenMP Parallelization
 *
 * Same algorithm as naive_attention_serial, but parallelized using OpenMP.
 * Uses parallel for loops and reductions for matrix multiplication and softmax.
 *
 * @param Q Query vector [1 x d]
 * @param K Key cache [T x d]
 * @param V Value cache [T x d]
 * @param T Sequence length
 * @param d Hidden dimension
 * @param num_threads Number of OpenMP threads
 * @return Output vector [1 x d]
 */
std::vector<float> naive_attention_omp(
    const float* Q,
    const float* K,
    const float* V,
    int T,
    int d,
    int num_threads = 0
);

/**
 * Naive Attention with MPI + OpenMP Hybrid Parallelization
 *
 * Distributes T tokens across multiple MPI ranks.
 * Each rank internally uses OpenMP for parallelization.
 *
 * @param Q_local Query vector [1 x d] (only rank 0 needs valid data)
 * @param K_local Local portion of Key cache [T_local x d]
 * @param V_local Local portion of Value cache [T_local x d]
 * @param T_local Local sequence length (this rank's portion)
 * @param T_global Total sequence length (across all ranks)
 * @param d Hidden dimension
 * @param num_omp_threads Number of OpenMP threads per rank
 * @param comm MPI communicator
 * @return Output vector [1 x d] (only rank 0 has valid result)
 */
std::vector<float> naive_attention_mpi(
    const float* Q_local,
    const float* K_local,
    const float* V_local,
    int T_local,
    int T_global,
    int d,
    int num_omp_threads,
    MPI_Comm comm
);

/**
 * Simplified Naive Attention with MPI (no OpenMP)
 *
 * Same as naive_attention_mpi but without internal OpenMP parallelization.
 *
 * @param Q_local Query vector [1 x d] (only rank 0 needs valid data)
 * @param K_local Local portion of Key cache [T_local x d]
 * @param V_local Local portion of Value cache [T_local x d]
 * @param T_local Local sequence length (this rank's portion)
 * @param T_global Total sequence length (across all ranks)
 * @param d Hidden dimension
 * @param comm MPI communicator
 * @return Output vector [1 x d] (only rank 0 has valid result)
 */
std::vector<float> naive_attention_mpi_simple(
    const float* Q_local,
    const float* K_local,
    const float* V_local,
    int T_local,
    int T_global,
    int d,
    MPI_Comm comm
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
 * Compute L2 error between two vectors
 */
float compute_l2_error(const float* a, const float* b, int size);

/**
 * Compute max absolute error between two vectors
 */
float compute_max_error(const float* a, const float* b, int size);

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

/**
 * NUMA-aware Streaming Attention (OpenMP)
 *
 * Same as streaming_attention_omp, but documented for NUMA optimization.
 * Callers should perform first-touch initialization of K, V arrays.
 */
std::vector<float> streaming_attention_omp_numa(
    const float* Q,
    const float* K,
    const float* V,
    int T,
    int d,
    int block_size,
    int num_threads = 0
);

/**
 * Streaming Block Attention (MPI + OpenMP Hybrid)
 *
 * Distributes KV cache across multiple MPI ranks.
 * Each rank processes its local portion, then results are reduced via MPI.
 *
 * @param Q_local Query vector [1 x d] (only rank 0 needs valid data)
 * @param K_local Local portion of Key cache [T_local x d]
 * @param V_local Local portion of Value cache [T_local x d]
 * @param T_local Local sequence length for this rank
 * @param T_global Total sequence length across all ranks
 * @param d Hidden dimension
 * @param block_size Size of each block for streaming computation
 * @param comm MPI communicator (MPI_COMM_WORLD)
 * @return Output vector [1 x d] (only rank 0 has valid result)
 */
std::vector<float> streaming_attention_mpi_simple(
    const float* Q_local,
    const float* K_local,
    const float* V_local,
    int T_local,
    int T_global,
    int d,
    int block_size,
    MPI_Comm comm
);

#endif // ATTENTION_H
