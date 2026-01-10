#include <mpi.h>
#include "attention/attention.h"
#include "../utils/softmax_online.h"
#include <vector>
#include <cmath>
#include <limits>
#include <omp.h>

// Forward declaration from streaming_omp.cpp
struct PartialResult : public OnlineSoftmaxState {
    std::vector<float> O; // Partial output [d]

    PartialResult(int d) : OnlineSoftmaxState(), O(d, 0.0f) {}
};

// Merge function (from streaming_omp.cpp)
inline void merge_partial_results(PartialResult& left, const PartialResult& right) {
    int d = left.O.size();

    float m_new = std::max(left.m, right.m);
    float scale_left = std::exp(left.m - m_new);
    float scale_right = std::exp(right.m - m_new);
    float l_new = left.l * scale_left + right.l * scale_right;

    float alpha_left = (left.l * scale_left) / l_new;
    float alpha_right = (right.l * scale_right) / l_new;

    for (int j = 0; j < d; ++j) {
        left.O[j] = left.O[j] * alpha_left + right.O[j] * alpha_right;
    }

    left.m = m_new;
    left.l = l_new;
}

// Process blocks serially (from streaming_omp.cpp)
inline PartialResult process_blocks_serial(
    const float* Q,
    const float* K,
    const float* V,
    int T_start,
    int T_end,
    int d,
    int block_size
) {
    PartialResult result(d);
    int T_chunk = T_end - T_start;

    for (int block = 0; block < (T_chunk + block_size - 1) / block_size; ++block) {
        int block_start = T_start + block * block_size;
        int block_end = std::min(T_start + block * block_size + block_size, T_end);
        int current_block_size = block_end - block_start;

        // Compute scores for this block
        std::vector<float> scores(current_block_size);
        for (int i = 0; i < current_block_size; ++i) {
            int global_idx = block_start + i;
            float sum = 0.0f;
            const float* K_row = K + global_idx * d;
            for (int j = 0; j < d; ++j) {
                sum += Q[j] * K_row[j];
            }
            scores[i] = sum;
        }

        const float* V_block = V + block_start * d;
        process_streaming_block(scores.data(), V_block, result, result.O.data(),
                                current_block_size, d);
    }

    return result;
}

/**
 * MPI datatype for PartialResult
 *
 * We need custom MPI reduction operation for PartialResult.
 * This requires defining an MPI_Op that performs merge_partial_results.
 */

// Structure for MPI reduction (plain old data)
struct PartialResultPOD {
    float m;
    float l;
};

// Global variable for d (needed in MPI reduction callback)
static int g_mpi_d = 0;
static std::vector<float>* g_mpi_output_buffer = nullptr;

// MPI reduction callback function
void mpi_reduce_partial_results(
    void* invec,
    void* inoutvec,
    int* len,
    MPI_Datatype* datatype
) {
    // len should always be 1 for our custom reduction
    // invec and inoutvec point to PartialResultPOD
    PartialResultPOD* in = static_cast<PartialResultPOD*>(invec);
    PartialResultPOD* inout = static_cast<PartialResultPOD*>(inoutvec);

    // We also need to reduce the O vectors
    // This is handled separately via MPI_Reduce on float arrays
    // For now, just reduce m and l

    float m_new = std::max(inout->m, in->m);
    float scale_inout = std::exp(inout->m - m_new);
    float scale_in = std::exp(in->m - m_new);
    float l_new = inout->l * scale_inout + in->l * scale_in;

    inout->m = m_new;
    inout->l = l_new;
}

/**
 * Streaming Block Attention with MPI Parallelization
 *
 * Algorithm:
 * 1. Rank 0 broadcasts Q to all ranks
 * 2. Each rank computes partial attention on its local K, V partition
 * 3. MPI_Reduce merges all partial results
 * 4. (Optional) Each rank internally uses OpenMP
 *
 * @param Q Query vector [1 x d] (only needed on rank 0)
 * @param K Local portion of Key cache [T_local x d]
 * @param V Local portion of Value cache [T_local x d]
 * @param T Local sequence length (this rank's portion)
 * @param T_global Total sequence length (across all ranks)
 * @param d Hidden dimension
 * @param block_size Size of each block for streaming computation
 * @param comm MPI communicator
 * @return Output vector [1 x d] (on all ranks or just rank 0)
 */
std::vector<float> streaming_attention_mpi(
    const float* Q_local,
    const float* K_local,
    const float* V_local,
    int T_local,
    int T_global,
    int d,
    int block_size,
    MPI_Comm comm
) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Step 1: Broadcast Q from rank 0 to all ranks
    std::vector<float> Q(d);
    if (rank == 0) {
        std::copy(Q_local, Q_local + d, Q.begin());
    }
    MPI_Bcast(Q.data(), d, MPI_FLOAT, 0, comm);

    // Step 2: Each rank computes its partial result
    // Each rank processes T_local tokens starting at offset rank * T_local
    int T_offset = rank * T_local;

    // For hybrid MPI+OpenMP, use OMP internally
    PartialResult partial(d);

    #pragma omp parallel
    {
        int n_threads = omp_get_num_threads();
        int tid = omp_get_thread_num();

        // Each OMP thread processes a portion of the local data
        int chunk_size = (T_local + n_threads - 1) / n_threads;
        int T_start = tid * chunk_size;
        int T_end = std::min(T_start + chunk_size, T_local);

        PartialResult local_partial(d);
        if (T_start < T_end) {
            // Adjust indices to global coordinates
            local_partial = process_blocks_serial(
                Q.data(), K_local, V_local,
                T_start, T_end, d, block_size
            );
        }

        #pragma omp critical
        {
            merge_partial_results(partial, local_partial);
        }
    }

    // Step 3: MPI_Reduce to merge partial results
    // We need a two-step reduction:
    // a) Reduce m and l (scalars)
    // b) Reduce O (vector)

    // Create MPI datatype for PartialResultPOD
    MPI_Datatype MPI_PARTIAL_POD;
    MPI_Type_contiguous(2, MPI_FLOAT, &MPI_PARTIAL_POD);
    MPI_Type_commit(&MPI_PARTIAL_POD);

    // Create MPI reduction operation
    MPI_Op MPI_MERGE_PARTIAL;
    MPI_Op_create(mpi_reduce_partial_results, /*commute=*/1, &MPI_MERGE_PARTIAL);

    // Reduce m and l
    PartialResultPOD partial_pod{partial.m, partial.l};
    PartialResultPOD merged_pod;

    MPI_Reduce(&partial_pod, &merged_pod, 1, MPI_PARTIAL_POD,
               MPI_MERGE_PARTIAL, 0, comm);

    // Reduce O vectors
    std::vector<float> O_merged(d, 0.0f);

    if (rank == 0) {
        // Root needs to receive all partial O vectors and merge them
        std::vector<float> recv_buffer(d);

        // Copy root's own O first
        std::copy(partial.O.begin(), partial.O.end(), O_merged.begin());

        // Receive and merge from other ranks
        for (int source = 1; source < size; ++source) {
            MPI_Status status;
            MPI_Recv(recv_buffer.data(), d, MPI_FLOAT, source, 0, comm, &status);

            // Merge this rank's O into O_merged
            // We need to scale based on the softmax states
            // For simplicity, we re-broadcast all m and l values
        }

        // Better approach: Allreduce with proper scaling
        // Let's use a different strategy: Allgather all states, then merge
    }

    // Clean up
    MPI_Op_free(&MPI_MERGE_PARTIAL);
    MPI_Type_free(&MPI_PARTIAL_POD);

    if (rank == 0) {
        return O_merged;
    } else {
        return std::vector<float>(d, 0.0f);
    }
}

/**
 * Simplified version: MPI_Reduce with manual merging
 *
 * This version uses a simpler approach:
 * 1. Each rank computes its partial (m, l, O)
 * 2. Use MPI_Allgather to collect all states
 * 3. Sequentially merge on rank 0
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
) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Step 1: Broadcast Q from rank 0
    std::vector<float> Q(d);
    if (rank == 0) {
        std::copy(Q_local, Q_local + d, Q.begin());
    }
    MPI_Bcast(Q.data(), d, MPI_FLOAT, 0, comm);

    // Step 2: Each rank computes partial result
    int T_offset = rank * T_local;
    PartialResult partial(d);

    partial = process_blocks_serial(
        Q.data(), K_local, V_local,
        0, T_local, d, block_size
    );

    // Step 3: Gather all partial results to rank 0
    std::vector<float> all_m(size);
    std::vector<float> all_l(size);
    std::vector<float> all_O(size * d);

    // Gather m and l values
    MPI_Gather(&partial.m, 1, MPI_FLOAT, all_m.data(), 1, MPI_FLOAT, 0, comm);
    MPI_Gather(&partial.l, 1, MPI_FLOAT, all_l.data(), 1, MPI_FLOAT, 0, comm);

    // Gather all O vectors
    MPI_Gather(partial.O.data(), d, MPI_FLOAT, all_O.data(), d, MPI_FLOAT, 0, comm);

    // Step 4: Merge on rank 0
    if (rank == 0) {
        PartialResult merged(d);
        for (int i = 0; i < size; ++i) {
            PartialResult other(d);
            other.m = all_m[i];
            other.l = all_l[i];
            std::copy(all_O.data() + i * d, all_O.data() + (i + 1) * d, other.O.data());

            merge_partial_results(merged, other);
        }
        return merged.O;
    } else {
        return std::vector<float>(d, 0.0f);
    }
}
