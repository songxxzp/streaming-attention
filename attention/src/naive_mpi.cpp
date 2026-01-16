#include <mpi.h>
#include "attention.h"
#include <vector>
#include <cmath>
#include <limits>
#include <omp.h>

/**
 * Naive Attention with MPI + OpenMP Hybrid Parallelization
 *
 * Algorithm:
 * 1. Rank 0 broadcasts Q to all ranks
 * 2. Each rank computes attention for its local portion of T tokens
 * 3. Each rank internally uses OpenMP for parallelization
 * 4. Gather partial outputs to rank 0 and sum them
 *
 * Data distribution:
 * - T tokens are distributed evenly across MPI ranks
 * - Each rank's local tokens are further parallelized with OpenMP
 *
 * @param Q_local Query vector [1 x d] (only rank 0 needs valid data)
 * @param K_local Local portion of Key cache [T_local x d]
 * @param V_local Local portion of Value cache [T_local x d]
 * @param T_local Local sequence length (this rank's portion)
 * @param T_global Total sequence length (across all ranks)
 * @param d Hidden dimension
 * @param num_omp_threads Number of OpenMP threads per rank (0 = use OMP_NUM_THREADS)
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
) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Set OpenMP thread count if specified
    if (num_omp_threads > 0) {
        omp_set_num_threads(num_omp_threads);
    }

    // Step 1: Broadcast Q from rank 0 to all ranks
    std::vector<float> Q(d);
    if (rank == 0) {
        std::copy(Q_local, Q_local + d, Q.begin());
    }
    MPI_Bcast(Q.data(), d, MPI_FLOAT, 0, comm);

    // Step 2: Each rank computes partial output for its local tokens
    // Each rank processes T_local tokens (token indices: rank * T_local to (rank+1) * T_local - 1)
    // But actually T_local is just the local portion, so we use 0 to T_local

    // We'll compute:
    // - Partial output: sum of (weight * V) for local tokens
    // - Partial sum_exp: sum of exp_scores for local tokens (needed for normalization)

    std::vector<float> partial_output(d, 0.0f);
    float partial_sum_exp = 0.0f;
    float partial_max_score = -std::numeric_limits<float>::infinity();

    // First pass: compute local max score (for numerical stability)
    #pragma omp parallel reduction(max:partial_max_score)
    {
        float local_max = -std::numeric_limits<float>::infinity();

        #pragma omp for nowait
        for (int i = 0; i < T_local; ++i) {
            // Compute dot product: Q · K_local[i]
            float sum = 0.0f;
            const float* K_row = K_local + i * d;
            for (int j = 0; j < d; ++j) {
                sum += Q[j] * K_row[j];
            }
            local_max = std::max(local_max, sum);
        }

        partial_max_score = std::max(partial_max_score, local_max);
    }

    // Second pass: compute exp scores and sum, and weighted output
    // But first we need the global max score
    float global_max_score;
    MPI_Allreduce(&partial_max_score, &global_max_score, 1, MPI_FLOAT, MPI_MAX, comm);

    // Now compute exp scores and partial output with global max
    #pragma omp parallel
    {
        std::vector<float> local_output(d, 0.0f);
        float local_sum_exp = 0.0f;

        #pragma omp for
        for (int i = 0; i < T_local; ++i) {
            // Compute dot product: Q · K_local[i]
            float sum = 0.0f;
            const float* K_row = K_local + i * d;
            for (int j = 0; j < d; ++j) {
                sum += Q[j] * K_row[j];
            }

            // Compute exp(score - global_max)
            float exp_score = std::exp(sum - global_max_score);
            local_sum_exp += exp_score;

            // Add weighted V to local output
            const float* V_row = V_local + i * d;
            for (int j = 0; j < d; ++j) {
                local_output[j] += exp_score * V_row[j];
            }
        }

        // Merge local results
        #pragma omp critical
        {
            partial_sum_exp += local_sum_exp;
            for (int j = 0; j < d; ++j) {
                partial_output[j] += local_output[j];
            }
        }
    }

    // Step 3: Gather all partial_sum_exp to rank 0 and compute global sum
    float global_sum_exp;
    MPI_Reduce(&partial_sum_exp, &global_sum_exp, 1, MPI_FLOAT, MPI_SUM, 0, comm);

    // Step 4: Gather all partial outputs to rank 0 and sum them
    std::vector<float> global_output(d, 0.0f);

    if (rank == 0) {
        // Start with rank 0's own partial output
        for (int j = 0; j < d; ++j) {
            global_output[j] = partial_output[j];
        }

        // Receive partial outputs from other ranks
        for (int source = 1; source < size; ++source) {
            std::vector<float> recv_buffer(d);
            MPI_Status status;
            MPI_Recv(recv_buffer.data(), d, MPI_FLOAT, source, 0, comm, &status);

            // Accumulate
            for (int j = 0; j < d; ++j) {
                global_output[j] += recv_buffer[j];
            }
        }

        // Normalize by global_sum_exp
        for (int j = 0; j < d; ++j) {
            global_output[j] /= global_sum_exp;
        }

        return global_output;
    } else {
        // Send partial output to rank 0
        MPI_Send(partial_output.data(), d, MPI_FLOAT, 0, 0, comm);
        return std::vector<float>(d, 0.0f);  // Dummy return
    }
}

/**
 * Simplified version using MPI_Gather instead of point-to-point
 *
 * This version is cleaner and more readable
 */
std::vector<float> naive_attention_mpi_simple(
    const float* Q_local,
    const float* K_local,
    const float* V_local,
    int T_local,
    int T_global,
    int d,
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

    // Step 2: Find global max score (for numerical stability)
    float local_max = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < T_local; ++i) {
        float sum = 0.0f;
        const float* K_row = K_local + i * d;
        for (int j = 0; j < d; ++j) {
            sum += Q[j] * K_row[j];
        }
        local_max = std::max(local_max, sum);
    }

    float global_max;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_FLOAT, MPI_MAX, comm);

    // Step 3: Compute partial output and sum_exp
    std::vector<float> partial_output(d, 0.0f);
    float partial_sum_exp = 0.0f;

    for (int i = 0; i < T_local; ++i) {
        float sum = 0.0f;
        const float* K_row = K_local + i * d;
        for (int j = 0; j < d; ++j) {
            sum += Q[j] * K_row[j];
        }

        float exp_score = std::exp(sum - global_max);
        partial_sum_exp += exp_score;

        const float* V_row = V_local + i * d;
        for (int j = 0; j < d; ++j) {
            partial_output[j] += exp_score * V_row[j];
        }
    }

    // Step 4: Gather all partial_sum_exp to compute global sum
    std::vector<float> all_sum_exp(size);
    MPI_Gather(&partial_sum_exp, 1, MPI_FLOAT, all_sum_exp.data(), 1, MPI_FLOAT, 0, comm);

    // Step 5: Gather all partial outputs
    std::vector<float> all_outputs(size * d);
    MPI_Gather(partial_output.data(), d, MPI_FLOAT, all_outputs.data(), d, MPI_FLOAT, 0, comm);

    // Step 6: Merge and normalize on rank 0
    if (rank == 0) {
        std::vector<float> global_output(d, 0.0f);
        float global_sum_exp = 0.0f;

        // Sum all sum_exp values
        for (int i = 0; i < size; ++i) {
            global_sum_exp += all_sum_exp[i];
        }

        // Sum all outputs and normalize
        for (int i = 0; i < size; ++i) {
            float scale = all_sum_exp[i] / global_sum_exp;
            const float* rank_output = all_outputs.data() + i * d;
            for (int j = 0; j < d; ++j) {
                global_output[j] += rank_output[j];
            }
        }

        // The outputs are already implicitly normalized by the scale factors
        // But we need to divide by global_sum_exp one more time because:
        // partial_output[i] = sum(exp_score_k * V[k]) for local k
        // global_output = sum over ranks of partial_output[i]
        // We want: output = sum over all k of (exp_score_k / global_sum_exp) * V[k]
        // = (1/global_sum_exp) * sum over all k of exp_score_k * V[k]
        // = (1/global_sum_exp) * global_output_before_normalize

        // Actually, let me reconsider: partial_output already contains sum(exp * V)
        // So we just need to divide by global_sum_exp
        for (int j = 0; j < d; ++j) {
            global_output[j] /= global_sum_exp;
        }

        return global_output;
    } else {
        return std::vector<float>(d, 0.0f);
    }
}
