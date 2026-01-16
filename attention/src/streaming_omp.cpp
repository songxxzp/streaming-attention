#include "attention.h"
#include "../utils/softmax_online.h"
#include <vector>
#include <cmath>
#include <limits>
#include <omp.h>

/**
 * Structure to hold partial results from a thread/chunk
 * Extends OnlineSoftmaxState with output vector O
 */
struct PartialResult : public OnlineSoftmaxState {
    std::vector<float> O; // Partial output [d]

    PartialResult(int d) : OnlineSoftmaxState(), O(d, 0.0f) {}
};

/**
 * Merge two partial results (reduction operation for online softmax)
 *
 * This is the key operation for parallelizing online softmax.
 * Given two independent partial states, compute their merged state.
 *
 * @param left   First partial state (modified in-place)
 * @param right  Second partial state
 */
inline void merge_partial_results(PartialResult& left, const PartialResult& right) {
    int d = left.O.size();

    // m_new = max(m1, m2)
    float m_new = std::max(left.m, right.m);

    // l_new = l1 * exp(m1 - m_new) + l2 * exp(m2 - m_new)
    float scale_left = std::exp(left.m - m_new);
    float scale_right = std::exp(right.m - m_new);
    float l_new = left.l * scale_left + right.l * scale_right;

    // O_new = O1 * (l1 * scale_left / l_new) + O2 * (l2 * scale_right / l_new)
    float alpha_left = (left.l * scale_left) / l_new;
    float alpha_right = (right.l * scale_right) / l_new;

    for (int j = 0; j < d; ++j) {
        left.O[j] = left.O[j] * alpha_left + right.O[j] * alpha_right;
    }

    left.m = m_new;
    left.l = l_new;
}

/**
 * Process a sequence of blocks and return partial result
 */
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
        // Process block - result inherits from OnlineSoftmaxState
        process_streaming_block(scores.data(), V_block, result, result.O.data(),
                                current_block_size, d);
    }

    return result;
}

/**
 * Streaming Block Attention with OpenMP Parallelization
 *
 * Parallelization Strategy:
 * 1. Divide K, V into chunks (each chunk contains multiple blocks)
 * 2. Each thread processes its chunk independently, producing partial (m, l, O)
 * 3. Use tree reduction to merge partial results
 */
std::vector<float> streaming_attention_omp(
    const float* Q,
    const float* K,
    const float* V,
    int T,
    int d,
    int block_size,
    int num_threads
) {
    // Set number of threads
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    // Determine number of chunks based on thread count
    int n_threads = omp_get_max_threads();

    // Each thread processes a chunk of blocks
    // We want at least 2 blocks per thread for load balancing
    int min_blocks_per_thread = 2;
    int n_chunks = std::min(n_threads, (T + block_size * min_blocks_per_thread - 1) /
                             (block_size * min_blocks_per_thread));
    n_chunks = std::max(1, n_chunks);

    std::vector<PartialResult> partials;

    // Parallel computation of partial results
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        #pragma omp single
        {
            partials.resize(n_chunks, PartialResult(d));
        }

        #pragma omp barrier

        // Each thread processes its assigned chunk
        if (tid < n_chunks) {
            int chunk_size = (T + n_chunks - 1) / n_chunks;
            int T_start = tid * chunk_size;
            int T_end = std::min(T_start + chunk_size, T);

            partials[tid] = process_blocks_serial(Q, K, V, T_start, T_end, d, block_size);
        }
    }

    // Sequential tree reduction
    PartialResult merged(d);
    for (int i = 0; i < n_chunks; ++i) {
        merge_partial_results(merged, partials[i]);
    }

    return merged.O;
}

/**
 * NUMA-aware version with first-touch initialization
 *
 * This version ensures that K, V are allocated in a NUMA-aware manner
 * (assuming the caller has done first-touch allocation).
 */
std::vector<float> streaming_attention_omp_numa(
    const float* Q,
    const float* K,
    const float* V,
    int T,
    int d,
    int block_size,
    int num_threads
) {
    // Same implementation as streaming_attention_omp
    // The NUMA optimization depends on how K, V are allocated by the caller
    return streaming_attention_omp(Q, K, V, T, d, block_size, num_threads);
}
