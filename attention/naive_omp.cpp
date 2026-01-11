#include "attention.h"
#include <cmath>
#include <vector>
#include <limits>
#include <iostream>
#include <omp.h>

/**
 * Naive Attention with OpenMP Parallelization
 *
 * Algorithm (same as naive_serial, but parallelized):
 * 1. Compute S = Q @ K^T  -> scores [T] (parallelized)
 * 2. Compute softmax(S) (parallelized reduction)
 * 3. Compute O = softmax(S) @ V (parallelized)
 *
 * Time complexity: O(Td)
 * Space complexity: O(T) for storing scores
 *
 * Parallelization strategy:
 * - Step 1: Parallelize dot products across T positions
 * - Step 2: Parallelize max and sum computation using reduction
 * - Step 3: Parallelize weighted sum across T positions
 */
std::vector<float> naive_attention_omp(
    const float* Q,
    const float* K,
    const float* V,
    int T,
    int d,
    int num_threads
) {
    // Step 1: Compute scores S = Q @ K^T
    // Q is [1 x d], K is [T x d], output S is [T]
    std::vector<float> scores(T);

    // Parallelize dot products across T positions
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < T; ++i) {
        // Compute dot product: Q · K[i]
        float sum = 0.0f;
        const float* K_row = K + i * d;

        // Inner loop: can be auto-vectorized by compiler
        for (int j = 0; j < d; ++j) {
            sum += Q[j] * K_row[j];
        }
        scores[i] = sum;
    }

    // Step 2: Compute softmax(S) with numerical stability
    // Find max for numerical stability (parallel reduction)
    float max_score = -std::numeric_limits<float>::infinity();
    #pragma omp parallel for reduction(max:max_score) num_threads(num_threads)
    for (int i = 0; i < T; ++i) {
        max_score = std::max(max_score, scores[i]);
    }

    // Compute exp and sum (parallel reduction)
    std::vector<float> exp_scores(T);
    float sum_exp = 0.0f;
    #pragma omp parallel for reduction(+:sum_exp) num_threads(num_threads)
    for (int i = 0; i < T; ++i) {
        exp_scores[i] = std::exp(scores[i] - max_score);
        sum_exp += exp_scores[i];
    }

    // Normalize (parallel)
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < T; ++i) {
        exp_scores[i] /= sum_exp;
    }

    // Step 3: Compute output O = softmax(S) @ V
    // exp_scores is [T], V is [T x d], output is [1 x d]
    std::vector<float> output(d, 0.0f);

    // Parallelize across d dimensions
    #pragma omp parallel for num_threads(num_threads)
    for (int j = 0; j < d; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < T; ++i) {
            float weight = exp_scores[i];
            const float* V_row = V + i * d;
            sum += weight * V_row[j];
        }
        output[j] = sum;
    }

    return output;
}
