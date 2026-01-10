#include "attention/attention.h"
#include <cmath>
#include <vector>
#include <limits>
#include <iostream>

/**
 * Naive Attention (Serial Baseline)
 *
 * Algorithm:
 * 1. Compute S = Q @ K^T  -> scores [T]
 * 2. Compute softmax(S)
 * 3. Compute O = softmax(S) @ V
 *
 * Time complexity: O(Td)
 * Space complexity: O(T) for storing scores
 */
std::vector<float> naive_attention_serial(
    const float* Q,
    const float* K,
    const float* V,
    int T,
    int d
) {
    // Step 1: Compute scores S = Q @ K^T
    // Q is [1 x d], K is [T x d], output S is [T]
    std::vector<float> scores(T);

    for (int i = 0; i < T; ++i) {
        // Compute dot product: Q · K[i]
        float sum = 0.0f;
        const float* K_row = K + i * d;
        for (int j = 0; j < d; ++j) {
            sum += Q[j] * K_row[j];
        }
        scores[i] = sum;
    }

    // Step 2: Compute softmax(S) with numerical stability
    // Find max for numerical stability
    float max_score = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < T; ++i) {
        max_score = std::max(max_score, scores[i]);
    }

    // Compute exp and sum
    std::vector<float> exp_scores(T);
    float sum_exp = 0.0f;
    for (int i = 0; i < T; ++i) {
        exp_scores[i] = std::exp(scores[i] - max_score);
        sum_exp += exp_scores[i];
    }

    // Normalize
    for (int i = 0; i < T; ++i) {
        exp_scores[i] /= sum_exp;
    }

    // Step 3: Compute output O = softmax(S) @ V
    // exp_scores is [T], V is [T x d], output is [1 x d]
    std::vector<float> output(d, 0.0f);

    for (int i = 0; i < T; ++i) {
        float weight = exp_scores[i];
        const float* V_row = V + i * d;
        for (int j = 0; j < d; ++j) {
            output[j] += weight * V_row[j];
        }
    }

    return output;
}

// Utility functions
float compute_l2_error(const float* a, const float* b, int size) {
    float sum_sq = 0.0f;
    float sum_ref = 0.0f;

    for (int i = 0; i < size; ++i) {
        float diff = a[i] - b[i];
        sum_sq += diff * diff;
        sum_ref += b[i] * b[i];
    }

    float ref_norm = std::sqrt(sum_ref);
    if (ref_norm < 1e-10f) {
        return std::sqrt(sum_sq);
    }
    return std::sqrt(sum_sq) / ref_norm;
}

float compute_max_error(const float* a, const float* b, int size) {
    float max_err = 0.0f;
    for (int i = 0; i < size; ++i) {
        max_err = std::max(max_err, std::abs(a[i] - b[i]));
    }
    return max_err;
}
