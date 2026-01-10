#ifndef SOFTMAX_ONLINE_H
#define SOFTMAX_ONLINE_H

#include <cmath>
#include <vector>
#include <limits>

/**
 * Online Softmax Accumulator for Streaming Attention
 *
 * This implements the online softmax algorithm that maintains running statistics
 * while processing data in blocks. The key insight is that we can maintain:
 *   - m: running maximum
 *   - l: running normalizer (sum of exp(values - m))
 *   - O: running output
 *
 * For each new block with scores S_b and values V_b:
 *   1. m_new = max(m, max(S_b))
 *   2. l_new = l * exp(m - m_new) + sum(exp(S_b - m_new))
 *   3. O_new = O * (l * exp(m - m_new) / l_new) + sum(exp(S_b - m_new) * V_b) / l_new
 */
class OnlineSoftmaxState {
public:
    OnlineSoftmaxState() : m(-std::numeric_limits<float>::infinity()), l(1.0f) {}

    float m;  // Running maximum
    float l;  // Running normalizer
};

/**
 * Process a single block in streaming attention
 *
 * @param scores     Block scores Q @ K_b^T [block_size]
 * @param values     Block values V_b [block_size x d]
 * @param state      Current softmax state (m, l), will be updated
 * @param output     Current output accumulator [d], will be updated in-place
 * @param block_size Number of elements in this block
 * @param d          Hidden dimension
 */
inline void process_streaming_block(
    const float* scores,
    const float* values,
    OnlineSoftmaxState& state,
    float* output,
    int block_size,
    int d
) {
    // Step 1: Find max in current block and global max
    float m_new = state.m;
    for (int i = 0; i < block_size; ++i) {
        m_new = std::max(m_new, scores[i]);
    }

    // Step 2: Compute sum of exp(scores - m_new) for this block
    float sum_exp_block = 0.0f;
    for (int i = 0; i < block_size; ++i) {
        sum_exp_block += std::exp(scores[i] - m_new);
    }

    // Step 3: Update normalizer
    // l_new = l * exp(m - m_new) + sum_exp_block
    float scale_old = std::exp(state.m - m_new);
    float l_new = state.l * scale_old + sum_exp_block;

    // Step 4: Update output
    // O_new = O * (l * scale_old / l_new) + sum(exp(s_i - m_new) * V_i) / l_new
    float alpha_old = (state.l * scale_old) / l_new;
    float alpha_new = 1.0f / l_new;

    // Scale old output
    for (int j = 0; j < d; ++j) {
        output[j] *= alpha_old;
    }

    // Add weighted sum of values from this block
    for (int i = 0; i < block_size; ++i) {
        float weight = alpha_new * std::exp(scores[i] - m_new);
        const float* v_row = values + i * d;
        for (int j = 0; j < d; ++j) {
            output[j] += weight * v_row[j];
        }
    }

    // Step 5: Update state
    state.m = m_new;
    state.l = l_new;
}

/**
 * Simplified version: only update softmax state without computing output
 * Useful for multi-pass algorithms or validation
 */
inline void update_softmax_state(
    const float* scores,
    OnlineSoftmaxState& state,
    int block_size
) {
    float m_new = state.m;
    for (int i = 0; i < block_size; ++i) {
        m_new = std::max(m_new, scores[i]);
    }

    float sum_exp_block = 0.0f;
    for (int i = 0; i < block_size; ++i) {
        sum_exp_block += std::exp(scores[i] - m_new);
    }

    float scale_old = std::exp(state.m - m_new);
    state.l = state.l * scale_old + sum_exp_block;
    state.m = m_new;
}

#endif // SOFTMAX_ONLINE_H
