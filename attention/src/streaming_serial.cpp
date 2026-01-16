#include "attention.h"
#include "../utils/softmax_online.h"
#include <vector>
#include <cmath>
#include <limits>

/**
 * Streaming Block Attention (Serial)
 *
 * Algorithm (Online Softmax):
 * - Process K, V in blocks of size `block_size`
 * - Maintain running statistics: m (max), l (normalizer), O (output)
 * - Update statistics for each block incrementally
 *
 * Time complexity: O(Td)
 * Space complexity: O(block_size) + O(d) for scores of one block
 *
 * Mathematically equivalent to naive_attention_serial
 */
std::vector<float> streaming_attention_serial(
    const float* Q,
    const float* K,
    const float* V,
    int T,
    int d,
    int block_size
) {
    // Initialize output accumulator
    std::vector<float> output(d, 0.0f);

    // Initialize online softmax state
    OnlineSoftmaxState state;

    // Process blocks
    int num_blocks = (T + block_size - 1) / block_size;

    for (int block = 0; block < num_blocks; ++block) {
        int block_start = block * block_size;
        int block_end = std::min(block_start + block_size, T);
        int current_block_size = block_end - block_start;

        // Compute scores for this block: S_b = Q @ K_b^T
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

        // Get values for this block
        const float* V_block = V + block_start * d;

        // Process this block using online softmax
        process_streaming_block(
            scores.data(),
            V_block,
            state,
            output.data(),
            current_block_size,
            d
        );
    }

    return output;
}
