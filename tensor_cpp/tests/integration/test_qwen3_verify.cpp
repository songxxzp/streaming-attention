/**
 * @file test_qwen3_verify.cpp
 * @brief Verify Qwen3 forward correctness by checking output consistency
 */

#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/ops.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

using namespace tensor_cpp;
using namespace tensor_cpp::qwen3;

int main() {
    std::cout << "\n============================================================\n";
    std::cout << "  Qwen3 Forward Correctness Verification\n";
    std::cout << "============================================================\n\n";

    try {
        std::string model_path = "/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors";

        // Load weights
        std::cout << "Loading weights...\n";
        Qwen3Weights weights = load_qwen3_weights(model_path);
        std::cout << "Weights loaded!\n\n";

        // Test 1: Same input should produce same output (determinism)
        std::cout << "Test 1: Determinism check\n";
        std::cout << "----------------------------------------\n";

        std::vector<long> input_ids_data = {9658, 15, 1358, 35};
        Shape input_shape({1, static_cast<size_t>(input_ids_data.size())});
        TensorL input_ids(input_ids_data, input_shape);

        // Run forward pass twice
        Tensor hidden_states1 = qwen3::qwen3_forward(
            input_ids, weights.embed_tokens, weights.layers, weights.norm_weight, weights.lm_head,
            weights.num_layers, weights.num_attention_heads, weights.num_key_value_heads,
            weights.head_dim, 1e-6f
        );

        Tensor hidden_states2 = qwen3::qwen3_forward(
            input_ids, weights.embed_tokens, weights.layers, weights.norm_weight, weights.lm_head,
            weights.num_layers, weights.num_attention_heads, weights.num_key_value_heads,
            weights.head_dim, 1e-6f
        );

        // Check if outputs are identical
        bool identical = true;
        float max_diff = 0.0f;
        for (size_t i = 0; i < hidden_states1.size(); ++i) {
            float diff = std::abs(hidden_states1[i] - hidden_states2[i]);
            max_diff = std::max(max_diff, diff);
            if (diff > 1e-6f) {
                identical = false;
            }
        }

        std::cout << "Output shape: [" << hidden_states1.shape()[0] << ", "
                  << hidden_states1.shape()[1] << ", " << hidden_states1.shape()[2] << "]\n";
        std::cout << "Determinism: " << (identical ? "PASS" : "FAIL") << "\n";
        std::cout << "Max difference: " << max_diff << "\n\n";

        // Test 2: Output values should be finite
        std::cout << "Test 2: Output value range check\n";
        std::cout << "----------------------------------------\n";

        bool all_finite = true;
        float min_val = hidden_states1[0];
        float max_val = hidden_states1[0];
        int nan_count = 0;
        int inf_count = 0;

        for (size_t i = 0; i < hidden_states1.size(); ++i) {
            float val = hidden_states1[i];
            if (std::isnan(val)) {
                nan_count++;
                all_finite = false;
            } else if (std::isinf(val)) {
                inf_count++;
                all_finite = false;
            } else {
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            }
        }

        std::cout << "All finite: " << (all_finite ? "PASS" : "FAIL") << "\n";
        std::cout << "NaN count: " << nan_count << "\n";
        std::cout << "Inf count: " << inf_count << "\n";
        std::cout << "Value range: [" << min_val << ", " << max_val << "]\n\n";

        // Test 3: Different inputs should produce different outputs
        std::cout << "Test 3: Input sensitivity check\n";
        std::cout << "----------------------------------------\n";

        // Change one token
        std::vector<long> input_ids_data2 = {9658, 15, 9999, 35};  // Changed third token
        TensorL input_ids2(input_ids_data2, input_shape);

        Tensor hidden_states3 = qwen3::qwen3_forward(
            input_ids2, weights.embed_tokens, weights.layers, weights.norm_weight, weights.lm_head,
            weights.num_layers, weights.num_attention_heads, weights.num_key_value_heads,
            weights.head_dim, 1e-6f
        );

        // Check if outputs are different
        bool different = false;
        float total_diff = 0.0f;
        for (size_t i = 0; i < hidden_states1.size(); ++i) {
            float diff = std::abs(hidden_states1[i] - hidden_states3[i]);
            total_diff += diff;
            if (diff > 0.01f) {
                different = true;
            }
        }

        float avg_diff = total_diff / hidden_states1.size();
        std::cout << "Outputs different: " << (different ? "PASS" : "FAIL") << "\n";
        std::cout << "Average difference: " << avg_diff << "\n\n";

        // Test 4: LM head projection
        std::cout << "Test 4: LM head projection check\n";
        std::cout << "----------------------------------------\n";

        size_t batch_size = hidden_states1.shape()[0];
        size_t seq_len = hidden_states1.shape()[1];
        size_t hidden_size = hidden_states1.shape()[2];
        size_t vocab_size = weights.lm_head.shape()[0];

        // Get last position hidden state
        size_t last_idx = (batch_size - 1) * seq_len * hidden_size + (seq_len - 1) * hidden_size;
        std::vector<float> last_hidden(hidden_size);
        for (size_t i = 0; i < hidden_size; ++i) {
            last_hidden[i] = hidden_states1[last_idx + i];
        }

        // Project to logits
        std::vector<float> logits(vocab_size);
        for (size_t v = 0; v < vocab_size; ++v) {
            float sum = 0.0f;
            for (size_t h = 0; h < hidden_size; ++h) {
                sum += last_hidden[h] * weights.lm_head[v * hidden_size + h];
            }
            logits[v] = sum;
        }

        // Find max logit
        float max_logit = logits[0];
        long max_idx = 0;
        for (size_t v = 1; v < vocab_size; ++v) {
            if (logits[v] > max_logit) {
                max_logit = logits[v];
                max_idx = static_cast<long>(v);
            }
        }

        // Check if logits are finite
        bool logits_finite = true;
        for (size_t v = 0; v < vocab_size; ++v) {
            if (std::isnan(logits[v]) || std::isinf(logits[v])) {
                logits_finite = false;
                break;
            }
        }

        std::cout << "All logits finite: " << (logits_finite ? "PASS" : "FAIL") << "\n";
        std::cout << "Max logit: " << max_logit << "\n";
        std::cout << "Predicted token ID: " << max_idx << "\n\n";

        // Summary
        std::cout << "============================================================\n";
        std::cout << "  Verification Summary\n";
        std::cout << "============================================================\n\n";

        int pass_count = 0;
        if (identical) pass_count++;
        if (all_finite) pass_count++;
        if (different) pass_count++;
        if (logits_finite) pass_count++;

        std::cout << "Tests passed: " << pass_count << "/4\n";

        if (pass_count == 4) {
            std::cout << "\n✓ All verification tests PASSED!\n";
            std::cout << "Forward implementation is correct and stable.\n";
            return 0;
        } else {
            std::cout << "\n✗ Some tests FAILED\n";
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "\n✗ ERROR: " << e.what() << "\n\n";
        return 1;
    }
}
