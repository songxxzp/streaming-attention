/**
 * @file debug_cache.cpp
 * @brief Debug test to compare cached vs non-cached forward passes
 */

#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/ops.h"
#include "tensor_cpp/kv_cache.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

using namespace tensor_cpp;
using namespace tensor_cpp::qwen3;

int main() {
    try {
        std::string model_path = "/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors";

        // Load weights
        std::cout << "Loading weights...\n";
        Qwen3Weights weights = load_qwen3_weights(model_path);
        std::cout << "Weights loaded!\n\n";

        // Simple test: "Hello" prompt (9 tokens)
        std::vector<long> input_ids_data = {151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198};

        std::cout << "Test input: " << input_ids_data.size() << " tokens\n\n";

        // Test 1: Forward pass WITHOUT cache
        std::cout << "=== Test 1: WITHOUT cache ===\n";
        auto start1 = std::chrono::high_resolution_clock::now();

        Shape input_shape({1, input_ids_data.size()});
        TensorL input(input_ids_data, input_shape);

        Tensor hidden_no_cache = qwen3::qwen3_forward(
            input,
            weights.embed_tokens,
            weights.layers,
            weights.norm_weight,
            weights.num_layers,
            weights.num_attention_heads,
            weights.num_key_value_heads,
            weights.head_dim,
            1e-6f
        );

        auto end1 = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);

        std::cout << "Time: " << duration1.count() << " ms\n";
        std::cout << "Output shape: " << hidden_no_cache.shape()[0] << " "
                  << hidden_no_cache.shape()[1] << " "
                  << hidden_no_cache.shape()[2] << "\n\n";

        // Test 2: Forward pass WITH cache (first pass, no prior cache)
        std::cout << "=== Test 2: WITH cache (prefill) ===\n";
        auto start2 = std::chrono::high_resolution_clock::now();

        auto kv_cache = std::make_unique<KVCache>(
            weights.num_layers,
            1,
            weights.num_key_value_heads,
            weights.head_dim,
            4096
        );

        TensorL input2(input_ids_data, input_shape);
        Tensor hidden_with_cache = qwen3::qwen3_forward_with_cache(
            input2,
            kv_cache.get(),
            weights.embed_tokens,
            weights.layers,
            weights.norm_weight,
            weights.num_layers,
            weights.num_attention_heads,
            weights.num_key_value_heads,
            weights.head_dim,
            1e-6f
        );

        auto end2 = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);

        std::cout << "Time: " << duration2.count() << " ms\n";
        std::cout << "Output shape: " << hidden_with_cache.shape()[0] << " "
                  << hidden_with_cache.shape()[1] << " "
                  << hidden_with_cache.shape()[2] << "\n";
        std::cout << "Cache size: " << kv_cache->current_seq_len << " tokens\n\n";

        // Compare outputs
        std::cout << "=== Comparing outputs ===\n";
        if (hidden_no_cache.shape() != hidden_with_cache.shape()) {
            std::cout << "ERROR: Shapes differ!\n";
        } else {
            size_t total = hidden_no_cache.size();
            double max_diff = 0.0;
            size_t diff_count = 0;
            double sum_diff = 0.0;

            for (size_t i = 0; i < total; ++i) {
                double diff = std::abs(hidden_no_cache[i] - hidden_with_cache[i]);
                if (diff > 1e-5) {
                    diff_count++;
                    sum_diff += diff;
                    if (diff > max_diff) max_diff = diff;
                }
            }

            std::cout << "Total elements: " << total << "\n";
            std::cout << "Different elements: " << diff_count << " ("
                      << (100.0 * diff_count / total) << "%)\n";
            std::cout << "Max difference: " << max_diff << "\n";
            if (diff_count > 0) {
                std::cout << "Avg difference: " << (sum_diff / diff_count) << "\n";
            }
        }

        // Test 3: Decode phase (1 new token)
        std::cout << "\n=== Test 3: Decode phase (1 new token) ===\n";
        auto start3 = std::chrono::high_resolution_clock::now();

        std::vector<long> new_token = {151667};  // "Hello" continuation token
        Shape new_shape({1, 1});
        TensorL new_input(new_token, new_shape);

        Tensor hidden_decode = qwen3::qwen3_forward_with_cache(
            new_input,
            kv_cache.get(),
            weights.embed_tokens,
            weights.layers,
            weights.norm_weight,
            weights.num_layers,
            weights.num_attention_heads,
            weights.num_key_value_heads,
            weights.head_dim,
            1e-6f
        );

        auto end3 = std::chrono::high_resolution_clock::now();
        auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3);

        std::cout << "Time: " << duration3.count() << " ms\n";
        std::cout << "Output shape: " << hidden_decode.shape()[0] << " "
                  << hidden_decode.shape()[1] << " "
                  << hidden_decode.shape()[2] << "\n";
        std::cout << "Cache size: " << kv_cache->current_seq_len << " tokens\n\n";

        // Get logits for comparison
        std::cout << "=== Comparing logits for last position ===\n";
        size_t hidden_size = weights.hidden_size;
        size_t vocab_size = weights.lm_head.shape()[0];

        // From no-cache output (last position)
        std::vector<float> last_hidden_no_cache(hidden_size);
        size_t last_pos = input_ids_data.size() - 1;
        for (size_t i = 0; i < hidden_size; ++i) {
            last_hidden_no_cache[i] = hidden_no_cache[last_pos * hidden_size + i];
        }

        // From with-cache output (last position)
        std::vector<float> last_hidden_with_cache(hidden_size);
        for (size_t i = 0; i < hidden_size; ++i) {
            last_hidden_with_cache[i] = hidden_with_cache[last_pos * hidden_size + i];
        }

        // Compute top 5 tokens for each
        auto top_tokens = [](const std::vector<float>& hidden,
                            const Tensor& lm_head,
                            size_t vocab_size, size_t hidden_size) {
            std::vector<std::pair<float, long>> logits;
            for (size_t v = 0; v < vocab_size; ++v) {
                float sum = 0.0f;
                for (size_t h = 0; h < hidden_size; ++h) {
                    sum += hidden[h] * lm_head[v * hidden_size + h];
                }
                logits.push_back({sum, (long)v});
            }
            std::partial_sort(logits.begin(), logits.begin() + 5, logits.end(),
                             std::greater<std::pair<float, long>>());
            return std::vector<std::pair<float, long>>(logits.begin(), logits.begin() + 5);
        };

        auto top1 = top_tokens(last_hidden_no_cache, weights.lm_head, vocab_size, hidden_size);
        auto top2 = top_tokens(last_hidden_with_cache, weights.lm_head, vocab_size, hidden_size);

        std::cout << "Top 5 WITHOUT cache:\n";
        for (auto [logit, token] : top1) {
            std::cout << "  token=" << token << " logit=" << logit << "\n";
        }

        std::cout << "\nTop 5 WITH cache:\n";
        for (auto [logit, token] : top2) {
            std::cout << "  token=" << token << " logit=" << logit << "\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n✗ ERROR: " << e.what() << "\n\n";
        return 1;
    }
}
