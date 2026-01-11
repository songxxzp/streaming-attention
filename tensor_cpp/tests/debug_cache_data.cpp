/**
 * @file debug_cache_data.cpp
 * @brief Debug KV cache data to find corruption
 */

#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/ops.h"
#include "tensor_cpp/kv_cache.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace tensor_cpp;
using namespace tensor_cpp::qwen3;

// Helper to print tensor stats
void print_tensor_stats(const char* name, const Tensor& t) {
    float min_val = t[0];
    float max_val = t[0];
    float sum = 0.0f;

    for (size_t i = 0; i < t.size(); ++i) {
        min_val = std::min(min_val, t[i]);
        max_val = std::max(max_val, t[i]);
        sum += t[i];
    }

    std::cout << "  " << name << ": size=" << t.size()
              << ", range=[" << min_val << ", " << max_val << "], mean=" << (sum / t.size()) << "\n";
}

int main() {
    std::cout << "\n============================================================\n";
    std::cout << "  KV Cache Data Debug Test\n";
    std::cout << "============================================================\n\n";

    try {
        std::string model_path = "/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors";

        std::cout << "Loading weights...\n";
        Qwen3Weights weights = load_qwen3_weights(model_path);
        std::cout << "Weights loaded!\n\n";

        // Simple test: "Hello"
        std::vector<long> input_ids_data = {151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198};

        auto kv_cache = std::make_unique<KVCache>(
            weights.num_layers, 1, weights.num_key_value_heads,
            weights.head_dim, 4096
        );

        // ====================================================================
        // Phase 1: Prefill
        // ====================================================================
        std::cout << "========== PHASE 1: PREFILL ==========\n";
        {
            Shape input_shape({1, input_ids_data.size()});
            TensorL input(input_ids_data, input_shape);

            Tensor hidden_states = qwen3::qwen3_forward_with_cache(
                input, kv_cache.get(), weights.embed_tokens,
                weights.layers, weights.norm_weight,
                weights.num_layers, weights.num_attention_heads,
                weights.num_key_value_heads, weights.head_dim, 1e-6f
            );

            std::cout << "After prefill: current_seq_len = " << kv_cache->current_seq_len << "\n";

            // Check layer 0 cache
            Tensor k0_cached = kv_cache->get_cached_keys(0, kv_cache->current_seq_len);
            std::cout << "Layer 0 K cache: shape = ["
                      << k0_cached.shape()[0] << ", " << k0_cached.shape()[1] << ", "
                      << k0_cached.shape()[2] << ", " << k0_cached.shape()[3] << "]\n";
            print_tensor_stats("Layer 0 K", k0_cached);

            // Get first prediction
            size_t hidden_size = hidden_states.shape()[2];
            size_t vocab_size = weights.lm_head.shape()[0];
            size_t batch_size = hidden_states.shape()[0];
            size_t seq_len = hidden_states.shape()[1];

            size_t last_idx = (batch_size - 1) * seq_len * hidden_size + (seq_len - 1) * hidden_size;
            std::vector<float> last_hidden(hidden_size);
            for (size_t i = 0; i < hidden_size; ++i) {
                last_hidden[i] = hidden_states[last_idx + i];
            }

            std::vector<float> logits(vocab_size);
            for (size_t v = 0; v < vocab_size; ++v) {
                float sum = 0.0f;
                for (size_t h = 0; h < hidden_size; ++h) {
                    sum += last_hidden[h] * weights.lm_head[v * hidden_size + h];
                }
                logits[v] = sum;
            }

            long predicted_token = 0;
            float max_logit = logits[0];
            for (size_t v = 1; v < vocab_size; ++v) {
                if (logits[v] > max_logit) {
                    max_logit = logits[v];
                    predicted_token = static_cast<long>(v);
                }
            }

            std::cout << "Predicted token: " << predicted_token << "\n\n";
            input_ids_data.push_back(predicted_token);
        }

        // ====================================================================
        // Phase 2: Decode - Check if cache gets corrupted
        // ====================================================================
        std::cout << "========== PHASE 2: DECODE ==========\n";

        for (int step = 0; step < 5; ++step) {
            std::cout << "\n--- Step " << (step + 1) << " ---\n";
            std::cout << "BEFORE: current_seq_len = " << kv_cache->current_seq_len << "\n";

            // Check cache before (should have previous data)
            if (kv_cache->current_seq_len > 0) {
                Tensor k_before = kv_cache->get_cached_keys(0, kv_cache->current_seq_len);
                std::cout << "Cache BEFORE: shape = ["
                          << k_before.shape()[0] << ", " << k_before.shape()[1] << ", "
                          << k_before.shape()[2] << ", " << k_before.shape()[3] << "]\n";

                // Check first few values to see if they're consistent
                std::cout << "  First 5 values of first batch, first head, first position: ";
                for (int i = 0; i < 5; ++i) {
                    std::cout << std::fixed << std::setprecision(4) << k_before[i] << " ";
                }
                std::cout << "\n";
            }

            // Forward pass with new token
            std::vector<long> new_token = {input_ids_data.back()};
            std::cout << "Input token: " << new_token[0] << "\n";

            Shape input_shape({1, 1});
            TensorL input(new_token, input_shape);

            Tensor hidden_states = qwen3::qwen3_forward_with_cache(
                input, kv_cache.get(), weights.embed_tokens,
                weights.layers, weights.norm_weight,
                weights.num_layers, weights.num_attention_heads,
                weights.num_key_value_heads, weights.head_dim, 1e-6f
            );

            std::cout << "AFTER: current_seq_len = " << kv_cache->current_seq_len << "\n";

            // Check cache after
            Tensor k_after = kv_cache->get_cached_keys(0, kv_cache->current_seq_len);
            std::cout << "Cache AFTER: shape = ["
                      << k_after.shape()[0] << ", " << k_after.shape()[1] << ", "
                      << k_after.shape()[2] << ", " << k_after.shape()[3] << "]\n";
            print_tensor_stats("Layer 0 K", k_after);

            // Check if first few values changed
            std::cout << "  First 5 values of first batch, first head, first position: ";
            for (int i = 0; i < 5; ++i) {
                std::cout << std::fixed << std::setprecision(4) << k_after[i] << " ";
            }
            std::cout << "\n";

            // Get prediction
            size_t hidden_size = hidden_states.shape()[2];
            size_t vocab_size = weights.lm_head.shape()[0];

            std::vector<float> last_hidden(hidden_size);
            for (size_t i = 0; i < hidden_size; ++i) {
                last_hidden[i] = hidden_states[i];
            }

            std::vector<float> logits(vocab_size);
            for (size_t v = 0; v < vocab_size; ++v) {
                float sum = 0.0f;
                for (size_t h = 0; h < hidden_size; ++h) {
                    sum += last_hidden[h] * weights.lm_head[v * hidden_size + h];
                }
                logits[v] = sum;
            }

            long predicted_token = 0;
            float max_logit = logits[0];
            for (size_t v = 1; v < vocab_size; ++v) {
                if (logits[v] > max_logit) {
                    max_logit = logits[v];
                    predicted_token = static_cast<long>(v);
                }
            }

            std::cout << "Predicted token: " << predicted_token << " (logit=" << max_logit << ")\n";
            input_ids_data.push_back(predicted_token);
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n✗ ERROR: " << e.what() << "\n\n";
        return 1;
    }
}
