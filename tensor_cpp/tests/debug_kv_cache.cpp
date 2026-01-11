/**
 * @file debug_kv_cache.cpp
 * @brief Debug KV cache to find the bug
 */

#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/ops.h"
#include "tensor_cpp/kv_cache.h"
#include <iostream>
#include <vector>
#include <cmath>

using namespace tensor_cpp;
using namespace tensor_cpp::qwen3;

int main() {
    std::cout << "\n============================================================\n";
    std::cout << "  KV Cache Debug Test\n";
    std::cout << "============================================================\n\n";

    try {
        std::string model_path = "/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors";

        // Load weights
        std::cout << "Loading weights...\n";
        Qwen3Weights weights = load_qwen3_weights(model_path);
        std::cout << "Weights loaded!\n\n";

        // Simple test: "Hello"
        std::vector<long> input_ids_data = {151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198};

        // Create KV cache
        auto kv_cache = std::make_unique<KVCache>(
            weights.num_layers,          // 28 layers
            1,                            // batch_size
            weights.num_key_value_heads,  // 8 KV heads
            weights.head_dim,             // 128 head_dim
            4096                          // max_seq_len
        );

        std::cout << "Initial cache state:\n";
        std::cout << "  current_seq_len: " << kv_cache->current_seq_len << "\n\n";

        // ====================================================================
        // Phase 1: Prefill - Process initial prompt
        // ====================================================================
        std::cout << "========== PHASE 1: PREFILL ==========\n";
        std::cout << "Input tokens: " << input_ids_data.size() << "\n";

        {
            Shape input_shape({1, input_ids_data.size()});
            TensorL input(input_ids_data, input_shape);

            Tensor hidden_states = qwen3::qwen3_forward_with_cache(
                input,
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

            std::cout << "Prefill complete!\n";
            std::cout << "  current_seq_len after prefill: " << kv_cache->current_seq_len << "\n";
            std::cout << "  Expected: " << input_ids_data.size() << "\n\n";

            // Check first layer cache
            std::cout << "First layer cache check:\n";
            Tensor k_cached = kv_cache->get_cached_keys(0, kv_cache->current_seq_len);
            std::cout << "  Cached K shape: [" << k_cached.shape()[0] << ", "
                      << k_cached.shape()[1] << ", " << k_cached.shape()[2] << ", "
                      << k_cached.shape()[3] << "]\n";

            // Get first token prediction
            size_t batch_size = hidden_states.shape()[0];
            size_t seq_len = hidden_states.shape()[1];
            size_t hidden_size = hidden_states.shape()[2];
            size_t vocab_size = weights.lm_head.shape()[0];

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

            std::cout << "  First predicted token: " << predicted_token << " (logit=" << max_logit << ")\n\n";
            input_ids_data.push_back(predicted_token);
        }

        // ====================================================================
        // Phase 2: Decode - Generate one token at a time
        // ====================================================================
        std::cout << "========== PHASE 2: DECODE ==========\n";

        for (int step = 0; step < 3; ++step) {
            std::cout << "\n--- Step " << (step + 1) << " ---\n";
            std::cout << "Cache seq_len BEFORE forward: " << kv_cache->current_seq_len << "\n";

            // Use ONLY the last token
            std::vector<long> new_token = {input_ids_data.back()};
            std::cout << "Input token: " << new_token[0] << "\n";

            Shape input_shape({1, 1});
            TensorL input(new_token, input_shape);

            Tensor hidden_states = qwen3::qwen3_forward_with_cache(
                input,
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

            std::cout << "Cache seq_len AFTER forward: " << kv_cache->current_seq_len << "\n";

            // Check cache again
            Tensor k_cached = kv_cache->get_cached_keys(0, kv_cache->current_seq_len);
            std::cout << "Cached K shape: [" << k_cached.shape()[0] << ", "
                      << k_cached.shape()[1] << ", " << k_cached.shape()[2] << ", "
                      << k_cached.shape()[3] << "]\n";

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

        std::cout << "\n========== FINAL TOKENS ==========\n";
        std::cout << "Generated tokens: ";
        for (size_t i = 9; i < input_ids_data.size(); ++i) {
            std::cout << input_ids_data[i] << " ";
        }
        std::cout << "\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n✗ ERROR: " << e.what() << "\n\n";
        return 1;
    }
}
