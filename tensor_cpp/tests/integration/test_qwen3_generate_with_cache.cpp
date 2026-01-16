/**
 * @file test_qwen3_generate_with_cache.cpp
 * @brief Qwen3 autoregressive generation test with KV cache
 *
 * This test demonstrates the performance improvement from using KV cache.
 * With KV cache, each generation step only processes the new token,
 * instead of reprocessing the entire sequence.
 */

#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/ops.h"
#include "tensor_cpp/kv_cache.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdio>
#include <memory>

using namespace tensor_cpp;
using namespace tensor_cpp::qwen3;

int main() {
    std::cout << "\n============================================================\n";
    std::cout << "  Qwen3 Text Generation Test WITH KV CACHE\n";
    std::cout << "============================================================\n\n";

    try {
        std::string model_path = "/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors";

        // Load weights
        std::cout << "Loading weights...\n";
        Qwen3Weights weights = load_qwen3_weights(model_path);
        std::cout << "Weights loaded!\n\n";

        // Test prompts with correct prompt template applied
        // Format: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
        std::vector<std::pair<std::string, std::vector<long>>> tests = {
            {"Hello", {151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198}},
            {"Hello, world!", {151644, 872, 198, 9707, 11, 1879, 0, 151645, 198, 151644, 77091, 198}},
            {"The capital of France is", {151644, 872, 198, 785, 6722, 315, 9625, 374, 151645, 198, 151644, 77091, 198}}
        };

        // For each test case
        for (size_t test_idx = 0; test_idx < tests.size(); ++test_idx) {
            auto& [prompt, input_ids_data] = tests[test_idx];

            std::cout << "\n============================================================\n";
            std::cout << "  Test " << (test_idx + 1) << "\n";
            std::cout << "============================================================\n\n";

            std::cout << "Prompt: \"" << prompt << "\"\n";
            std::cout << "Input tokens (" << input_ids_data.size() << "): ";
            for (auto id : input_ids_data) {
                std::cout << id << " ";
            }
            std::cout << "\n\n";

            // Create KV cache
            std::cout << "Initializing KV cache...\n";
            auto kv_cache = std::make_unique<KVCache>(
                weights.num_layers,          // 28 layers
                1,                            // batch_size
                weights.num_key_value_heads,  // 8 KV heads
                weights.head_dim,             // 128 head_dim
                4096                          // max_seq_len
            );
            std::cout << "KV cache initialized!\n\n";

            // Generation parameters
            size_t max_new_tokens = 12;  // Generate up to 12 new tokens

            std::cout << "Generating " << max_new_tokens << " tokens...\n\n";
            std::cout << "Phase: PREFILL (processing initial prompt)\n";
            auto total_start = std::chrono::high_resolution_clock::now();

            // ====================================================================
            // Phase 1: Prefill - Process initial prompt with all tokens
            // ====================================================================
            {
                auto phase_start = std::chrono::high_resolution_clock::now();

                // Create input tensor with all initial tokens
                Shape input_shape({1, input_ids_data.size()});
                TensorL input(input_ids_data, input_shape);

                // Forward pass with KV cache (cache will be initialized)
                Tensor hidden_states = qwen3::qwen3_forward_with_cache(
                    input,
                    kv_cache.get(),
                    weights.embed_tokens,
                    weights.layers,
                    weights.norm_weight,
                    weights.lm_head,
                    weights.num_layers,
                    weights.num_attention_heads,
                    weights.num_key_value_heads,
                    weights.head_dim,
                    1e-6f
                );

                // Extract first predicted token from prefill output
                size_t batch_size = hidden_states.shape()[0];
                size_t seq_len = hidden_states.shape()[1];
                size_t hidden_size = hidden_states.shape()[2];
                size_t vocab_size = weights.lm_head.shape()[0];

                // Get last position hidden state
                size_t last_idx = (batch_size - 1) * seq_len * hidden_size + (seq_len - 1) * hidden_size;
                std::vector<float> last_hidden(hidden_size);
                for (size_t i = 0; i < hidden_size; ++i) {
                    last_hidden[i] = hidden_states[last_idx + i];
                }

                // Compute logits and predict first token
                std::vector<float> logits(vocab_size);
                #pragma omp parallel for if(vocab_size > 1000)
                for (size_t v = 0; v < vocab_size; ++v) {
                    float sum = 0.0f;
                    for (size_t h = 0; h < hidden_size; ++h) {
                        sum += last_hidden[h] * weights.lm_head[v * hidden_size + h];
                    }
                    logits[v] = sum;
                }

                // Get predicted token (greedy decoding)
                long predicted_token = 0;
                float max_logit = logits[0];
                for (size_t v = 1; v < vocab_size; ++v) {
                    if (logits[v] > max_logit) {
                        max_logit = logits[v];
                        predicted_token = static_cast<long>(v);
                    }
                }

                // Add first predicted token to input list
                input_ids_data.push_back(predicted_token);

                auto phase_end = std::chrono::high_resolution_clock::now();
                auto phase_duration = std::chrono::duration_cast<std::chrono::milliseconds>(phase_end - phase_start);

                std::cout << "  Prefill time: " << phase_duration.count() << " ms\n";
                std::cout << "  Tokens processed: " << (input_ids_data.size() - 1) << "\n";
                std::cout << "  First predicted token: " << predicted_token << " (logit=" << max_logit << ")\n";
                std::cout << "  Cache initialized: " << kv_cache->current_seq_len << " tokens\n\n";
            }

            std::cout << "Phase: DECODE (generating tokens one by one)\n";
            std::cout << "  With KV cache, each step only processes 1 new token!\n\n";

            // ====================================================================
            // Phase 2: Decode - Generate remaining tokens one at a time using KV cache
            // ====================================================================
            std::vector<long> generated_tokens;

            for (size_t step = 1; step < max_new_tokens; ++step) {  // Start from 1 since we already generated 1 token
                auto step_start = std::chrono::high_resolution_clock::now();

                // Create input tensor with ONLY the last token
                // This is the key difference: we only process the new token!
                std::vector<long> new_token = {input_ids_data.back()};
                Shape input_shape({1, 1});
                TensorL input(new_token, input_shape);

                // Forward pass with KV cache
                // Cache contains all previous tokens' K/V, so we only compute for new token
                Tensor hidden_states = qwen3::qwen3_forward_with_cache(
                    input,
                    kv_cache.get(),
                    weights.embed_tokens,
                    weights.layers,
                    weights.norm_weight,
                    weights.lm_head,
                    weights.num_layers,
                    weights.num_attention_heads,
                    weights.num_key_value_heads,
                    weights.head_dim,
                    1e-6f
                );

                // Get last token's hidden state
                size_t batch_size = hidden_states.shape()[0];
                size_t seq_len = hidden_states.shape()[1];  // Should be 1
                size_t hidden_size = hidden_states.shape()[2];
                size_t vocab_size = weights.lm_head.shape()[0];

                // Get the hidden state (only one position since seq_len=1)
                std::vector<float> last_hidden(hidden_size);
                for (size_t i = 0; i < hidden_size; ++i) {
                    last_hidden[i] = hidden_states[i];
                }

                // Compute logits
                std::vector<float> logits(vocab_size);
                #pragma omp parallel for if(vocab_size > 1000)
                for (size_t v = 0; v < vocab_size; ++v) {
                    float sum = 0.0f;
                    for (size_t h = 0; h < hidden_size; ++h) {
                        sum += last_hidden[h] * weights.lm_head[v * hidden_size + h];
                    }
                    logits[v] = sum;
                }

                // Get predicted token (greedy decoding)
                long predicted_token = 0;
                float max_logit = logits[0];
                for (size_t v = 1; v < vocab_size; ++v) {
                    if (logits[v] > max_logit) {
                        max_logit = logits[v];
                        predicted_token = static_cast<long>(v);
                    }
                }

                auto step_end = std::chrono::high_resolution_clock::now();
                auto step_duration = std::chrono::duration_cast<std::chrono::milliseconds>(step_end - step_start);

                std::cout << "Step " << std::setw(2) << (step + 1) << ": "  // step + 1 to show 2, 3, ... (1 was prefill)
                          << "token=" << std::setw(6) << predicted_token
                          << "  logit=" << std::fixed << std::setprecision(2) << max_logit
                          << "  time=" << std::setw(4) << step_duration.count() << " ms"
                          << "  (cached_tokens=" << kv_cache->current_seq_len << ")\n";

                // Append predicted token
                input_ids_data.push_back(predicted_token);
                generated_tokens.push_back(predicted_token);

                // Check for EOS token
                if (predicted_token == 151645) {  // EOS token
                    std::cout << "\n→ EOS token reached\n";
                    break;
                }

                // Limit sequence length
                if (input_ids_data.size() > 50) {
                    std::cout << "\n→ Max sequence length reached\n";
                    break;
                }
            }

            auto total_end = std::chrono::high_resolution_clock::now();
            auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);

            std::cout << "\n------------------------------------------------------------\n";
            std::cout << "Generation Summary:\n";
            std::cout << "  Total time: " << total_duration.count() << " ms\n";
            std::cout << "  Tokens generated: " << generated_tokens.size() << "\n";
            if (generated_tokens.size() > 0) {
                std::cout << "  Average time per token: " << (total_duration.count() / generated_tokens.size()) << " ms\n";
                std::cout << "  Tokens per second: " << std::fixed << std::setprecision(2)
                          << (1000.0 * generated_tokens.size() / total_duration.count()) << "\n";
            }
            std::cout << "  Final cache size: " << kv_cache->current_seq_len << " tokens\n";

            // Decode output using Python
            std::cout << "\n------------------------------------------------------------\n";
            std::cout << "  Decoding output...\n";
            std::cout << "------------------------------------------------------------\n";

            std::stringstream token_ss;
            for (size_t i = 0; i < input_ids_data.size(); ++i) {
                if (i > 0) token_ss << ", ";
                token_ss << input_ids_data[i];
            }

            std::string decode_cmd = "python3 << 'PYEOF'\n"
                "try:\n"
                "    from tokenizers import Tokenizer\n"
                "    tokenizer = Tokenizer.from_file('/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/tokenizer.json')\n"
                "    tokens = [" + token_ss.str() + "]\n"
                "    text = tokenizer.decode(tokens)\n"
                "    print('OUTPUT:', repr(text))\n"
                "except Exception as e:\n"
                "    print('ERROR:', str(e))\n"
                "PYEOF";

            FILE* decode_pipe = popen(decode_cmd.c_str(), "r");
            if (decode_pipe) {
                char decode_buffer[2048];
                while (fgets(decode_buffer, sizeof(decode_buffer), decode_pipe)) {
                    std::cout << decode_buffer;
                }
                pclose(decode_pipe);
            }

            std::cout << "\n";
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n✗ ERROR: " << e.what() << "\n\n";
        return 1;
    }
}
