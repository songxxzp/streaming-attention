/**
 * @file test_qwen3_generate.cpp
 * @brief Qwen3 autoregressive generation test with tokenizer output
 */

#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/ops.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdio>

using namespace tensor_cpp;
using namespace tensor_cpp::qwen3;

int main() {
    std::cout << "\n============================================================\n";
    std::cout << "  Qwen3 Text Generation Test\n";
    std::cout << "============================================================\n\n";

    try {
        std::string model_path = "/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors";

        // Load weights
        std::cout << "Loading weights...\n";
        Qwen3Weights weights = load_qwen3_weights(model_path);
        std::cout << "Weights loaded!\n\n";

        // Test prompts with pre-computed token IDs (from current tokenizer)
        std::vector<std::pair<std::string, std::vector<long>>> tests = {
            {"Hello, world!", {9707, 11, 1879, 0}},
            {"The capital of France is", {785, 6722, 315, 9625, 374}},
            {"What is machine learning?", {3838, 374, 5662, 6832, 30}}
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

            // Generation parameters
            size_t max_new_tokens = 12;  // Generate up to 12 new tokens

            std::cout << "Generating " << max_new_tokens << " tokens...\n\n";
            auto total_start = std::chrono::high_resolution_clock::now();

            // Autoregressive generation loop
            for (size_t step = 0; step < max_new_tokens; ++step) {
                auto step_start = std::chrono::high_resolution_clock::now();

                // Create input tensor
                Shape input_shape({1, input_ids_data.size()});
                TensorL input(input_ids_data, input_shape);

                // Forward pass
                Tensor hidden_states = qwen3::qwen3_forward(
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

                // Get last token's hidden state
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

                // Compute logits for last position
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

                std::cout << "Step " << std::setw(2) << (step + 1) << ": "
                          << "token=" << std::setw(6) << predicted_token
                          << "  logit=" << std::fixed << std::setprecision(2) << max_logit
                          << "  time=" << std::setw(4) << step_duration.count() << " ms\n";

                // Append predicted token
                input_ids_data.push_back(predicted_token);

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
            std::cout << "  Tokens generated: " << max_new_tokens << "\n";
            std::cout << "  Average time per token: " << (total_duration.count() / max_new_tokens) << " ms\n";
            std::cout << "  Tokens per second: " << std::fixed << std::setprecision(2)
                      << (1000.0 * max_new_tokens / total_duration.count()) << "\n";

            // Decode output using Python
            std::cout << "\n------------------------------------------------------------\n";
            std::cout << "  Decoding output...\n";
            std::cout << "------------------------------------------------------------\n";

            std::stringstream token_ss;
            for (size_t i = 0; i < input_ids_data.size(); ++i) {
                if (i > 0) token_ss << " ";
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
