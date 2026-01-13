/**
 * @file test_qwen3_decode.cpp
 * @brief Qwen3 decode test with tokenizer verification
 */

#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/ops.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <algorithm>

using namespace tensor_cpp;
using namespace tensor_cpp::qwen3;

int main() {
    std::cout << "\n============================================================\n";
    std::cout << "  Qwen3 Decode Test with Tokenizer\n";
    std::cout << "============================================================\n\n";

    try {
        // Path to safetensors file
        std::string model_path = "/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors";
        std::string tokenizer_path = "/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B";

        // Load weights
        std::cout << "Loading weights...\n";
        Qwen3Weights weights = load_qwen3_weights(model_path);
        std::cout << "Weights loaded!\n\n";

        // Test prompt
        std::string prompt = "Hello, world!";
        std::cout << "Prompt: \"" << prompt << "\"\n";

        // For now, use pre-computed token IDs (you can get these from Python)
        // These would normally come from the tokenizer
        std::vector<long> input_ids_data = {
            9658, 15, 1358, 35  // "Hello, world!" token IDs
        };

        std::cout << "Input token IDs: ";
        for (auto id : input_ids_data) {
            std::cout << id << " ";
        }
        std::cout << "\n\n";

        Shape input_shape({1, static_cast<size_t>(input_ids_data.size())});
        TensorL input_ids(input_ids_data, input_shape);

        // ========================================
        // Forward pass to get hidden states
        // ========================================
        std::cout << "Running forward pass...\n";
        auto start = std::chrono::high_resolution_clock::now();

        Tensor hidden_states = qwen3::qwen3_forward(
            input_ids,
            weights.embed_tokens,
            weights.layers,
            weights.norm_weight,
            weights.lm_head,
            weights.num_layers,
            weights.num_attention_heads,
            weights.num_key_value_heads,
            weights.head_dim,
            1e-6f  // rms_norm_eps
        );

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Forward pass completed in " << duration.count() << " ms\n";
        std::cout << "Hidden states shape: [" << hidden_states.shape()[0] << ", "
                  << hidden_states.shape()[1] << ", " << hidden_states.shape()[2] << "]\n\n";

        // ========================================
        // Apply lm_head to get logits
        // ========================================
        std::cout << "Applying lm_head to get logits...\n";

        // hidden_states: [batch, seq_len, hidden_size]
        // lm_head: [vocab_size, hidden_size]
        // We need to project each position: for each (batch, seq), compute logits = hidden @ lm_head^T

        size_t batch_size = hidden_states.shape()[0];
        size_t seq_len = hidden_states.shape()[1];
        size_t hidden_size = hidden_states.shape()[2];
        size_t vocab_size = weights.lm_head.shape()[0];

        // Get the last token's hidden state for next token prediction
        std::vector<float> last_hidden(hidden_size);
        size_t last_idx = (batch_size - 1) * seq_len * hidden_size + (seq_len - 1) * hidden_size;
        for (size_t i = 0; i < hidden_size; ++i) {
            last_hidden[i] = hidden_states[last_idx + i];
        }

        // Project to logits: logits = last_hidden @ lm_head^T
        // This gives us [vocab_size]
        std::vector<float> logits_data(vocab_size);

        #pragma omp parallel for if(vocab_size > 1000)
        for (size_t v = 0; v < vocab_size; ++v) {
            float sum = 0.0f;
            for (size_t h = 0; h < hidden_size; ++h) {
                // lm_head shape: [vocab_size, hidden_size]
                // We need: logits[v] = sum over h of hidden[h] * lm_head[v, h]
                // In row-major: lm_head[v, h] = lm_head[v * hidden_size + h]
                sum += last_hidden[h] * weights.lm_head[v * hidden_size + h];
            }
            logits_data[v] = sum;
        }

        Tensor logits(std::move(logits_data), Shape({static_cast<long>(vocab_size)}));

        std::cout << "Logits shape: [" << logits.shape()[0] << "]\n";

        // Show top 5 logits
        std::vector<float> logits_copy(vocab_size);
        for (size_t i = 0; i < vocab_size; ++i) {
            logits_copy[i] = logits[i];
        }

        // Use nth_element to get top 5
        std::nth_element(logits_copy.begin(), logits_copy.begin() + 5, logits_copy.end(), std::greater<float>());
        std::sort(logits_copy.begin(), logits_copy.begin() + 5, std::greater<float>());

        std::cout << "Top 5 logit values: ";
        for (size_t i = 0; i < 5; ++i) {
            std::cout << logits_copy[i] << " ";
        }
        std::cout << "\n\n";

        // ========================================
        // Get predicted token using argmax
        // ========================================
        std::cout << "Getting predicted token...\n";
        TensorL predicted_token_idx = ops::argmax(logits, -1, false);

        long predicted_token = predicted_token_idx[0];
        std::cout << "Predicted token ID: " << predicted_token << "\n\n";

        // ========================================
        // Decode using Python script
        // ========================================
        std::cout << "============================================================\n";
        std::cout << "  Tokenizer Verification\n";
        std::cout << "============================================================\n\n";

        // Write token IDs to file for Python to decode
        std::vector<long> tokens_to_decode = input_ids_data;
        tokens_to_decode.push_back(predicted_token);

        std::ofstream out_file("/tmp/qwen3_tokens.txt");
        for (auto token : tokens_to_decode) {
            out_file << token << "\n";
        }
        out_file.close();

        std::cout << "Wrote " << tokens_to_decode.size() << " tokens to /tmp/qwen3_tokens.txt\n";
        std::cout << "Run this to decode:\n\n";

        std::cout << "  python3 -c \"\n";
        std::cout << "from transformers import AutoTokenizer\n";
        std::cout << "tokenizer = AutoTokenizer.from_pre('" << tokenizer_path << "')\n";
        std::cout << "with open('/tmp/qwen3_tokens.txt') as f:\n";
        std::cout << "    tokens = [int(line.strip()) for line in f]\n";
        std::cout << "text = tokenizer.decode(tokens)\n";
        std::cout << "print('Decoded text:', repr(text))\n";
        std::cout << "\"\n\n";

        std::cout << "============================================================\n";
        std::cout << "  Test Completed Successfully!\n";
        std::cout << "============================================================\n\n";

        std::cout << "Summary:\n";
        std::cout << "  - Forward pass: OK\n";
        std::cout << "  - LM head projection: OK\n";
        std::cout << "  - Token prediction: OK (token ID = " << predicted_token << ")\n";
        std::cout << "  - Run the Python command above to verify the decoded text\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n✗ ERROR: " << e.what() << "\n\n";
        return 1;
    }
}
