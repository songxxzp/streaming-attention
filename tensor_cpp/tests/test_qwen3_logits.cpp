/**
 * @file test_qwen3_logits.cpp
 * @brief Output detailed logits for debugging comparison with transformers
 */

#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/ops.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>

using namespace tensor_cpp;
using namespace tensor_cpp::qwen3;

void save_tensor(const Tensor& t, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    size_t total = 1;
    for (size_t i = 0; i < t.shape().ndim(); ++i) {
        total *= t.shape()[i];
    }
    out.write(reinterpret_cast<const char*>(t.data()), total * sizeof(float));
    out.close();
    std::cout << "  Saved to " << filename << " (" << total << " floats)\n";
}

void print_tensor_stats(const Tensor& t, const std::string& name) {
    size_t total = 1;
    for (size_t i = 0; i < t.shape().ndim(); ++i) {
        total *= t.shape()[i];
    }

    float min_val = t[0];
    float max_val = t[0];
    double sum = 0.0;
    double sum_sq = 0.0;

    for (size_t i = 0; i < total; ++i) {
        min_val = std::min(min_val, t[i]);
        max_val = std::max(max_val, t[i]);
        sum += t[i];
        sum_sq += static_cast<double>(t[i]) * t[i];
    }

    float mean = sum / total;
    float variance = (sum_sq / total) - (mean * mean);
    float std_dev = std::sqrt(std::max(0.0f, variance));

    std::cout << name << ":\n";
    std::cout << "  Shape: " << t.shape().to_string() << "\n";
    std::cout << "  Range: [" << min_val << ", " << max_val << "]\n";
    std::cout << "  Mean: " << mean << "\n";
    std::cout << "  Std: " << std_dev << "\n";
}

int main() {
    std::cout << "\n============================================================\n";
    std::cout << "  Qwen3 Logits Debugging Test\n";
    std::cout << "============================================================\n\n";

    try {
        std::string model_path = "/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors";

        // Load weights
        std::cout << "Loading weights...\n";
        Qwen3Weights weights = load_qwen3_weights(model_path);
        std::cout << "Weights loaded!\n\n";

        // Simple test: single token "Hello" (token 9707)
        std::vector<long> input_ids_data = {9707};

        std::cout << "Input: [9707] (token for 'Hello')\n\n";

        // Create input tensor
        Shape input_shape({1, 1});  // batch=1, seq_len=1
        TensorL input(input_ids_data, input_shape);

        std::cout << "Running forward pass...\n";

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

        std::cout << "Forward complete!\n\n";

        // Print hidden states stats
        print_tensor_stats(hidden_states, "Hidden States (last layer, last token)");

        // Save hidden states
        save_tensor(hidden_states, "/tmp/cpp_hidden_states.bin");

        // Compute logits for last position
        size_t batch_size = hidden_states.shape()[0];
        size_t seq_len = hidden_states.shape()[1];
        size_t hidden_size = hidden_states.shape()[2];
        size_t vocab_size = weights.lm_head.shape()[0];

        std::cout << "\nComputing logits...\n";
        std::cout << "  Hidden size: " << hidden_size << "\n";
        std::cout << "  Vocab size: " << vocab_size << "\n";

        // Get last position hidden state
        std::vector<float> last_hidden(hidden_size);
        size_t last_idx = (batch_size - 1) * seq_len * hidden_size + (seq_len - 1) * hidden_size;
        for (size_t i = 0; i < hidden_size; ++i) {
            last_hidden[i] = hidden_states[last_idx + i];
        }

        // Save last hidden state
        {
            std::ofstream out("/tmp/cpp_last_hidden.bin", std::ios::binary);
            out.write(reinterpret_cast<const char*>(last_hidden.data()), hidden_size * sizeof(float));
            out.close();
            std::cout << "  Saved last hidden state to /tmp/cpp_last_hidden.bin\n";
        }

        // Compute logits
        std::vector<float> logits(vocab_size);
        #pragma omp parallel for if(vocab_size > 1000)
        for (size_t v = 0; v < vocab_size; ++v) {
            float sum = 0.0f;
            for (size_t h = 0; h < hidden_size; ++h) {
                // lm_head shape: [vocab_size, hidden_size]
                // We need: logits[v] = sum over h of hidden[h] * lm_head[v, h]
                // In row-major: lm_head[v, h] = lm_head[v * hidden_size + h]
                sum += last_hidden[h] * weights.lm_head[v * hidden_size + h];
            }
            logits[v] = sum;
        }

        // Find top tokens
        std::cout << "\nTop 20 tokens:\n";
        std::vector<std::pair<float, size_t>> logits_with_idx;
        for (size_t i = 0; i < vocab_size; ++i) {
            logits_with_idx.push_back({logits[i], i});
        }
        std::sort(logits_with_idx.begin(), logits_with_idx.end(), std::greater<std::pair<float, size_t>>());

        for (int i = 0; i < 20; ++i) {
            std::cout << "  [" << i << "] token=" << logits_with_idx[i].second
                      << " logit=" << std::fixed << std::setprecision(4) << logits_with_idx[i].first << "\n";
        }

        // Save logits
        {
            std::ofstream out("/tmp/cpp_logits.bin", std::ios::binary);
            out.write(reinterpret_cast<const char*>(logits.data()), vocab_size * sizeof(float));
            out.close();
            std::cout << "\nSaved logits to /tmp/cpp_logits.bin\n";
        }

        // Print logits stats
        double logit_sum = 0.0;
        double logit_sum_sq = 0.0;
        for (size_t i = 0; i < vocab_size; ++i) {
            logit_sum += logits[i];
            logit_sum_sq += static_cast<double>(logits[i]) * logits[i];
        }
        float logit_mean = logit_sum / vocab_size;
        float logit_var = (logit_sum_sq / vocab_size) - (logit_mean * logit_mean);
        float logit_std = std::sqrt(std::max(0.0f, logit_var));

        std::cout << "\nLogits statistics:\n";
        std::cout << "  Mean: " << logit_mean << "\n";
        std::cout << "  Std: " << logit_std << "\n";
        std::cout << "  Min: " << logits_with_idx.back().first << " (token " << logits_with_idx.back().second << ")\n";
        std::cout << "  Max: " << logits_with_idx.front().first << " (token " << logits_with_idx.front().second << ")\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n✗ ERROR: " << e.what() << "\n\n";
        return 1;
    }
}
