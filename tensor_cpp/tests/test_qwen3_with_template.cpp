/**
 * @file test_qwen3_with_template.cpp
 * @brief Test with correct prompt template token ids
 */

#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/ops.h"
#include <iostream>
#include <fstream>
#include <vector>
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
}

void print_stats(const Tensor& t, const std::string& name) {
    size_t total = 1;
    for (size_t i = 0; i < t.shape().ndim(); ++i) {
        total *= t.shape()[i];
    }

    float min_val = t[0];
    float max_val = t[0];
    double sum = 0.0;

    for (size_t i = 0; i < total; ++i) {
        min_val = std::min(min_val, t[i]);
        max_val = std::max(max_val, t[i]);
        sum += t[i];
    }

    std::cout << name << ": mean=" << (sum/total) << ", range=[" << min_val << ", " << max_val << "]\n";
}

int main() {
    try {
        std::string model_path = "/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors";

        std::cout << "\n============================================================\n";
        std::cout << "  C++ Qwen3 测试 (使用正确的Prompt Template)\n";
        std::cout << "============================================================\n\n";

        // Load weights
        std::cout << "Loading weights...\n";
        Qwen3Weights weights = load_qwen3_weights(model_path);
        std::cout << "Weights loaded!\n\n";

        // 应用了prompt template后的正确token ids
        // "Hello" -> [151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198]
        std::vector<long> input_ids_data = {
            151644,  // <|im_start|>
            872,     // user
            198,     // \n
            9707,    // Hello
            151645,  // <|im_end|>
            198,     // \n
            151644,  // <|im_start|>
            77091,   // assistant
            198      // \n
        };

        std::cout << "Input: 'Hello' (with prompt template)\n";
        std::cout << "Token IDs (" << input_ids_data.size() << " tokens): ";
        for (size_t i = 0; i < input_ids_data.size(); ++i) {
            std::cout << input_ids_data[i];
            if (i < input_ids_data.size() - 1) std::cout << ", ";
        }
        std::cout << "\n\n";

        // Create input tensor
        Shape input_shape({1, static_cast<long>(input_ids_data.size())});
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
        print_stats(hidden_states, "Hidden States (all positions)");

        // Get last position hidden state
        size_t batch_size = hidden_states.shape()[0];
        size_t seq_len = hidden_states.shape()[1];
        size_t hidden_size = hidden_states.shape()[2];

        std::vector<float> last_hidden(hidden_size);
        size_t last_idx = (batch_size - 1) * seq_len * hidden_size + (seq_len - 1) * hidden_size;
        for (size_t i = 0; i < hidden_size; ++i) {
            last_hidden[i] = hidden_states[last_idx + i];
        }

        print_stats(Tensor(last_hidden, Shape({static_cast<long>(hidden_size)})),
                   "Last position hidden state");

        // Save last hidden state
        {
            std::ofstream out("/tmp/cpp_hidden_template.bin", std::ios::binary);
            out.write(reinterpret_cast<const char*>(last_hidden.data()), hidden_size * sizeof(float));
            out.close();
            std::cout << "  Saved to /tmp/cpp_hidden_template.bin\n";
        }

        // Compute logits
        size_t vocab_size = weights.lm_head.shape()[0];
        std::cout << "\nComputing logits...\n";
        std::cout << "  Vocab size: " << vocab_size << "\n";

        std::vector<float> logits(vocab_size);
        #pragma omp parallel for if(vocab_size > 1000)
        for (size_t v = 0; v < vocab_size; ++v) {
            float sum = 0.0f;
            for (size_t h = 0; h < hidden_size; ++h) {
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
                      << " logit=" << logits_with_idx[i].first << "\n";
        }

        // Save logits
        {
            std::ofstream out("/tmp/cpp_logits_template.bin", std::ios::binary);
            out.write(reinterpret_cast<const char*>(logits.data()), vocab_size * sizeof(float));
            out.close();
            std::cout << "\nSaved logits to /tmp/cpp_logits_template.bin\n";
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
