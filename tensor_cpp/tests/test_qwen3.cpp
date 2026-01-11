/**
 * @file test_qwen3.cpp
 * @brief Qwen3 model forward pass test
 */

#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/ops.h"
#include <iostream>
#include <vector>
#include <chrono>

using namespace tensor_cpp;
using namespace tensor_cpp::qwen3;

int main() {
    std::cout << "\n============================================================\n";
    std::cout << "  Qwen3 Forward Pass Test\n";
    std::cout << "============================================================\n\n";

    try {
        // Path to safetensors file
        std::string model_path = "/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors";

        // Create simple input: "Hello, world!" token IDs (just example numbers)
        // In practice, you'd use a tokenizer to get these
        std::vector<long> input_ids_data = {
            9658, 15, 1358, 35  // Example token IDs for "Hello, world!"
        };

        Shape input_shape({1, 4});  // batch_size=1, seq_len=4

        std::cout << "Input token IDs: ";
        for (auto id : input_ids_data) {
            std::cout << id << " ";
        }
        std::cout << "\n";
        std::cout << "Input shape: [batch=1, seq_len=4]\n\n";

        // Test weight loading
        std::cout << "Loading weights...\n";
        Qwen3Weights weights = load_qwen3_weights(model_path);

        std::cout << "\nWeights loaded successfully!\n";
        std::cout << "  vocab_size: " << weights.vocab_size << "\n";
        std::cout << "  hidden_size: " << weights.hidden_size << "\n";
        std::cout << "  num_layers: " << weights.num_layers << "\n";
        std::cout << "  num_attention_heads: " << weights.num_attention_heads << "\n";
        std::cout << "  num_key_value_heads: " << weights.num_key_value_heads << "\n";
        std::cout << "  head_dim: " << weights.head_dim << "\n";

        // Create input_ids tensor
        TensorL input_ids(input_ids_data, input_shape);

        std::cout << "\nRunning full forward pass...\n";
        auto start = std::chrono::high_resolution_clock::now();

        // Run full forward pass
        Tensor output = qwen3::qwen3_forward(
            input_ids,
            weights.embed_tokens,
            weights.layers,
            weights.norm_weight,
            weights.num_layers,
            weights.num_attention_heads,
            weights.num_key_value_heads,
            weights.head_dim,
            1e-6f  // rms_norm_eps
        );

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "\nForward pass completed!\n";
        std::cout << "Output shape: [";
        for (size_t i = 0; i < output.shape().ndim(); ++i) {
            std::cout << output.shape()[i];
            if (i < output.shape().ndim() - 1) std::cout << ", ";
        }
        std::cout << "]\n";

        // Show first few values of output
        std::cout << "First 5 output values: ";
        for (size_t i = 0; i < std::min(size_t(5), output.size()); ++i) {
            std::cout << output[i] << " ";
        }
        std::cout << "\n";

        std::cout << "Inference time: " << duration.count() << " ms\n";

        std::cout << "\n============================================================\n";
        std::cout << "  Test PASSED - Full forward pass working!\n";
        std::cout << "============================================================\n\n";

        std::cout << "Next steps:\n";
        std::cout << "  1. Use tokenizer to convert text to token IDs\n";
        std::cout << "  2. Implement argmax to get predicted tokens\n";
        std::cout << "  3. Add decode loop for autoregressive generation\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n✗ ERROR: " << e.what() << "\n\n";
        return 1;
    }
}
