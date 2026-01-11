/**
 * @file test_save_layers.cpp
 * @brief Save outputs of each layer for debugging
 */

#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/ops.h"
#include <iostream>
#include <fstream>
#include <vector>

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

int main() {
    try {
        std::string model_path = "/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors";

        // Load weights
        std::cout << "Loading weights...\n";
        Qwen3Weights weights = load_qwen3_weights(model_path);
        std::cout << "Weights loaded!\n\n";

        // Simple test: single token "Hello" (token 9707)
        std::vector<long> input_ids_data = {9707};
        Shape input_shape({1, 1});
        TensorL input(input_ids_data, input_shape);

        // Embedding
        Tensor hidden = ops::embedding(input, weights.embed_tokens);
        save_tensor(hidden, "/tmp/cpp_embedding.bin");
        std::cout << "Saved embedding to /tmp/cpp_embedding.bin\n";

        // Precompute RoPE
        auto [cos, sin] = compute_rope_freqs(1, 128, 1000000.0f);

        // Process through layers and save outputs
        for (size_t layer_idx = 0; layer_idx < 3; ++layer_idx) {  // Only first 3 layers
            const Qwen3LayerWeights& layer = weights.layers[layer_idx];

            hidden = qwen3_decoder_layer(
                hidden, 16, 8, 128, 1e-6f,
                layer.input_layernorm_weight,
                layer.qkv_projs,
                layer.o_proj,
                layer.q_norm_weight,
                layer.k_norm_weight,
                layer.post_attention_layernorm_weight,
                layer.gate_proj,
                layer.up_proj,
                layer.down_proj,
                cos, sin
            );

            // Save layer output
            std::string filename = "/tmp/cpp_layer" + std::to_string(layer_idx) + "_output.bin";
            save_tensor(hidden, filename);
            std::cout << "Saved layer " << layer_idx << " output to " << filename << "\n";
        }

        std::cout << "\nDone! Saved first 3 layer outputs.\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
