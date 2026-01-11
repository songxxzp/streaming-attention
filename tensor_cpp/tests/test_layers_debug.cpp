/**
 * @file test_layers_debug.cpp
 * @brief Layer-by-layer comparison with PyTorch to find divergence
 */

#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/ops.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <numeric>
#include <algorithm>

using namespace tensor_cpp;
using namespace tensor_cpp::qwen3;

float compute_std(const float* data, size_t size, float mean) {
    float sum_sq_diff = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float diff = data[i] - mean;
        sum_sq_diff += diff * diff;
    }
    return std::sqrt(sum_sq_diff / size);
}

void compare_with_reference(const Tensor& cpp_tensor, const char* filename, const char* name) {
    // Load PyTorch reference (binary format, no numpy header)
    std::ifstream ref(filename, std::ios::binary);
    if (!ref) {
        std::cerr << "  Warning: Could not load " << filename << "\n";
        return;
    }

    // Check file size
    ref.seekg(0, std::ios::end);
    std::streamsize file_size = ref.tellg();
    ref.seekg(0, std::ios::beg);

    size_t expected_bytes = cpp_tensor.size() * sizeof(float);
    std::cerr << "  DEBUG: " << name << " - file size=" << file_size << ", expected=" << expected_bytes << "\n";

    std::vector<float> pytorch_data(cpp_tensor.size());
    ref.read(reinterpret_cast<char*>(pytorch_data.data()), cpp_tensor.size() * sizeof(float));

    // Verify we read the correct amount
    std::streamsize bytes_read = ref.gcount();
    std::cerr << "  DEBUG: " << name << " - bytes read=" << bytes_read << "\n";

    if (bytes_read != expected_bytes) {
        std::cerr << "  ERROR: Only read " << bytes_read << " bytes, expected " << expected_bytes << "\n";
        return;
    }

    // Compare
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    size_t count_large = 0;

    for (size_t i = 0; i < cpp_tensor.size(); ++i) {
        float diff = std::abs(cpp_tensor[i] - pytorch_data[i]);
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
        if (diff > 0.01f) count_large++;
    }

    float mean_diff = sum_diff / cpp_tensor.size();
    float pytorch_mean = std::accumulate(pytorch_data.begin(), pytorch_data.end(), 0.0f) / pytorch_data.size();

    std::cout << "  " << name << ":\n";
    std::cout << "    C++:      mean=" << cpp_tensor.mean() << ", range=[" << cpp_tensor.min() << ", " << cpp_tensor.max() << "]\n";
    std::cout << "    PyTorch:  mean=" << pytorch_mean << ", range=[" << *std::min_element(pytorch_data.begin(), pytorch_data.end()) << ", " << *std::max_element(pytorch_data.begin(), pytorch_data.end()) << "]\n";
    std::cout << "    Max diff: " << max_diff << ", Mean diff: " << mean_diff << "\n";
    std::cout << "    Elements > 0.01: " << count_large << "/" << cpp_tensor.size() << "\n";

    if (max_diff > 1.0f) {
        std::cout << "    ⚠️  WARNING: Large difference!\n";
    } else if (max_diff > 0.1f) {
        std::cout << "    ⚠️  Moderate difference\n";
    } else {
        std::cout << "    ✓ Good\n";
    }
}

int main() {
    std::cout << "\n============================================================\n";
    std::cout << "  Layer-by-Layer Debug Test\n";
    std::cout << "============================================================\n\n";

    try {
        // Load weights
        std::cout << "Loading weights...\n";
        std::string model_path = "/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors";
        Qwen3Weights weights = load_qwen3_weights(model_path);
        std::cout << "Weights loaded!\n\n";

        // Single token
        long input_token_id = 9707;  // "Hello"
        std::vector<long> input_ids_data = {input_token_id};
        Shape input_shape({1, 1});
        TensorL input_ids(input_ids_data, input_shape);

        // Embedding
        Tensor hidden = ops::embedding(input_ids, weights.embed_tokens);
        std::cout << "Input embedding: mean=" << hidden.mean() << "\n";

        // Compare input
        compare_with_reference(hidden, "/tmp/pytorch_input.bin", "Input embedding");

        // Process each layer
        for (size_t layer_idx = 0; layer_idx < weights.num_layers; ++layer_idx) {
            std::cout << "\n============================================================\n";
            std::cout << "Layer " << layer_idx << "\n";
            std::cout << "============================================================\n";

            const Qwen3LayerWeights& layer = weights.layers[layer_idx];

            // Process layer using qwen3_decoder_layer
            size_t num_heads = weights.num_attention_heads;
            size_t kv_heads = weights.num_key_value_heads;
            size_t head_dim = weights.head_dim;

            // Compute RoPE frequencies
            auto [cos, sin] = qwen3::compute_rope_freqs(hidden.shape()[1], head_dim, 1000000.0f);

            // Run full layer
            Tensor layer_output = qwen3::qwen3_decoder_layer(
                hidden,
                num_heads,
                kv_heads,
                head_dim,
                1e-6f,
                layer.input_layernorm_weight,
                layer.qkv_projs,
                layer.o_proj,
                layer.q_norm_weight,
                layer.k_norm_weight,
                layer.post_attention_layernorm_weight,
                layer.gate_proj,
                layer.up_proj,
                layer.down_proj,
                cos,
                sin
            );

            // Compare layer output
            char output_filename[128];
            snprintf(output_filename, sizeof(output_filename), "/tmp/pytorch_layer%d_output.bin", (int)layer_idx);
            compare_with_reference(layer_output, output_filename, "Layer output");

            // Update hidden for next layer
            hidden = layer_output;

            // Print summary for first few layers
            if (layer_idx < 3 || layer_idx % 7 == 0 || layer_idx == 27) {
                std::cout << "    Output: mean=" << hidden.mean() << ", range=[" << hidden.min() << ", " << hidden.max() << "]\n";
            }
        }

        // Final norm
        std::cout << "\n============================================================\n";
        std::cout << "Final Layer Norm\n";
        std::cout << "============================================================\n";

        hidden = ops::rms_norm(hidden, &weights.norm_weight, 1e-6f);
        compare_with_reference(hidden, "/tmp/pytorch_final_norm.bin", "Final norm");

        std::cout << "\n============================================================\n";
        std::cout << "Test Complete!\n";
        std::cout << "============================================================\n\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n✗ ERROR: " << e.what() << "\n\n";
        return 1;
    }
}
