#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/ops.h"
#include <iostream>
#include <fstream>

using namespace tensor_cpp;
using namespace tensor_cpp::qwen3;

int main() {
    try {
        std::cout << "Testing o_proj (output projection)...\n\n";

        // Load weights
        Qwen3Weights weights = load_qwen3_weights("/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors");
        const Qwen3LayerWeights& layer = weights.layers[0];

        // Load the attention output (before o_proj) from test_align_qwen3
        std::ifstream ref("/tmp/cpp_attn_output.bin", std::ios::binary);
        
        // Get file size
        ref.seekg(0, std::ios::end);
        size_t num_floats = ref.tellg() / sizeof(float);
        ref.seekg(0, std::ios::beg);

        std::vector<float> attn_data(num_floats);
        ref.read(reinterpret_cast<char*>(attn_data.data()), num_floats * sizeof(float));

        // Create tensor: [batch, seq, hidden]
        Tensor attn_output(std::move(attn_data), Shape({1, 1, 1024}));

        std::cout << "Input (before o_proj):\n";
        std::cout << "  Shape: [" << attn_output.shape()[0] << ", " << attn_output.shape()[1] << ", " << attn_output.shape()[2] << "]\n";
        std::cout << "  Mean: " << attn_output.mean() << "\n";
        std::cout << "  Range: [" << attn_output.min() << ", " << attn_output.max() << "]\n\n";

        // Apply o_proj
        Tensor result = ops::linear(attn_output, layer.o_proj);

        std::cout << "Output (after o_proj):\n";
        std::cout << "  Shape: [" << result.shape()[0] << ", " << result.shape()[1] << ", " << result.shape()[2] << "]\n";
        std::cout << "  Mean: " << result.mean() << "\n";
        std::cout << "  Range: [" << result.min() << ", " << result.max() << "]\n\n";

        // Load PyTorch reference if available
        std::ifstream pytorch_ref("/tmp/pytorch_layer0_attn_out.bin", std::ios::binary);
        if (pytorch_ref) {
            std::vector<float> pytorch_data(result.size());
            pytorch_ref.read(reinterpret_cast<char*>(pytorch_data.data()), result.size() * sizeof(float));

            // Compare
            float max_diff = 0.0f;
            for (size_t i = 0; i < result.size(); ++i) {
                float diff = std::abs(result[i] - pytorch_data[i]);
                max_diff = std::max(max_diff, diff);
            }

            float pytorch_mean = 0.0f;
            for (size_t i = 0; i < pytorch_data.size(); ++i) pytorch_mean += pytorch_data[i];
            pytorch_mean /= pytorch_data.size();

            std::cout << "PyTorch reference:\n";
            std::cout << "  Mean: " << pytorch_mean << "\n";
            std::cout << "  Max diff: " << max_diff << "\n";

            if (max_diff < 0.001) {
                std::cout << "  ✓ Excellent match!\n";
            } else if (max_diff < 0.01) {
                std::cout << "  ✓ Good match\n";
            } else if (max_diff < 0.1) {
                std::cout << "  ⚠️  Moderate difference\n";
            } else {
                std::cout << "  ✗ POOR MATCH\n";
            }
        } else {
            std::cout << "No PyTorch reference found at /tmp/pytorch_layer0_attn_out.bin\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
