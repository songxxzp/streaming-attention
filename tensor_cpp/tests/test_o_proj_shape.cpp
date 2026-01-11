#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/ops.h"
#include <iostream>

using namespace tensor_cpp;
using namespace tensor_cpp::qwen3;

int main() {
    try {
        std::cout << "Checking o_proj weight shape...\n\n";

        // Load weights
        Qwen3Weights weights = load_qwen3_weights("/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors");
        const Qwen3LayerWeights& layer = weights.layers[0];

        std::cout << "Layer 0 o_proj:\n";
        std::cout << "  Shape: [" << layer.o_proj.shape()[0] << ", " << layer.o_proj.shape()[1] << "]\n";
        std::cout << "  Size: " << layer.o_proj.size() << " elements\n";
        std::cout << "  Expected shape for o_proj: [1024, 1024]\n\n";

        // Try to create a simple test input
        std::vector<float> test_data(1024, 0.1f);
        Tensor test_input(test_data, Shape({1, 1024}));  // [1, 1024]

        std::cout << "Test input shape: [" << test_input.shape()[0] << ", " << test_input.shape()[1] << "]\n";
        std::cout << "  in_features (last dim): " << test_input.shape()[1] << "\n\n";

        // Try to apply o_proj
        try {
            Tensor result = ops::linear(test_input, layer.o_proj);
            std::cout << "SUCCESS! Result shape: [" << result.shape()[0] << ", " << result.shape()[1] << "]\n";
        } catch (const std::exception& e) {
            std::cout << "ERROR: " << e.what() << "\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
