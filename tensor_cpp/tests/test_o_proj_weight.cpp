#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/ops.h"
#include <iostream>
#include <fstream>

using namespace tensor_cpp;
using namespace tensor_cpp::qwen3;

int main() {
    try {
        // Load weights
        Qwen3Weights weights = load_qwen3_weights("/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors");
        
        const Qwen3LayerWeights& layer = weights.layers[0];
        
        std::cout << "C++ loaded o_proj:\n";
        std::cout << "  Shape: [" << layer.o_proj.shape()[0] << ", " << layer.o_proj.shape()[1] << "]\n";
        std::cout << "  Size: " << layer.o_proj.size() << " elements\n";
        std::cout << "  Mean: " << layer.o_proj.mean() << "\n";
        std::cout << "  First 10 values: ";
        for (int i = 0; i < 10; ++i) {
            std::cout << layer.o_proj[i] << " ";
        }
        std::cout << "\n\n";
        
        // Load PyTorch reference
        std::ifstream ref("/tmp/pytorch_o_proj_weight.bin", std::ios::binary);
        std::vector<float> pytorch_data(layer.o_proj.size());
        ref.read(reinterpret_cast<char*>(pytorch_data.data()), layer.o_proj.size() * sizeof(float));
        
        float pytorch_mean = 0;
        for (size_t i = 0; i < pytorch_data.size(); ++i) pytorch_mean += pytorch_data[i];
        pytorch_mean /= pytorch_data.size();
        
        std::cout << "PyTorch o_proj:\n";
        std::cout << "  Mean: " << pytorch_mean << "\n";
        std::cout << "  First 10 values: ";
        for (int i = 0; i < 10; ++i) {
            std::cout << pytorch_data[i] << " ";
        }
        std::cout << "\n\n";
        
        // Compare
        float max_diff = 0;
        for (size_t i = 0; i < layer.o_proj.size(); ++i) {
            float diff = std::abs(layer.o_proj[i] - pytorch_data[i]);
            max_diff = std::max(max_diff, diff);
        }
        
        std::cout << "Max difference: " << max_diff << "\n";
        
        if (max_diff < 0.001) {
            std::cout << "✓ Weights loaded correctly!\n";
        } else {
            std::cout << "✗ Weight loading discrepancy!\n";
        }
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
