#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/ops.h"
#include <iostream>
#include <fstream>

using namespace tensor_cpp;
using namespace tensor_cpp::qwen3;

int main() {
    try {
        // Load weights
        Qwen3Weights weights = load_qwen3_weights("/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors");
        
        // Create input embedding
        std::vector<long> input_ids_data = {9707};
        Shape input_shape({1, 1});
        TensorL input_ids(input_ids_data, input_shape);
        
        Tensor hidden = ops::embedding(input_ids, weights.embed_tokens);
        std::cout << "Input: mean=" << hidden.mean() << ", range=[" << hidden.min() << ", " << hidden.max() << "]\n\n";

        // Call qwen3_decoder_layer for layer 0
        const Qwen3LayerWeights& layer = weights.layers[0];
        size_t num_heads = weights.num_attention_heads;
        size_t kv_heads = weights.num_key_value_heads;
        size_t head_dim = weights.head_dim;
        
        auto [cos, sin] = qwen3::compute_rope_freqs(hidden.shape()[1], head_dim, 1000000.0f);
        
        Tensor output = qwen3::qwen3_decoder_layer(
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
        
        std::cout << "Layer 0 output: mean=" << output.mean() << ", range=[" << output.min() << ", " << output.max() << "]\n";
        
        // Load PyTorch reference
        std::ifstream ref("/tmp/pytorch_layer0_output.bin", std::ios::binary);
        std::vector<float> pytorch_data(output.size());
        ref.read(reinterpret_cast<char*>(pytorch_data.data()), output.size() * sizeof(float));
        
        float pytorch_mean = 0;
        for (size_t i = 0; i < pytorch_data.size(); ++i) pytorch_mean += pytorch_data[i];
        pytorch_mean /= pytorch_data.size();
        
        std::cout << "PyTorch: mean=" << pytorch_mean << "\n\n";
        
        // Check a few individual values
        std::cout << "First 10 values:\n";
        std::cout << "  C++:     ";
        for (int i = 0; i < 10; ++i) std::cout << output[i] << " ";
        std::cout << "\n  PyTorch: ";
        for (int i = 0; i < 10; ++i) std::cout << pytorch_data[i] << " ";
        std::cout << "\n";
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
