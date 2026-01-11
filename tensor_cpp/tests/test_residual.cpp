#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/ops.h"
#include <iostream>
#include <fstream>

using namespace tensor_cpp;
using namespace tensor_cpp::qwen3;

int main() {
    try {
        std::cout << "Testing residual connection computation...\n\n";
        
        // Load weights
        Qwen3Weights weights = load_qwen3_weights("/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors");
        
        // Create input
        std::vector<long> input_ids_data = {9707};
        Shape input_shape({1, 1});
        TensorL input_ids(input_ids_data, input_shape);
        
        Tensor hidden = ops::embedding(input_ids, weights.embed_tokens);
        std::cout << "Step 1 - Input embedding:\n";
        std::cout << "  Shape: [" << hidden.shape()[0] << ", " << hidden.shape()[1] << ", " << hidden.shape()[2] << "]\n";
        std::cout << "  mean=" << hidden.mean() << ", range=[" << hidden.min() << ", " << hidden.max() << "]\n\n";
        
        const Qwen3LayerWeights& layer = weights.layers[0];
        size_t num_heads = weights.num_attention_heads;
        size_t kv_heads = weights.num_key_value_heads;
        size_t head_dim = weights.head_dim;
        
        // Step 2: Input layernorm
        Tensor hidden_norm = ops::rms_norm(hidden, &layer.input_layernorm_weight, 1e-6f);
        std::cout << "Step 2 - Input layernorm:\n";
        std::cout << "  mean=" << hidden_norm.mean() << ", range=[" << hidden_norm.min() << ", " << hidden_norm.max() << "]\n\n";
        
        // Step 3: Run attention
        auto [cos, sin] = qwen3::compute_rope_freqs(hidden.shape()[1], head_dim, 1000000.0f);
        
        Tensor attn_out = qwen3::qwen3_attention(
            hidden_norm,
            num_heads,
            kv_heads,
            head_dim,
            layer.qkv_projs,
            layer.o_proj,
            layer.q_norm_weight,
            layer.k_norm_weight,
            cos,
            sin,
            false
        );
        
        std::cout << "Step 3 - Attention output:\n";
        std::cout << "  Shape: [" << attn_out.shape()[0] << ", " << attn_out.shape()[1] << ", " << attn_out.shape()[2] << "]\n";
        std::cout << "  mean=" << attn_out.mean() << ", range=[" << attn_out.min() << ", " << attn_out.max() << "]\n\n";
        
        // Step 4: First residual
        Tensor hidden_after_attn = hidden + attn_out;
        std::cout << "Step 4 - After first residual (input + attn_out):\n";
        std::cout << "  mean=" << hidden_after_attn.mean() << ", range=[" << hidden_after_attn.min() << ", " << hidden_after_attn.max() << "]\n\n";
        
        // Compare with PyTorch reference
        std::ifstream ref("/tmp/pytorch_layer0_after_residual.bin", std::ios::binary);
        if (ref) {
            std::vector<float> pytorch_data(hidden_after_attn.size());
            ref.read(reinterpret_cast<char*>(pytorch_data.data()), hidden_after_attn.size() * sizeof(float));
            
            float max_diff = 0;
            for (size_t i = 0; i < hidden_after_attn.size(); ++i) {
                float diff = std::abs(hidden_after_attn[i] - pytorch_data[i]);
                max_diff = std::max(max_diff, diff);
            }
            
            float pytorch_mean = 0;
            for (size_t i = 0; i < pytorch_data.size(); ++i) pytorch_mean += pytorch_data[i];
            pytorch_mean /= pytorch_data.size();
            
            std::cout << "  PyTorch: mean=" << pytorch_mean << "\n";
            std::cout << "  Max diff: " << max_diff << "\n";
            
            if (max_diff < 0.1) {
                std::cout << "  ✓ Good match\n";
            } else {
                std::cout << "  ✗ MISMATCH!\n";
            }
        }
        
        std::cout << "\n";
        
        // Step 5: Post-attention layernorm
        Tensor post_norm = ops::rms_norm(hidden_after_attn, &layer.post_attention_layernorm_weight, 1e-6f);
        std::cout << "Step 5 - Post-attention layernorm:\n";
        std::cout << "  mean=" << post_norm.mean() << ", range=[" << post_norm.min() << ", " << post_norm.max() << "]\n\n";
        
        // Step 6: MLP
        Tensor mlp_out = qwen3::qwen3_mlp(post_norm, layer.gate_proj, layer.up_proj, layer.down_proj);
        std::cout << "Step 6 - MLP output:\n";
        std::cout << "  mean=" << mlp_out.mean() << ", range=[" << mlp_out.min() << ", " << mlp_out.max() << "]\n\n";
        
        // Step 7: Second residual
        Tensor output = hidden_after_attn + mlp_out;
        std::cout << "Step 7 - Final output (after second residual):\n";
        std::cout << "  mean=" << output.mean() << ", range=[" << output.min() << ", " << output.max() << "]\n";
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
