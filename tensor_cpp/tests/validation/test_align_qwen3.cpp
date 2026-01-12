/**
 * @file test_align_qwen3.cpp
 * @brief Alignment test for Qwen3 implementation - compare with PyTorch
 */

#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/ops.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

using namespace tensor_cpp;
using namespace tensor_cpp::qwen3;

// Helper function to save tensor to numpy file
void save_to_npy(const float* data, size_t size, const char* filename) {
    // Simple numpy save (for 1D arrays)
    std::ofstream out(filename, std::ios::binary);
    out.write(reinterpret_cast<const char*>(data), size * sizeof(float));
    std::cout << "  Saved to " << filename << " (" << size << " floats)\n";
}

// Helper function to compare arrays
void compare_arrays(const float* a, const float* b, size_t size, const char* name, float tol = 1e-4) {
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    size_t count = 0;

    for (size_t i = 0; i < size; ++i) {
        float diff = std::abs(a[i] - b[i]);
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
        if (diff > tol) count++;
    }

    std::cout << "  " << name << ":\n";
    std::cout << "    Max diff: " << max_diff << "\n";
    std::cout << "    Mean diff: " << (sum_diff / size) << "\n";
    std::cout << "    Elements > " << tol << ": " << count << "/" << size << "\n";

    if (max_diff > 0.1f) {
        std::cout << "    ⚠️  WARNING: Large difference detected!\n";
    } else if (max_diff > 0.01f) {
        std::cout << "    ⚠️  NOTE: Moderate difference\n";
    } else {
        std::cout << "    ✓ Good match\n";
    }
}

// Helper function to compute standard deviation
float compute_std(const float* data, size_t size, float mean) {
    float sum_sq_diff = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float diff = data[i] - mean;
        sum_sq_diff += diff * diff;
    }
    return std::sqrt(sum_sq_diff / size);
}

int main() {
    std::cout << "\n============================================================\n";
    std::cout << "  Qwen3 Alignment Test (C++ vs PyTorch)\n";
    std::cout << "============================================================\n\n";

    try {
        // Load weights
        std::cout << "Loading weights...\n";
        std::string model_path = "/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors";
        Qwen3Weights weights = load_qwen3_weights(model_path);
        std::cout << "Weights loaded!\n\n";

        // Test with single token
        long input_token_id = 9707;  // "Hello"
        std::cout << "Test token: " << input_token_id << " (\"Hello\")\n\n";

        // ============================================================
        // Step 1: Embedding
        // ============================================================
        std::cout << "Step 1: Token Embedding\n";
        std::vector<long> input_ids_data = {input_token_id};
        Shape input_shape({1, 1});
        TensorL input_ids(input_ids_data, input_shape);

        Tensor hidden_states = ops::embedding(input_ids, weights.embed_tokens);
        std::cout << "  Shape: [" << hidden_states.shape()[0] << ", "
                  << hidden_states.shape()[1] << ", " << hidden_states.shape()[2] << "]\n";
        std::cout << "  Range: [" << hidden_states.min() << ", " << hidden_states.max() << "]\n";
        std::cout << "  Mean: " << hidden_states.mean() << ", Std: " << compute_std(hidden_states.data(), hidden_states.size(), hidden_states.mean()) << "\n";
        save_to_npy(hidden_states.data(), hidden_states.size(), "/tmp/cpp_embedding.bin");

        // Load PyTorch reference and compare
        {
            std::ifstream ref("/tmp/pytorch_embedding.bin", std::ios::binary);
            std::vector<float> pytorch_data(hidden_states.size());
            ref.read(reinterpret_cast<char*>(pytorch_data.data()), pytorch_data.size() * sizeof(float));
            compare_arrays(hidden_states.data(), pytorch_data.data(), hidden_states.size(), "vs PyTorch embedding");
        }

        // ============================================================
        // Step 2: Input LayerNorm
        // ============================================================
        std::cout << "\nStep 2: Input LayerNorm (Layer 0)\n";
        Tensor normed = ops::rms_norm(hidden_states,
            &weights.layers[0].input_layernorm_weight, 1e-6f);
        std::cout << "  Range: [" << normed.min() << ", " << normed.max() << "]\n";
        std::cout << "  Mean: " << normed.mean() << ", Std: " << compute_std(normed.data(), normed.size(), normed.mean()) << "\n";
        save_to_npy(normed.data(), normed.size(), "/tmp/cpp_after_input_norm.bin");

        {
            std::ifstream ref("/tmp/pytorch_after_input_norm.bin", std::ios::binary);
            std::vector<float> pytorch_data(normed.size());
            ref.read(reinterpret_cast<char*>(pytorch_data.data()), pytorch_data.size() * sizeof(float));
            compare_arrays(normed.data(), pytorch_data.data(), normed.size(), "vs PyTorch after norm");
        }

        // ============================================================
        // Step 3: QKV Projections
        // ============================================================
        std::cout << "\nStep 3: QKV Projections (Layer 0)\n";

        // Split QKV from layer 0
        const Qwen3LayerWeights& layer = weights.layers[0];
        size_t batch = 1, seq = 1, q_heads = 16, kv_heads = 8, head_dim = 128;
        size_t hidden_size = 1024;
        size_t q_size = 16 * 128;  // num_heads * head_dim
        size_t k_size = 8 * 128;
        size_t v_size = 8 * 128;

        // Extract Q, K, V (same logic as in qwen3_decoder_layer)
        std::vector<float> q_data(q_size * hidden_size);
        std::vector<float> k_data(k_size * hidden_size);
        std::vector<float> v_data(v_size * hidden_size);

        for (size_t row = 0; row < q_size; ++row) {
            for (size_t col = 0; col < hidden_size; ++col) {
                q_data[row * hidden_size + col] = layer.qkv_projs[row * hidden_size + col];
            }
        }
        for (size_t row = 0; row < k_size; ++row) {
            for (size_t col = 0; col < hidden_size; ++col) {
                k_data[row * hidden_size + col] = layer.qkv_projs[(q_size + row) * hidden_size + col];
            }
        }
        for (size_t row = 0; row < v_size; ++row) {
            for (size_t col = 0; col < hidden_size; ++col) {
                v_data[row * hidden_size + col] = layer.qkv_projs[(q_size + k_size + row) * hidden_size + col];
            }
        }

        Tensor q_proj(std::move(q_data), Shape({static_cast<long>(q_size), static_cast<long>(hidden_size)}));
        Tensor k_proj(std::move(k_data), Shape({static_cast<long>(k_size), static_cast<long>(hidden_size)}));
        Tensor v_proj(std::move(v_data), Shape({static_cast<long>(v_size), static_cast<long>(hidden_size)}));

        // Apply linear projections
        Tensor q_proj_out = ops::linear(normed, q_proj, nullptr);
        Tensor k_proj_out = ops::linear(normed, k_proj, nullptr);
        Tensor v_proj_out = ops::linear(normed, v_proj, nullptr);

        std::cout << "  Q output: shape [" << q_proj_out.shape()[0] << ", " << q_proj_out.shape()[1] << "]\n";
        std::cout << "    Range: [" << q_proj_out.min() << ", " << q_proj_out.max() << "]\n";
        std::cout << "    Mean: " << q_proj_out.mean() << ", Std: " << compute_std(q_proj_out.data(), q_proj_out.size(), q_proj_out.mean()) << "\n";

        std::cout << "  K output: shape [" << k_proj_out.shape()[0] << ", " << k_proj_out.shape()[1] << "]\n";
        std::cout << "    Range: [" << k_proj_out.min() << ", " << k_proj_out.max() << "]\n";
        std::cout << "    Mean: " << k_proj_out.mean() << ", Std: " << compute_std(k_proj_out.data(), k_proj_out.size(), k_proj_out.mean()) << "\n";

        std::cout << "  V output: shape [" << v_proj_out.shape()[0] << ", " << v_proj_out.shape()[1] << "]\n";
        std::cout << "    Range: [" << v_proj_out.min() << ", " << v_proj_out.max() << "]\n";
        std::cout << "    Mean: " << v_proj_out.mean() << ", Std: " << compute_std(v_proj_out.data(), v_proj_out.size(), v_proj_out.mean()) << "\n";

        save_to_npy(q_proj_out.data(), q_proj_out.size(), "/tmp/cpp_q_proj.bin");
        save_to_npy(k_proj_out.data(), k_proj_out.size(), "/tmp/cpp_k_proj.bin");
        save_to_npy(v_proj_out.data(), v_proj_out.size(), "/tmp/cpp_v_proj.bin");

        {
            std::ifstream ref_q("/tmp/pytorch_q_proj.bin", std::ios::binary);
            std::vector<float> pytorch_q(q_proj_out.size());
            ref_q.read(reinterpret_cast<char*>(pytorch_q.data()), pytorch_q.size() * sizeof(float));
            compare_arrays(q_proj_out.data(), pytorch_q.data(), q_proj_out.size(), "Q projection vs PyTorch");

            std::ifstream ref_k("/tmp/pytorch_k_proj.bin", std::ios::binary);
            std::vector<float> pytorch_k(k_proj_out.size());
            ref_k.read(reinterpret_cast<char*>(pytorch_k.data()), pytorch_k.size() * sizeof(float));
            compare_arrays(k_proj_out.data(), pytorch_k.data(), k_proj_out.size(), "K projection vs PyTorch");
        }

        // ============================================================
        // Step 4: Reshape and Transpose
        // ============================================================
        std::cout << "\nStep 4: Reshape and Transpose\n";

        Tensor q_reshaped = q_proj_out.view({batch, seq, q_heads, head_dim});
        Tensor k_reshaped = k_proj_out.view({batch, seq, kv_heads, head_dim});
        Tensor v_reshaped = v_proj_out.view({batch, seq, kv_heads, head_dim});

        Tensor q_final = q_reshaped.transpose(1, 2);  // [batch, heads, seq, head_dim]
        Tensor k_final = k_reshaped.transpose(1, 2);
        Tensor v_final = v_reshaped.transpose(1, 2);

        std::cout << "  Q: [" << q_final.shape()[0] << ", " << q_final.shape()[1] << ", "
                  << q_final.shape()[2] << ", " << q_final.shape()[3] << "]\n";
        std::cout << "  K: [" << k_final.shape()[0] << ", " << k_final.shape()[1] << ", "
                  << k_final.shape()[2] << ", " << k_final.shape()[3] << "]\n";
        std::cout << "  V: [" << v_final.shape()[0] << ", " << v_final.shape()[1] << ", "
                  << v_final.shape()[2] << ", " << v_final.shape()[3] << "]\n";

        save_to_npy(q_final.data(), q_final.size(), "/tmp/cpp_q_reshaped.bin");
        save_to_npy(k_final.data(), k_final.size(), "/tmp/cpp_k_reshaped.bin");
        save_to_npy(v_final.data(), v_final.size(), "/tmp/cpp_v_reshaped.bin");

        {
            std::ifstream ref("/tmp/pytorch_q_reshaped.bin", std::ios::binary);
            std::vector<float> pytorch_data(q_final.size());
            ref.read(reinterpret_cast<char*>(pytorch_data.data()), pytorch_data.size() * sizeof(float));
            compare_arrays(q_final.data(), pytorch_data.data(), q_final.size(), "Q reshaped vs PyTorch");

            std::ifstream ref_k("/tmp/pytorch_k_reshaped.bin", std::ios::binary);
            std::vector<float> pytorch_k(k_final.size());
            ref_k.read(reinterpret_cast<char*>(pytorch_k.data()), pytorch_k.size() * sizeof(float));
            compare_arrays(k_final.data(), pytorch_k.data(), k_final.size(), "K reshaped vs PyTorch");
        }

        // ============================================================
        // Step 5: QKNorm
        // ============================================================
        std::cout << "\nStep 5: QKNorm (per-head RMS normalization)\n";

        // Apply QKNorm using same logic as qwen3_attention
        const float* q_norm_w = layer.q_norm_weight.data();
        const float* k_norm_w = layer.k_norm_weight.data();

        size_t q_total_elements = batch * q_heads * seq * head_dim;
        std::vector<float> q_normed_data(q_total_elements);

        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < q_heads; ++h) {
                for (size_t s = 0; s < seq; ++s) {
                    size_t base_idx = ((b * q_heads + h) * seq + s) * head_dim;

                    // Compute variance
                    float sum_sq = 0.0f;
                    for (size_t i = 0; i < head_dim; ++i) {
                        sum_sq += q_final[base_idx + i] * q_final[base_idx + i];
                    }
                    float variance = sum_sq / head_dim;
                    float rms = std::sqrt(variance + 1e-6f);

                    // Normalize and scale
                    for (size_t i = 0; i < head_dim; ++i) {
                        q_normed_data[base_idx + i] = (q_final[base_idx + i] / rms) * q_norm_w[i];
                    }
                }
            }
        }

        size_t k_total_elements = batch * kv_heads * seq * head_dim;
        std::vector<float> k_normed_data(k_total_elements);

        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < kv_heads; ++h) {
                for (size_t s = 0; s < seq; ++s) {
                    size_t base_idx = ((b * kv_heads + h) * seq + s) * head_dim;

                    float sum_sq = 0.0f;
                    for (size_t i = 0; i < head_dim; ++i) {
                        sum_sq += k_final[base_idx + i] * k_final[base_idx + i];
                    }
                    float variance = sum_sq / head_dim;
                    float rms = std::sqrt(variance + 1e-6f);

                    for (size_t i = 0; i < head_dim; ++i) {
                        k_normed_data[base_idx + i] = (k_final[base_idx + i] / rms) * k_norm_w[i];
                    }
                }
            }
        }

        Tensor q_normed(std::move(q_normed_data), q_final.shape());
        Tensor k_normed(std::move(k_normed_data), k_final.shape());

        std::cout << "  Q after QKNorm:\n";
        std::cout << "    Range: [" << q_normed.min() << ", " << q_normed.max() << "]\n";
        std::cout << "    Mean: " << q_normed.mean() << ", Std: " << compute_std(q_normed.data(), q_normed.size(), q_normed.mean()) << "\n";

        std::cout << "  K after QKNorm:\n";
        std::cout << "    Range: [" << k_normed.min() << ", " << k_normed.max() << "]\n";
        std::cout << "    Mean: " << k_normed.mean() << ", Std: " << compute_std(k_normed.data(), k_normed.size(), k_normed.mean()) << "\n";

        save_to_npy(q_normed.data(), q_normed.size(), "/tmp/cpp_q_after_qnorm.bin");
        save_to_npy(k_normed.data(), k_normed.size(), "/tmp/cpp_k_after_qnorm.bin");

        {
            std::ifstream ref_q("/tmp/pytorch_q_after_qnorm.bin", std::ios::binary);
            std::vector<float> pytorch_q(q_normed.size());
            ref_q.read(reinterpret_cast<char*>(pytorch_q.data()), pytorch_q.size() * sizeof(float));
            compare_arrays(q_normed.data(), pytorch_q.data(), q_normed.size(), "Q after QKNorm vs PyTorch");

            std::ifstream ref_k("/tmp/pytorch_k_after_qnorm.bin", std::ios::binary);
            std::vector<float> pytorch_k(k_normed.size());
            ref_k.read(reinterpret_cast<char*>(pytorch_k.data()), pytorch_k.size() * sizeof(float));
            compare_arrays(k_normed.data(), pytorch_k.data(), k_normed.size(), "K after QKNorm vs PyTorch");
        }

        // ============================================================
        // Step 6: RoPE
        // ============================================================
        std::cout << "\nStep 6: RoPE (Rotary Position Embedding)\n";

        // Compute RoPE frequencies
        auto [cos, sin] = qwen3::compute_rope_freqs(seq, head_dim, 1000000.0f);

        // Apply RoPE
        auto [q_rope, k_rope] = qwen3::apply_rotary_pos_emb(q_normed, k_normed, cos, sin);

        std::cout << "  Q after RoPE:\n";
        std::cout << "    Range: [" << q_rope.min() << ", " << q_rope.max() << "]\n";
        std::cout << "    Mean: " << q_rope.mean() << "\n";

        std::cout << "  K after RoPE:\n";
        std::cout << "    Range: [" << k_rope.min() << ", " << k_rope.max() << "]\n";
        std::cout << "    Mean: " << k_rope.mean() << "\n";

        save_to_npy(q_rope.data(), q_rope.size(), "/tmp/cpp_q_rope.bin");
        save_to_npy(k_rope.data(), k_rope.size(), "/tmp/cpp_k_rope.bin");

        {
            std::ifstream ref_q("/tmp/pytorch_q_rope.bin", std::ios::binary);
            std::vector<float> pytorch_q(q_rope.size());
            ref_q.read(reinterpret_cast<char*>(pytorch_q.data()), pytorch_q.size() * sizeof(float));
            compare_arrays(q_rope.data(), pytorch_q.data(), q_rope.size(), "Q RoPE vs PyTorch");

            std::ifstream ref_k("/tmp/pytorch_k_rope.bin", std::ios::binary);
            std::vector<float> pytorch_k(k_rope.size());
            ref_k.read(reinterpret_cast<char*>(pytorch_k.data()), pytorch_k.size() * sizeof(float));
            compare_arrays(k_rope.data(), pytorch_k.data(), k_rope.size(), "K RoPE vs PyTorch");
        }

        // ============================================================
        // Step 7: Repeat KV for GQA
        // ============================================================
        std::cout << "\nStep 7: Repeat KV for GQA\n";

        int n_rep = static_cast<int>(q_heads / kv_heads);
        Tensor k_repeated = qwen3::repeat_kv(k_rope, n_rep);
        Tensor v_repeated = qwen3::repeat_kv(v_final, n_rep);

        std::cout << "  K repeated: [" << k_repeated.shape()[0] << ", " << k_repeated.shape()[1] << ", "
                  << k_repeated.shape()[2] << ", " << k_repeated.shape()[3] << "]\n";
        std::cout << "  V repeated: [" << v_repeated.shape()[0] << ", " << v_repeated.shape()[1] << ", "
                  << v_repeated.shape()[2] << ", " << v_repeated.shape()[3] << "]\n";

        save_to_npy(k_repeated.data(), k_repeated.size(), "/tmp/cpp_k_repeated.bin");
        save_to_npy(v_repeated.data(), v_repeated.size(), "/tmp/cpp_v_repeated.bin");

        {
            std::ifstream ref_k("/tmp/pytorch_k_repeated.bin", std::ios::binary);
            std::vector<float> pytorch_k(k_repeated.size());
            ref_k.read(reinterpret_cast<char*>(pytorch_k.data()), pytorch_k.size() * sizeof(float));
            compare_arrays(k_repeated.data(), pytorch_k.data(), k_repeated.size(), "K repeated vs PyTorch");

            std::ifstream ref_v("/tmp/pytorch_v_repeated.bin", std::ios::binary);
            std::vector<float> pytorch_v(v_repeated.size());
            ref_v.read(reinterpret_cast<char*>(pytorch_v.data()), pytorch_v.size() * sizeof(float));
            compare_arrays(v_repeated.data(), pytorch_v.data(), v_repeated.size(), "V repeated vs PyTorch");
        }

        // ============================================================
        // Step 8: Attention
        // ============================================================
        std::cout << "\nStep 8: Self-Attention\n";

        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        Tensor causal_mask = qwen3::create_causal_mask(seq);
        Tensor mask_reshaped = causal_mask.view({1, 1, seq, seq});

        Tensor attn_output = ops::self_attention(q_rope, k_repeated, v_repeated, &mask_reshaped, scale);

        std::cout << "  Attention output: [" << attn_output.shape()[0] << ", " << attn_output.shape()[1] << ", "
                  << attn_output.shape()[2] << ", " << attn_output.shape()[3] << "]\n";
        std::cout << "    Range: [" << attn_output.min() << ", " << attn_output.max() << "]\n";
        std::cout << "    Mean: " << attn_output.mean() << "\n";

        // Transpose and reshape for output projection
        Tensor attn_output_t = attn_output.transpose(1, 2);  // [batch, seq, heads, head_dim]
        Tensor attn_output_flat = attn_output_t.contiguous().view({batch, seq, q_heads * head_dim});

        save_to_npy(attn_output_flat.data(), attn_output_flat.size(), "/tmp/cpp_attn_output.bin");

        {
            std::ifstream ref("/tmp/pytorch_attn_output.bin", std::ios::binary);
            std::vector<float> pytorch_data(attn_output_flat.size());
            ref.read(reinterpret_cast<char*>(pytorch_data.data()), pytorch_data.size() * sizeof(float));
            compare_arrays(attn_output_flat.data(), pytorch_data.data(), attn_output_flat.size(), "Attention output vs PyTorch");
        }

        std::cout << "\n============================================================\n";
        std::cout << "  Alignment Test Complete!\n";
        std::cout << "============================================================\n\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n✗ ERROR: " << e.what() << "\n\n";
        return 1;
    }
}
