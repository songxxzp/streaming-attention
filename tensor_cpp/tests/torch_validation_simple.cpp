/**
 * @file torch_validation_simple.cpp
 * @brief Simplified C++ validation - generate outputs for Python to validate
 */

#include "tensor_cpp/tensor.h"
#include "tensor_cpp/ops.h"
#include <iostream>
#include <fstream>
#include <vector>

using namespace tensor_cpp;
using namespace tensor_cpp::ops;

// Simple binary file writer
void save_binary(const char* path, const void* data, size_t size) {
    std::ofstream out(path, std::ios::binary);
    out.write(static_cast<const char*>(data), size);
    out.close();
}

void test_self_attention() {
    std::cout << "Testing self_attention_1...\n";

    // Fixed test configuration matching Python
    int batch = 2, heads = 2, seq = 8, dim = 16;
    float scale = 0.25f;  // 1/sqrt(16)

    // Load data from simple binary files
    std::vector<float> q_data(batch * heads * seq * dim);
    std::vector<float> k_data(batch * heads * seq * dim);
    std::vector<float> v_data(batch * heads * seq * dim);

    std::ifstream in_q("../test_data/self_attention_1_query.bin", std::ios::binary);
    std::ifstream in_k("../test_data/self_attention_1_key.bin", std::ios::binary);
    std::ifstream in_v("../test_data/self_attention_1_value.bin", std::ios::binary);

    if (!in_q || !in_k || !in_v) {
        std::cout << "  Creating test data...\n";

        // Create random data
        for (size_t i = 0; i < q_data.size(); ++i) {
            q_data[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
            k_data[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
            v_data[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        }

        // Save for Python
        save_binary("../test_data/self_attention_1_query.bin", q_data.data(), q_data.size() * sizeof(float));
        save_binary("../test_data/self_attention_1_key.bin", k_data.data(), k_data.size() * sizeof(float));
        save_binary("../test_data/self_attention_1_value.bin", v_data.data(), v_data.size() * sizeof(float));
    } else {
        in_q.read(reinterpret_cast<char*>(q_data.data()), q_data.size() * sizeof(float));
        in_k.read(reinterpret_cast<char*>(k_data.data()), k_data.size() * sizeof(float));
        in_v.read(reinterpret_cast<char*>(v_data.data()), v_data.size() * sizeof(float));
        in_q.close(); in_k.close(); in_v.close();
    }

    // Create tensors
    TensorF q(q_data, Shape({batch, heads, seq, dim}));
    TensorF k(k_data, Shape({batch, heads, seq, dim}));
    TensorF v(v_data, Shape({batch, heads, seq, dim}));

    // Run attention
    TensorF output = self_attention(q, k, v, nullptr, scale);

    // Save output
    save_binary("../test_data/cpp_self_attention_1_output.bin",
               output.data(), output.size() * sizeof(float));

    std::cout << "  ✓ Saved output: " << output.size() << " elements\n";
}

void test_cross_attention() {
    std::cout << "Testing cross_attention_1...\n";

    int batch = 2, heads = 2, q_len = 8, kv_len = 16, dim = 16;
    float scale = 0.25f;

    std::vector<float> q_data(batch * heads * q_len * dim);
    std::vector<float> k_data(batch * heads * kv_len * dim);
    std::vector<float> v_data(batch * heads * kv_len * dim);

    // Try to load, otherwise generate
    std::ifstream in_q("../test_data/cross_attention_1_query.bin", std::ios::binary);
    if (!in_q) {
        std::cout << "  Creating test data...\n";
        for (size_t i = 0; i < q_data.size(); ++i) q_data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        for (size_t i = 0; i < k_data.size(); ++i) k_data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        for (size_t i = 0; i < v_data.size(); ++i) v_data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;

        save_binary("../test_data/cross_attention_1_query.bin", q_data.data(), q_data.size() * sizeof(float));
        save_binary("../test_data/cross_attention_1_key.bin", k_data.data(), k_data.size() * sizeof(float));
        save_binary("../test_data/cross_attention_1_value.bin", v_data.data(), v_data.size() * sizeof(float));
    } else {
        in_q.read(reinterpret_cast<char*>(q_data.data()), q_data.size() * sizeof(float));
        in_q.close();

        std::ifstream in_k("../test_data/cross_attention_1_key.bin", std::ios::binary);
        std::ifstream in_v("../test_data/cross_attention_1_value.bin", std::ios::binary);
        in_k.read(reinterpret_cast<char*>(k_data.data()), k_data.size() * sizeof(float));
        in_v.read(reinterpret_cast<char*>(v_data.data()), v_data.size() * sizeof(float));
        in_k.close(); in_v.close();
    }

    TensorF q(q_data, Shape({batch, heads, q_len, dim}));
    TensorF k(k_data, Shape({batch, heads, kv_len, dim}));
    TensorF v(v_data, Shape({batch, heads, kv_len, dim}));

    TensorF output = cross_attention(q, k, v, nullptr, scale);

    save_binary("../test_data/cpp_cross_attention_1_output.bin",
               output.data(), output.size() * sizeof(float));

    std::cout << "  ✓ Saved output: " << output.size() << " elements\n";
}

void test_streaming_attention() {
    std::cout << "Testing streaming_attention_1...\n";

    int T = 512, d = 64;

    std::vector<float> Q_data(d);
    std::vector<float> K_data(T * d);
    std::vector<float> V_data(T * d);

    // Load or generate
    std::ifstream in_q("../test_data/streaming_attention_1_Q.bin", std::ios::binary);
    if (!in_q) {
        std::cout << "  Creating test data...\n";
        srand(42);
        for (size_t i = 0; i < Q_data.size(); ++i) Q_data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        for (size_t i = 0; i < K_data.size(); ++i) K_data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        for (size_t i = 0; i < V_data.size(); ++i) V_data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;

        save_binary("../test_data/streaming_attention_1_Q.bin", Q_data.data(), Q_data.size() * sizeof(float));
        save_binary("../test_data/streaming_attention_1_K.bin", K_data.data(), K_data.size() * sizeof(float));
        save_binary("../test_data/streaming_attention_1_V.bin", V_data.data(), V_data.size() * sizeof(float));
    } else {
        in_q.read(reinterpret_cast<char*>(Q_data.data()), Q_data.size() * sizeof(float));
        in_q.close();

        std::ifstream in_k("../test_data/streaming_attention_1_K.bin", std::ios::binary);
        std::ifstream in_v("../test_data/streaming_attention_1_V.bin", std::ios::binary);
        in_k.read(reinterpret_cast<char*>(K_data.data()), K_data.size() * sizeof(float));
        in_v.read(reinterpret_cast<char*>(V_data.data()), V_data.size() * sizeof(float));
        in_k.close(); in_v.close();
    }

    std::vector<float> output = streaming_attention_serial(Q_data.data(), K_data.data(), V_data.data(), T, d, 64);

    save_binary("../test_data/cpp_streaming_attention_1_output.bin",
               output.data(), output.size() * sizeof(float));

    std::cout << "  ✓ Saved output: " << output.size() << " elements\n";
}

void test_linear() {
    std::cout << "Testing linear_1...\n";

    int batch = 2, in_feat = 64, out_feat = 32;

    std::vector<float> x_data(batch * in_feat);
    std::vector<float> w_data(out_feat * in_feat);
    std::vector<float> b_data(out_feat);

    // Try to load
    std::ifstream in_x("../test_data/linear_1_x.bin", std::ios::binary);
    if (!in_x) {
        std::cout << "  Creating test data...\n";
        srand(42);
        for (auto& v : x_data) v = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        for (auto& v : w_data) v = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        for (auto& v : b_data) v = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;

        // Save for Python (use same seed)
        save_binary("../test_data/linear_1_x.bin", x_data.data(), x_data.size() * sizeof(float));
        save_binary("../test_data/linear_1_weight.bin", w_data.data(), w_data.size() * sizeof(float));
        save_binary("../test_data/linear_1_bias.bin", b_data.data(), b_data.size() * sizeof(float));
    } else {
        in_x.read(reinterpret_cast<char*>(x_data.data()), x_data.size() * sizeof(float));
        in_x.close();

        std::ifstream in_w("../test_data/linear_1_weight.bin", std::ios::binary);
        std::ifstream in_b("../test_data/linear_1_bias.bin", std::ios::binary);
        in_w.read(reinterpret_cast<char*>(w_data.data()), w_data.size() * sizeof(float));
        in_b.read(reinterpret_cast<char*>(b_data.data()), b_data.size() * sizeof(float));
        in_w.close(); in_b.close();
    }

    TensorF x(x_data, Shape({batch, in_feat}));
    TensorF weight(w_data, Shape({out_feat, in_feat}));
    TensorF bias(b_data, Shape({out_feat}));

    TensorF output = linear(x, weight, &bias);

    save_binary("../test_data/cpp_linear_1_output.bin",
               output.data(), output.size() * sizeof(float));

    std::cout << "  ✓ Saved output: " << output.size() << " elements\n";
}

int main() {
    std::cout << "\n";
    std::cout << "========================================================\n";
    std::cout << "  C++ Validation Test (Simplified)\n";
    std::cout << "========================================================\n\n";

    try {
        test_self_attention();
        test_cross_attention();
        test_streaming_attention();
        test_linear();

        std::cout << "\n";
        std::cout << "========================================================\n";
        std::cout << "  All C++ Outputs Generated\n";
        std::cout << "========================================================\n\n";
        std::cout << "Now run Python validation:\n";
        std::cout << "  python3 validate_cpp_outputs.py\n\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n✗ ERROR: " << e.what() << "\n\n";
        return 1;
    }
}
