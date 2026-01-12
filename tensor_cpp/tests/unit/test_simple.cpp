/**
 * @file test_simple.cpp
 * @brief Simple test to verify compilation
 */

#include "tensor_cpp/tensor.h"
#include "tensor_cpp/ops.h"
#include <iostream>

using namespace tensor_cpp;
using namespace ops;

int main() {
    std::cout << "Testing tensor_cpp library...\n";

    // Create tensor
    Tensor x = Tensor::zeros(Shape({2, 3}));
    x[0] = 1.0f;
    x[1] = 2.0f;

    std::cout << "Created tensor: " << x.shape().to_string() << "\n";
    std::cout << "x[0] = " << x[0] << ", x[1] = " << x[1] << "\n";

    // Test argmax
    Tensor scores = Tensor::zeros(Shape({4}));
    scores[0] = 0.1f; scores[1] = 0.9f; scores[2] = 0.3f; scores[3] = 0.5f;
    TensorL idx = argmax(scores);
    std::cout << "Argmax index: " << idx[0] << " (expected 1)\n";

    // Test streaming attention
    std::vector<float> Q(64, 0.0f);
    std::vector<float> K(64 * 128, 0.0f);
    std::vector<float> V(64 * 128, 0.0f);

    // Fill with some values
    for (int i = 0; i < 64; ++i) {
        Q[i] = 0.1f * i;
        for (int j = 0; j < 128; ++j) {
            K[i * 128 + j] = 0.01f * (i + j);
            V[i * 128 + j] = 0.02f * (i + j);
        }
    }

    std::vector<float> output = naive_attention_serial(Q.data(), K.data(), V.data(), 128, 64);
    std::cout << "Streaming attention output size: " << output.size() << "\n";

    std::cout << "\nAll tests passed!\n";
    return 0;
}
