/**
 * @file basic_usage.cpp
 * @brief Basic usage examples for the Tensor Library
 *
 * Compile:
 *   g++ -std=c++17 -O3 -I./include -fopenmp examples/basic_usage.cpp -o basic_usage
 *
 * Run:
 *   ./basic_usage
 */

#include "tensor_cpp/tensor.h"
#include "tensor_cpp/ops.h"
#include <iostream>

using namespace tensor_cpp;
using namespace ops;

int main() {
    std::cout << "=== Tensor Library Basic Usage Examples ===\n\n";

    // ========================================================================
    // Example 1: Creating Tensors
    // ========================================================================

    std::cout << "1. Creating Tensors\n";
    std::cout << "-------------------\n";

    // Create a zero tensor
    TensorF zeros = TensorF::zeros(Shape({2, 3}));
    std::cout << "Zeros (2x3):\n" << zeros.to_string() << "\n\n";

    // Create a random tensor
    TensorF random = TensorF::randn(Shape({2, 2}));
    std::cout << "Random (2x2):\n" << random.to_string() << "\n\n";

    // Create from data
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    TensorF custom(data, Shape({2, 2}));
    std::cout << "Custom (2x2):\n" << custom.to_string() << "\n\n";

    // ========================================================================
    // Example 2: Element-wise Operations
    // ========================================================================

    std::cout << "2. Element-wise Operations\n";
    std::cout << "--------------------------\n";

    TensorF a = TensorF::ones(Shape({2, 2}));
    a[0] = 1.0f; a[1] = 2.0f; a[2] = 3.0f; a[3] = 4.0f;

    TensorF b = a * 2.0f;  // Scalar multiplication
    std::cout << "a * 2:\n" << b.to_string() << "\n";

    TensorF c = a + b;     // Addition
    std::cout << "a + (a*2):\n" << c.to_string() << "\n\n";

    // ========================================================================
    // Example 3: Matrix Operations
    // ========================================================================

    std::cout << "3. Matrix Operations\n";
    std::cout << "--------------------\n";

    std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data2 = {5.0f, 6.0f, 7.0f, 8.0f};
    TensorF mat1(data1, Shape({2, 2}));
    TensorF mat2(data2, Shape({2, 2}));

    TensorF matmul_result = mat1.matmul(mat2);
    std::cout << "mat1 @ mat2:\n" << matmul_result.to_string() << "\n\n";

    // ========================================================================
    // Example 4: Reduction Operations
    // ========================================================================

    std::cout << "4. Reduction Operations\n";
    std::cout << "-----------------------\n";

    std::vector<float> data_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    TensorF x(data_x, Shape({2, 3}));
    std::cout << "Tensor x:\n" << x.to_string() << "\n";

    std::cout << "Sum: " << x.sum() << "\n";
    std::cout << "Mean: " << x.mean() << "\n";
    std::cout << "Max: " << x.max() << "\n";
    std::cout << "Min: " << x.min() << "\n\n";

    // ========================================================================
    // Example 5: Linear Layer
    // ========================================================================

    std::cout << "5. Linear Layer\n";
    std::cout << "---------------\n";

    // Input: (batch_size=2, in_features=4)
    TensorF input = TensorF::randn(Shape({2, 4}));

    // Weight: (out_features=3, in_features=4)
    TensorF weight = TensorF::randn(Shape({3, 4}));

    // Bias: (out_features=3)
    TensorF bias = TensorF::randn(Shape({3}));

    // Forward pass
    TensorF output = linear(input, weight, &bias);

    std::cout << "Input shape: " << input.shape().to_string() << "\n";
    std::cout << "Output shape: " << output.shape().to_string() << "\n";
    std::cout << "Output:\n" << output.to_string() << "\n\n";

    // ========================================================================
    // Example 6: Self-Attention
    // ========================================================================

    std::cout << "6. Self-Attention\n";
    std::cout << "-----------------\n";

    // Query, Key, Value: (batch=1, heads=2, seq_len=4, head_dim=8)
    size_t batch = 1, heads = 2, seq_len = 4, head_dim = 8;

    TensorF query = TensorF::randn(Shape({batch, heads, seq_len, head_dim}));
    TensorF key = TensorF::randn(Shape({batch, heads, seq_len, head_dim}));
    TensorF value = TensorF::randn(Shape({batch, heads, seq_len, head_dim}));

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    TensorF attn_output = self_attention(query, key, value,
                                          static_cast<const TensorF*>(nullptr),
                                          scale);

    std::cout << "Q/K/V shape: " << query.shape().to_string() << "\n";
    std::cout << "Output shape: " << attn_output.shape().to_string() << "\n\n";

    // ========================================================================
    // Example 7: Cross-Attention
    // ========================================================================

    std::cout << "7. Cross-Attention\n";
    std::cout << "------------------\n";

    // Query: (batch=1, heads=2, query_len=4, head_dim=8)
    // Key/Value: (batch=1, heads=2, kv_len=6, head_dim=8)
    size_t query_len = 4, kv_len = 6;

    TensorF query_cross = TensorF::randn(Shape({batch, heads, query_len, head_dim}));
    TensorF key_cross = TensorF::randn(Shape({batch, heads, kv_len, head_dim}));
    TensorF value_cross = TensorF::randn(Shape({batch, heads, kv_len, head_dim}));

    TensorF cross_output = cross_attention(query_cross, key_cross, value_cross,
                                            static_cast<const TensorF*>(nullptr),
                                            scale);

    std::cout << "Query shape: " << query_cross.shape().to_string() << "\n";
    std::cout << "Key/Value shape: " << key_cross.shape().to_string() << "\n";
    std::cout << "Output shape: " << cross_output.shape().to_string() << "\n\n";

    // ========================================================================
    // Example 8: Argmax
    // ========================================================================

    std::cout << "8. Argmax Operation\n";
    std::cout << "-------------------\n";

    std::vector<float> scores_data = {0.1f, 0.9f, 0.3f, 0.5f};
    TensorF scores(scores_data, Shape({4}));
    TensorL max_idx = argmax(scores);

    std::cout << "Scores: [0.1, 0.9, 0.3, 0.5]\n";
    std::cout << "Argmax index: " << max_idx[0] << " (expected 1)\n\n";

    // ========================================================================
    // Example 9: Embedding Lookup
    // ========================================================================

    std::cout << "9. Embedding Lookup\n";
    std::cout << "-------------------\n";

    // Weight: (num_embeddings=100, embedding_dim=64)
    TensorF embedding_weight = TensorF::randn(Shape({100, 64}));

    // Indices: (batch_size=2, seq_len=10)
    std::vector<long> indices_data(2 * 10);
    for (size_t i = 0; i < indices_data.size(); ++i) {
        indices_data[i] = static_cast<long>(i % 100);  // Indices 0-9 repeated
    }
    TensorL indices(indices_data, Shape({2, 10}));

    TensorF embeddings = embedding(indices, embedding_weight);

    std::cout << "Indices shape: " << indices.shape().to_string() << "\n";
    std::cout << "Embedding weight shape: " << embedding_weight.shape().to_string() << "\n";
    std::cout << "Output shape: " << embeddings.shape().to_string() << "\n\n";

    // ========================================================================
    // Example 10: Mathematical Functions
    // ========================================================================

    std::cout << "10. Mathematical Functions\n";
    std::cout << "--------------------------\n";

    std::vector<float> x_math_data = {1.0f, 2.0f, 3.0f, 4.0f};
    TensorF x_math(x_math_data, Shape({4}));

    std::cout << "x: [1, 2, 3, 4]\n";
    std::cout << "exp(x): [" << x_math.exp()[0] << ", " << x_math.exp()[1] << ", ...]\n";
    std::cout << "sqrt(x): [" << x_math.sqrt()[0] << ", " << x_math.sqrt()[1] << ", ...]\n";
    std::cout << "square(x): [" << x_math.square()[0] << ", " << x_math.square()[1] << ", ...]\n\n";

    std::cout << "=== All Examples Completed ===\n";

    return 0;
}
