/**
 * @file debug_linear_avx.cpp
 * @brief 比较linear和linear_avx2的输出
 */

#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/ops.h"
#include "tensor_cpp/attention_avx.h"
#include <iostream>
#include <cmath>

using namespace tensor_cpp;
using namespace tensor_cpp::ops;
using namespace tensor_cpp::ops::avx2;
using namespace tensor_cpp::qwen3;

int main() {
    std::string model_path = "/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors";
    std::cout << "Loading model...\n";
    Qwen3Weights weights = load_qwen3_weights(model_path);

    const auto& layer = weights.layers[0];

    // 创建测试输入 - 2D tensor [2, 1024]
    std::vector<float> input_data(2 * 1024);  // [2, 1024]
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = static_cast<float>(i % 100) / 100.0f;
    }

    Tensor input(std::move(input_data), Shape({2, 1024}));

    std::cout << "Testing linear vs linear_avx2...\n";
    std::cout << "Input shape: [" << input.shape()[0] << ", " << input.shape()[1] << "]\n";
    std::cout << "Q proj shape: [" << layer.q_proj.shape()[0] << ", " << layer.q_proj.shape()[1] << "]\n\n";

    // Baseline linear
    Tensor out1 = linear(input, layer.q_proj, nullptr);
    std::cout << "Baseline linear output shape: [" << out1.shape()[0] << ", " << out1.shape()[1] << "]\n";
    std::cout << "Baseline linear output size: " << out1.size() << "\n";

    // AVX2 linear
    Tensor out2 = linear_avx2(input, layer.q_proj, nullptr);
    std::cout << "AVX2 linear output shape: [" << out2.shape()[0] << ", " << out2.shape()[1] << "]\n";
    std::cout << "AVX2 linear output size: " << out2.size() << "\n\n";

    // 比较
    if (out1.size() != out2.size()) {
        std::cout << "ERROR: Size mismatch!\n";
        return 1;
    }

    float max_diff = 0.0f;
    float max_rel_diff = 0.0f;
    int mismatch_count = 0;

    for (size_t i = 0; i < out1.size(); ++i) {
        float v1 = out1.data()[i];
        float v2 = out2.data()[i];
        float diff = std::abs(v1 - v2);
        max_diff = std::max(max_diff, diff);

        float abs_val = std::max(std::abs(v1), std::abs(v2));
        float rel_diff = (abs_val > 1e-6f) ? (diff / abs_val) : diff;
        max_rel_diff = std::max(max_rel_diff, rel_diff);

        if (diff > 0.001f) {
            mismatch_count++;
            if (mismatch_count <= 10) {
                std::cout << "  [" << i << "] baseline=" << v1 << ", avx2=" << v2
                          << ", diff=" << diff << " (" << (rel_diff * 100.0f) << "%)\n";
            }
        }
    }

    std::cout << "\nResults:\n";
    std::cout << "  Max absolute diff: " << max_diff << "\n";
    std::cout << "  Max relative diff: " << (max_rel_diff * 100.0f) << "%\n";
    std::cout << "  Mismatch count: " << mismatch_count << " / " << out1.size() << "\n";

    if (max_diff > 0.1f || max_rel_diff > 0.01f) {
        std::cout << "\n*** ERROR: Linear outputs don't match! ***\n";
        return 1;
    } else {
        std::cout << "\n*** OK: Linear outputs match! ***\n";
        return 0;
    }
}
