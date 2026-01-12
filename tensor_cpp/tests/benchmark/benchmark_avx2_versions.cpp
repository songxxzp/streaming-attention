/**
 * @file benchmark_avx2_versions.cpp
 * @brief Compare Baseline vs AVX2 performance
 */

#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/qwen3_ops_avx.h"
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace tensor_cpp;
using namespace tensor_cpp::qwen3;

class Timer {
    std::chrono::high_resolution_clock::time_point start_;
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
};

void print_header() {
    std::cout << "\n============================================================\n";
    std::cout << "     Qwen3 Baseline vs AVX2 Performance\n";
    std::cout << "============================================================\n";
    #ifdef _OPENMP
    std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
    #endif
    std::cout << "============================================================\n\n";
}

void benchmark_forward_pass(
    const std::string& version,
    const TensorL& input_ids,
    const Qwen3Weights& weights
) {
    std::cout << "----------------------------------------\n";
    std::cout << version << "\n";
    std::cout << "----------------------------------------\n";

    Timer timer;
    Tensor output;

    if (version == "Baseline") {
        output = qwen3::qwen3_forward(
            input_ids,
            weights.embed_tokens,
            weights.layers,
            weights.norm_weight,
            weights.num_layers,
            weights.num_attention_heads,
            weights.num_key_value_heads,
            weights.head_dim,
            1e-6f
        );
    } else if (version == "AVX2") {
        output = avx2::qwen3_forward_avx(
            input_ids,
            weights.embed_tokens,
            weights.layers,
            weights.norm_weight,
            weights.num_layers,
            weights.num_attention_heads,
            weights.num_key_value_heads,
            weights.head_dim,
            1e-6f
        );
    }

    double elapsed = timer.elapsed_ms();

    std::cout << "  Time: " << std::fixed << std::setprecision(2) << elapsed << " ms\n";
    std::cout << "  Output shape: ["
              << output.shape()[0] << ", "
              << output.shape()[1] << ", "
              << output.shape()[2] << "]\n";

    // Calculate throughput
    size_t seq_len = input_ids.shape()[1];
    size_t num_tokens = input_ids.shape()[0] * seq_len;
    double tokens_per_sec = (num_tokens / elapsed) * 1000.0;
    std::cout << "  Throughput: " << std::fixed << std::setprecision(1)
              << tokens_per_sec << " tokens/sec\n";
}

int main() {
    print_header();

    try {
        // Path to model
        std::string model_path = "/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors";

        std::cout << "Loading model from: " << model_path << "\n";

        // Load weights
        Qwen3Weights weights = load_qwen3_weights(model_path);

        std::cout << "Model loaded successfully!\n";
        std::cout << "  Hidden size: " << weights.hidden_size << "\n";
        std::cout << "  Num layers: " << weights.num_layers << "\n";
        std::cout << "  Num attention heads: " << weights.num_attention_heads << "\n";
        std::cout << "  Num KV heads: " << weights.num_key_value_heads << "\n";
        std::cout << "  Head dim: " << weights.head_dim << "\n\n";

        // Test input: different sequence lengths
        std::vector<std::pair<int, int>> test_cases = {
            {1, 4},   // Short sequence
            {1, 16},  // Medium sequence
            {1, 32},  // Longer sequence
        };

        for (const auto& test_case : test_cases) {
            int batch_size = test_case.first;
            int seq_len = test_case.second;

            std::cout << "\n============================================================\n";
            std::cout << "Test Case: batch=" << batch_size << ", seq_len=" << seq_len << "\n";
            std::cout << "============================================================\n";

            // Create input
            std::vector<long> input_ids_data;
            for (int i = 0; i < batch_size * seq_len; ++i) {
                input_ids_data.push_back(static_cast<long>(i % 1000));
            }
            TensorL input_ids(input_ids_data, Shape({batch_size, seq_len}));

            // Benchmark each version
            benchmark_forward_pass("Baseline", input_ids, weights);
            benchmark_forward_pass("AVX2", input_ids, weights);
        }

        std::cout << "\n============================================================\n";
        std::cout << "Benchmark completed!\n";
        std::cout << "============================================================\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
