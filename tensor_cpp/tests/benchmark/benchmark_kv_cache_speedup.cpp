/**
 * @file benchmark_kv_cache_speedup.cpp
 * @brief Benchmark KV cache speedup for autoregressive generation
 */

#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/qwen3_ops_avx.h"
#include "tensor_cpp/kv_cache.h"
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
    std::cout << "     KV Cache Speedup Benchmark\n";
    std::cout << "============================================================\n";
    #ifdef _OPENMP
    std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
    #endif
    std::cout << "============================================================\n\n";
}

// Baseline: No KV cache (recompute everything each forward pass)
void benchmark_without_cache(
    const std::string& name,
    const TensorL& input_ids,
    const Qwen3Weights& weights,
    int num_generate_tokens
) {
    std::cout << "----------------------------------------\n";
    std::cout << name << "\n";
    std::cout << "----------------------------------------\n";

    Timer total_timer;

    // Initial forward pass
    Timer timer;
    Tensor output = qwen3::qwen3_forward(
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
    double prefill_time = timer.elapsed_ms();

    std::cout << "  Prefill (" << input_ids.shape()[1] << " tokens): "
              << std::fixed << std::setprecision(2) << prefill_time << " ms\n";

    // Autoregressive generation WITHOUT cache
    double total_decode_time = 0.0;
    std::vector<long> current_ids(input_ids.data(), input_ids.data() + input_ids.size());

    for (int i = 0; i < num_generate_tokens; ++i) {
        // Get last token
        long last_token = current_ids.back();

        // Create single-token input
        std::vector<long> single_token = {last_token};
        TensorL next_input(single_token, Shape({1, 1}));

        timer = Timer();
        Tensor next_output = qwen3::qwen3_forward(
            next_input,
            weights.embed_tokens,
            weights.layers,
            weights.norm_weight,
            weights.num_layers,
            weights.num_attention_heads,
            weights.num_key_value_heads,
            weights.head_dim,
            1e-6f
        );
        double token_time = timer.elapsed_ms();
        total_decode_time += token_time;

        // Sample next token (just take argmax for simplicity)
        size_t vocab_size = weights.lm_head.shape()[0];
        float max_logit = -std::numeric_limits<float>::infinity();
        long next_token_id = 0;

        const float* logits = next_output.data();
        for (size_t j = 0; j < vocab_size; ++j) {
            if (logits[j] > max_logit) {
                max_logit = logits[j];
                next_token_id = static_cast<long>(j);
            }
        }

        current_ids.push_back(next_token_id);

        if (i < 3 || i >= num_generate_tokens - 3) {
            std::cout << "  Token " << (i + 1) << ": " << token_time << " ms\n";
        } else if (i == 3) {
            std::cout << "  ...\n";
        }
    }

    double total_time = total_timer.elapsed_ms();

    std::cout << "\n  Decode total (" << num_generate_tokens << " tokens): "
              << total_decode_time << " ms\n";
    std::cout << "  Decode avg per token: "
              << (total_decode_time / num_generate_tokens) << " ms\n";
    std::cout << "  Total time: " << total_time << " ms\n";
}

// With KV cache: Only compute new tokens
void benchmark_with_cache(
    const std::string& name,
    const TensorL& input_ids,
    Qwen3Weights& weights,
    int num_generate_tokens,
    bool use_avx = false
) {
    std::cout << "----------------------------------------\n";
    std::cout << name << "\n";
    std::cout << "----------------------------------------\n";

    Timer total_timer;

    // Create KV cache
    KVCache kv_cache(
        weights.num_layers,
        1,  // batch_size
        weights.num_key_value_heads,
        weights.head_dim,
        4096  // max_seq_len
    );

    // Initial forward pass with cache initialization
    Timer timer;
    Tensor output;
    if (use_avx) {
        output = avx2::qwen3_forward_avx_with_cache(
            input_ids,
            &kv_cache,
            weights.embed_tokens,
            weights.layers,
            weights.norm_weight,
            weights.num_layers,
            weights.num_attention_heads,
            weights.num_key_value_heads,
            weights.head_dim,
            1e-6f
        );
    } else {
        output = qwen3::qwen3_forward_with_cache(
            input_ids,
            &kv_cache,
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
    double prefill_time = timer.elapsed_ms();

    std::cout << "  Prefill (" << input_ids.shape()[1] << " tokens): "
              << std::fixed << std::setprecision(2) << prefill_time << " ms\n";

    // Autoregressive generation WITH cache
    double total_decode_time = 0.0;
    std::vector<long> current_ids(input_ids.data(), input_ids.data() + input_ids.size());

    for (int i = 0; i < num_generate_tokens; ++i) {
        // Get last token
        long last_token = current_ids.back();

        // Create single-token input
        std::vector<long> single_token = {last_token};
        TensorL next_input(single_token, Shape({1, 1}));

        timer = Timer();
        Tensor next_output;
        if (use_avx) {
            next_output = avx2::qwen3_forward_avx_with_cache(
                next_input,
                &kv_cache,
                weights.embed_tokens,
                weights.layers,
                weights.norm_weight,
                weights.num_layers,
                weights.num_attention_heads,
                weights.num_key_value_heads,
                weights.head_dim,
                1e-6f
            );
        } else {
            next_output = qwen3::qwen3_forward_with_cache(
                next_input,
                &kv_cache,
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
        double token_time = timer.elapsed_ms();
        total_decode_time += token_time;

        // Sample next token
        size_t vocab_size = weights.lm_head.shape()[0];
        float max_logit = -std::numeric_limits<float>::infinity();
        long next_token_id = 0;

        const float* logits = next_output.data();
        for (size_t j = 0; j < vocab_size; ++j) {
            if (logits[j] > max_logit) {
                max_logit = logits[j];
                next_token_id = static_cast<long>(j);
            }
        }

        current_ids.push_back(next_token_id);

        if (i < 3 || i >= num_generate_tokens - 3) {
            std::cout << "  Token " << (i + 1) << ": " << token_time << " ms\n";
        } else if (i == 3) {
            std::cout << "  ...\n";
        }
    }

    double total_time = total_timer.elapsed_ms();

    std::cout << "\n  Decode total (" << num_generate_tokens << " tokens): "
              << total_decode_time << " ms\n";
    std::cout << "  Decode avg per token: "
              << (total_decode_time / num_generate_tokens) << " ms\n";
    std::cout << "  Total time: " << total_time << " ms\n";
}

int main() {
    print_header();

    try {
        // Path to model
        std::string model_path = "/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors";

        std::cout << "Loading model from: " << model_path << "\n";

        // Load weights
        Qwen3Weights weights = load_qwen3_weights(model_path);

        std::cout << "Model loaded successfully!\n\n";

        // Test scenarios: different prefill lengths
        std::vector<std::pair<int, int>> test_cases = {
            {4, 10},   // Short prefill, 10 tokens to generate
            {16, 10},  // Medium prefill
            {32, 10},  // Longer prefill
        };

        for (const auto& test_case : test_cases) {
            int prefill_len = test_case.first;
            int num_generate = test_case.second;

            std::cout << "\n============================================================\n";
            std::cout << "Test Case: prefill=" << prefill_len << ", generate=" << num_generate << "\n";
            std::cout << "============================================================\n";

            // Create input
            std::vector<long> input_ids_data;
            for (int i = 0; i < prefill_len; ++i) {
                input_ids_data.push_back(static_cast<long>(i % 1000));
            }
            TensorL input_ids(input_ids_data, Shape({1, static_cast<long>(prefill_len)}));

            // Benchmark without cache
            benchmark_without_cache("Baseline (no cache)", input_ids, weights, num_generate);

            // Benchmark with cache (baseline)
            benchmark_with_cache("Baseline (with cache)", input_ids, weights, num_generate, false);

            // Benchmark with cache (AVX2) - commented out due to bug
            // benchmark_with_cache("AVX2 (with cache)", input_ids, weights, num_generate, true);

            std::cout << "\n";
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
