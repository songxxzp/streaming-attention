/**
 * @file benchmark_kv_cache_correct.cpp
 * @brief Correctly benchmark KV cache speedup
 *
 * This test properly demonstrates KV cache speedup by comparing:
 * - Without cache: Recompute entire sequence each time (O(n²))
 * - With cache: Only compute new token, reuse cached KV (O(n))
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
    std::cout << "     KV Cache Speedup Benchmark (Correct Method)\n";
    std::cout << "============================================================\n";
    #ifdef _OPENMP
    std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
    #endif
    std::cout << "============================================================\n\n";
}

// WITHOUT KV CACHE: Recompute entire sequence each time (O(n²))
double benchmark_without_cache(
    const TensorL& initial_input,
    const Qwen3Weights& weights,
    int total_tokens
) {
    std::cout << "  WITHOUT KV CACHE (recompute everything each time):\n";

    Timer total_timer;

    // Initial forward pass
    Timer timer;
    Tensor output = qwen3::qwen3_forward(
        initial_input,
        weights.embed_tokens,
        weights.layers,
        weights.norm_weight,
        weights.num_layers,
        weights.num_attention_heads,
        weights.num_key_value_heads,
        weights.head_dim,
        1e-6f
    );
    double first_time = timer.elapsed_ms();

    std::cout << "    Initial " << initial_input.shape()[1] << " tokens: " << first_time << " ms\n";

    // Autoregressive generation WITHOUT cache
    // Each step recomputes ALL previous tokens
    std::vector<long> all_ids(initial_input.data(), initial_input.data() + initial_input.size());

    double generation_time = 0.0;
    for (int i = static_cast<int>(initial_input.shape()[1]); i < total_tokens; ++i) {
        // Sample next token (simple argmax)
        size_t vocab_size = weights.lm_head.shape()[0];
        float max_logit = -std::numeric_limits<float>::infinity();
        long next_token_id = 0;

        const float* logits = output.data();
        for (size_t j = 0; j < vocab_size; ++j) {
            if (logits[j] > max_logit) {
                max_logit = logits[j];
                next_token_id = static_cast<long>(j);
            }
        }
        all_ids.push_back(next_token_id);

        // Recompute ENTIRE sequence (this is the key difference!)
        TensorL full_input(all_ids, Shape({1, static_cast<long>(all_ids.size())}));

        timer = Timer();
        output = qwen3::qwen3_forward(
            full_input,
            weights.embed_tokens,
            weights.layers,
            weights.norm_weight,
            weights.num_layers,
            weights.num_attention_heads,
            weights.num_key_value_heads,
            weights.head_dim,
            1e-6f
        );
        double step_time = timer.elapsed_ms();
        generation_time += step_time;

        if ((i - initial_input.shape()[1]) < 3 || i >= total_tokens - 3) {
            std::cout << "    Step " << (i + 1) << " (" << all_ids.size() << " tokens): "
                      << step_time << " ms\n";
        } else if ((i - initial_input.shape()[1]) == 3) {
            std::cout << "    ...\n";
        }
    }

    double total_time = total_timer.elapsed_ms();

    std::cout << "    Generation time: " << generation_time << " ms\n";
    std::cout << "    Total time: " << total_time << " ms\n";

    return total_time;
}

// WITH KV CACHE: Only compute new token (O(n))
double benchmark_with_cache(
    const TensorL& initial_input,
    Qwen3Weights& weights,
    int total_tokens,
    bool use_avx = false
) {
    std::cout << "  WITH KV CACHE" << (use_avx ? " + AVX2" : "") << " (only compute new tokens):\n";

    Timer total_timer;

    // Create KV cache
    KVCache kv_cache(
        weights.num_layers,
        1,  // batch_size
        weights.num_key_value_heads,
        weights.head_dim,
        4096  // max_seq_len
    );

    // Initial forward pass with cache
    Timer timer;
    Tensor output;
    if (use_avx) {
        output = avx2::qwen3_forward_avx_with_cache(
            initial_input,
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
            initial_input,
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
    double first_time = timer.elapsed_ms();

    std::cout << "    Initial " << initial_input.shape()[1] << " tokens: " << first_time << " ms\n";

    // Autoregressive generation WITH cache
    std::vector<long> all_ids(initial_input.data(), initial_input.data() + initial_input.size());

    double generation_time = 0.0;
    for (int i = static_cast<int>(initial_input.shape()[1]); i < total_tokens; ++i) {
        // Sample next token
        size_t vocab_size = weights.lm_head.shape()[0];
        float max_logit = -std::numeric_limits<float>::infinity();
        long next_token_id = 0;

        const float* logits = output.data();
        for (size_t j = 0; j < vocab_size; ++j) {
            if (logits[j] > max_logit) {
                max_logit = logits[j];
                next_token_id = static_cast<long>(j);
            }
        }
        all_ids.push_back(next_token_id);

        // Compute ONLY the new token (cache handles the rest)
        std::vector<long> single_token = {next_token_id};
        TensorL next_input(single_token, Shape({1, 1}));

        timer = Timer();
        if (use_avx) {
            output = avx2::qwen3_forward_avx_with_cache(
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
            output = qwen3::qwen3_forward_with_cache(
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
        double step_time = timer.elapsed_ms();
        generation_time += step_time;

        if ((i - initial_input.shape()[1]) < 3 || i >= total_tokens - 3) {
            std::cout << "    Step " << (i + 1) << " (1 new token): "
                      << step_time << " ms\n";
        } else if ((i - initial_input.shape()[1]) == 3) {
            std::cout << "    ...\n";
        }
    }

    double total_time = total_timer.elapsed_ms();

    std::cout << "    Generation time: " << generation_time << " ms\n";
    std::cout << "    Total time: " << total_time << " ms\n";

    return total_time;
}

int main() {
    print_header();

    try {
        std::string model_path = "/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors";

        std::cout << "Loading model from: " << model_path << "\n";
        Qwen3Weights weights = load_qwen3_weights(model_path);
        std::cout << "Model loaded successfully!\n\n";

        // Test different generation lengths
        std::vector<int> generate_lengths = {10, 15, 20};

        for (int initial_len : {4}) {
            for (int total_tokens : generate_lengths) {
                std::cout << "\n============================================================\n";
                std::cout << "Test: Initial " << initial_len << " tokens, generate "
                         << (total_tokens - initial_len) << " more (total " << total_tokens << ")\n";
                std::cout << "============================================================\n";

                // Create initial input
                std::vector<long> input_ids_data;
                for (int i = 0; i < initial_len; ++i) {
                    input_ids_data.push_back(static_cast<long>(i % 1000));
                }
                TensorL initial_input(input_ids_data, Shape({1, initial_len}));

                // Benchmark WITHOUT cache (recompute everything)
                double time_without = benchmark_without_cache(initial_input, weights, total_tokens);

                std::cout << "\n";

                // Benchmark WITH cache (only compute new tokens)
                double time_with = benchmark_with_cache(initial_input, weights, total_tokens);

                // Benchmark WITH cache + AVX2
                std::cout << "\n";
                double time_with_avx = benchmark_with_cache(initial_input, weights, total_tokens, true);

                // Calculate speedup
                double speedup = time_without / time_with;
                double speedup_avx = time_without / time_with_avx;

                std::cout << "\n";
                std::cout << "  >>> SPEEDUP (Baseline): " << std::fixed << std::setprecision(2) << speedup << "x <<<\n";
                std::cout << "  >>> SPEEDUP (AVX2): " << speedup_avx << "x <<<\n";
                std::cout << "  (Without cache: " << time_without << " ms, With cache: " << time_with << " ms, With cache+AVX2: " << time_with_avx << " ms)\n";
                std::cout << "\n";
            }
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
