/**
 * @file benchmark_qwen3_decode_versions.cpp
 * @brief Compare decoding performance of different Qwen3 versions
 */

#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/qwen3_ops_mpi.h"
#include "tensor_cpp/qwen3_ops_avx.h"
#include "tensor_cpp/qwen3_ops_mpi_avx.h"
#include "tensor_cpp/kv_cache.h"
#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

using namespace tensor_cpp;
using namespace tensor_cpp::qwen3;

// Timer helper
class Timer {
    std::chrono::high_resolution_clock::time_point start_;
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
};

TensorL sample_from_logits(const Tensor& logits, long temperature = 1.0f) {
    // Simple argmax sampling (can be improved with temperature)
    size_t batch = logits.shape()[0];
    size_t vocab_size = logits.shape()[2];

    std::vector<long> tokens(batch);
    for (size_t b = 0; b < batch; ++b) {
        float max_val = -1e30f;
        long max_idx = 0;
        for (size_t v = 0; v < vocab_size; ++v) {
            float val = logits[b * vocab_size + v];
            if (val > max_val) {
                max_val = val;
                max_idx = static_cast<long>(v);
            }
        }
        tokens[b] = max_idx;
    }

    // Convert to TensorL
    return TensorL(tokens, {static_cast<long>(batch), 1});
}

void print_header(int rank, int size) {
    if (rank == 0) {
        std::cout << "\n============================================================\n";
        std::cout << "     Qwen3 Performance Benchmark: Decoding\n";
        std::cout << "============================================================\n";
        std::cout << "MPI processes: " << size << "\n";
        #ifdef _OPENMP
        std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
        #endif
        std::cout << "============================================================\n\n";
    }
}

void benchmark_decoding(
    const std::string& version,
    const std::vector<long>& prompt_ids,
    int num_new_tokens,
    const Qwen3Weights& weights,
    int rank,
    MPI_Comm comm = MPI_COMM_WORLD
) {
    if (rank == 0) {
        std::cout << "----------------------------------------\n";
        std::cout << version << "\n";
        std::cout << "----------------------------------------\n";
    }

    MPI_Barrier(comm);
    Timer total_timer;

    // Prefill phase
    TensorL prompt_input(prompt_ids, Shape({1, static_cast<long>(prompt_ids.size())}));
    Tensor hidden_states;

    if (version == "Baseline") {
        hidden_states = qwen3::qwen3_forward(
            prompt_input,
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
        hidden_states = qwen3::avx2::qwen3_forward_avx(
            prompt_input,
            weights.embed_tokens,
            weights.layers,
            weights.norm_weight,
            weights.num_layers,
            weights.num_attention_heads,
            weights.num_key_value_heads,
            weights.head_dim,
            1e-6f
        );
    } else if (version == "MPI") {
        hidden_states = qwen3::mpi::qwen3_forward_mpi_omp(
            prompt_input,
            weights.embed_tokens,
            weights.layers,
            weights.norm_weight,
            weights.num_layers,
            weights.num_attention_heads,
            weights.num_key_value_heads,
            weights.head_dim,
            1e-6f,
            comm
        );
    } else if (version == "MPI+AVX2") {
        hidden_states = qwen3::mpi_avx::qwen3_forward_mpi_avx(
            prompt_input,
            weights.embed_tokens,
            weights.layers,
            weights.norm_weight,
            weights.num_layers,
            weights.num_attention_heads,
            weights.num_key_value_heads,
            weights.head_dim,
            1e-6f,
            comm
        );
    }

    double prefill_time = total_timer.elapsed_ms();

    // Compute LM head logits for prefill
    size_t vocab_size = weights.lm_head.shape()[0];
    size_t hidden_size = weights.hidden_size;

    // Simple projection to vocab
    std::vector<float> logits_data(vocab_size);
    for (size_t v = 0; v < vocab_size; ++v) {
        float sum = 0.0f;
        for (size_t h = 0; h < hidden_size; ++h) {
            sum += hidden_states[(prompt_ids.size() - 1) * hidden_size + h] * weights.lm_head[v * hidden_size + h];
        }
        logits_data[v] = sum;
    }
    Tensor logits(std::move(logits_data), Shape({1, 1, static_cast<long>(vocab_size)}));

    // Sample first token
    TensorL first_token = sample_from_logits(logits);
    long next_token = first_token[0];

    Timer decode_timer;
    std::vector<long> generated_tokens;
    generated_tokens.push_back(next_token);

    // KV cache (simplified - not fully functional)
    KVCache* kv_cache = nullptr;

    // Decode phase - generate new tokens
    for (int step = 0; step < num_new_tokens; ++step) {
        // Single token forward pass
        std::vector<long> single_token = {next_token};
        TensorL token_input(single_token, Shape({1, 1}));

        if (version == "Baseline") {
            // For baseline, run full forward (simplified)
            hidden_states = qwen3::qwen3_forward(
                token_input,
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
            hidden_states = qwen3::avx2::qwen3_forward_avx(
                token_input,
                weights.embed_tokens,
                weights.layers,
                weights.norm_weight,
                weights.num_layers,
                weights.num_attention_heads,
                weights.num_key_value_heads,
                weights.head_dim,
                1e-6f
            );
        } else if (version == "MPI") {
            hidden_states = qwen3::mpi::qwen3_forward_mpi_omp(
                token_input,
                weights.embed_tokens,
                weights.layers,
                weights.norm_weight,
                weights.num_layers,
                weights.num_attention_heads,
                weights.num_key_value_heads,
                weights.head_dim,
                1e-6f,
                comm
            );
        } else if (version == "MPI+AVX2") {
            hidden_states = qwen3::mpi_avx::qwen3_forward_mpi_avx(
                token_input,
                weights.embed_tokens,
                weights.layers,
                weights.norm_weight,
                weights.num_layers,
                weights.num_attention_heads,
                weights.num_key_value_heads,
                weights.head_dim,
                1e-6f,
                comm
            );
        }

        // Project to vocab and sample
        std::vector<float> logits_data(vocab_size);
        for (size_t v = 0; v < vocab_size; ++v) {
            float sum = 0.0f;
            for (size_t h = 0; h < hidden_size; ++h) {
                sum += hidden_states[h] * weights.lm_head[v * hidden_size + h];
            }
            logits_data[v] = sum;
        }
        Tensor logits(std::move(logits_data), Shape({1, 1, static_cast<long>(vocab_size)}));

        TensorL next_token_tensor = sample_from_logits(logits);
        next_token = next_token_tensor[0];
        generated_tokens.push_back(next_token);
    }

    double decode_time = decode_timer.elapsed_ms();
    double total_time = total_timer.elapsed_ms();

    if (rank == 0) {
        std::cout << "  Prefill time: " << std::fixed << std::setprecision(2) << prefill_time << " ms\n";
        std::cout << "  Decode time: " << decode_time << " ms\n";
        std::cout << "  Total time: " << total_time << " ms\n";
        std::cout << "  Tokens generated: " << num_new_tokens << "\n";
        std::cout << "  Time per token: " << (decode_time / num_new_tokens) << " ms\n";
        std::cout << "  Tokens per second: " << (num_new_tokens / decode_time) * 1000.0 << "\n";
        std::cout << "  Generated text: ";
        for (size_t i = 0; i < std::min(size_t(10), generated_tokens.size()); ++i) {
            std::cout << generated_tokens[i] << " ";
        }
        if (generated_tokens.size() > 10) std::cout << "...";
        std::cout << "\n";
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    print_header(rank, size);

    try {
        // Path to model
        std::string model_path = "/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors";

        if (rank == 0) {
            std::cout << "Loading model from: " << model_path << "\n";
        }

        // Load weights
        Qwen3Weights weights;
        if (rank == 0) {
            weights = load_qwen3_weights(model_path);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            std::cout << "Model loaded successfully!\n\n";
        }

        // Test prompts
        std::vector<std::pair<std::vector<long>, std::string>> test_cases = {
            {{100, 200, 300}, "Short prompt (3 tokens)"},
            {{100, 200, 300, 400, 500, 600, 700, 800}, "Medium prompt (8 tokens)"},
        };

        int num_new_tokens = 10;

        for (const auto& test_case : test_cases) {
            const auto& prompt_ids = test_case.first;
            const auto& description = test_case.second;

            if (rank == 0) {
                std::cout << "\n============================================================\n";
                std::cout << description << "\n";
                std::cout << "Generating " << num_new_tokens << " new tokens\n";
                std::cout << "============================================================\n";
            }

            // Benchmark each version
            if (size == 1) {
                benchmark_decoding("Baseline", prompt_ids, num_new_tokens, weights, rank);
                benchmark_decoding("AVX2", prompt_ids, num_new_tokens, weights, rank);
            } else {
                if (rank == 0) {
                    benchmark_decoding("Baseline", prompt_ids, num_new_tokens, weights, rank);
                    benchmark_decoding("AVX2", prompt_ids, num_new_tokens, weights, rank);
                }
                benchmark_decoding("MPI", prompt_ids, num_new_tokens, weights, rank, MPI_COMM_WORLD);
                benchmark_decoding("MPI+AVX2", prompt_ids, num_new_tokens, weights, rank, MPI_COMM_WORLD);
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }

        if (rank == 0) {
            std::cout << "\n============================================================\n";
            std::cout << "Benchmark completed!\n";
            std::cout << "============================================================\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Rank " << rank << " error: " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    MPI_Finalize();
    return 0;
}
