/**
 * @file profile_avx2.cpp
 * @brief Profile AVX2 MLP implementation to find bottlenecks
 */

#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/qwen3_ops_avx.h"
#include "tensor_cpp/ops.h"
#include "tensor_cpp/ops_avx.h"
#include <immintrin.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <algorithm>

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

int main() {
    std::cout << "\n========================================\n";
    std::cout << "AVX2 MLP Profiling\n";
    std::cout << "========================================\n\n";

    // Test with Qwen3-0.6B dimensions
    size_t batch = 1;
    size_t seq_len = 4;
    size_t hidden_size = 1024;
    size_t intermediate_size = 4096;

    // Create input
    std::vector<float> hidden_data(batch * seq_len * hidden_size);
    for (size_t i = 0; i < hidden_data.size(); ++i) {
        hidden_data[i] = static_cast<float>(i % 100) / 100.0f;
    }
    Tensor hidden_states(std::move(hidden_data), Shape({static_cast<long>(batch), static_cast<long>(seq_len), static_cast<long>(hidden_size)}));

    // Create weights
    std::vector<float> gate_data(intermediate_size * hidden_size);
    std::vector<float> up_data(intermediate_size * hidden_size);
    std::vector<float> down_data(hidden_size * intermediate_size);
    for (size_t i = 0; i < intermediate_size * hidden_size; ++i) {
        gate_data[i] = static_cast<float>(i % 100) / 100.0f;
        up_data[i] = static_cast<float>(i % 100) / 100.0f;
    }
    for (size_t i = 0; i < hidden_size * intermediate_size; ++i) {
        down_data[i] = static_cast<float>(i % 100) / 100.0f;
    }

    Tensor gate_proj(std::move(gate_data), Shape({static_cast<long>(intermediate_size), static_cast<long>(hidden_size)}));
    Tensor up_proj(std::move(up_data), Shape({static_cast<long>(intermediate_size), static_cast<long>(hidden_size)}));
    Tensor down_proj(std::move(down_data), Shape({static_cast<long>(hidden_size), static_cast<long>(intermediate_size)}));

    const int warmup = 3;
    const int iterations = 10;

    std::cout << "Warming up (" << warmup << " iterations)...\n\n";

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        Tensor output1 = qwen3_mlp(hidden_states, gate_proj, up_proj, down_proj);
        Tensor output2 = avx2::qwen3_mlp_avx(hidden_states, gate_proj, up_proj, down_proj);
    }

    std::cout << "Benchmarking MLP (" << iterations << " iterations)...\n\n";

    // Benchmark baseline MLP
    {
        std::vector<double> times;
        for (int i = 0; i < iterations; ++i) {
            Timer timer;
            Tensor output = qwen3_mlp(hidden_states, gate_proj, up_proj, down_proj);
            times.push_back(timer.elapsed_ms());
        }
        double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        double min = *std::min_element(times.begin(), times.end());
        double max = *std::max_element(times.begin(), times.end());
        std::cout << "Baseline MLP:\n";
        std::cout << "  Avg: " << std::fixed << std::setprecision(2) << avg << " ms\n";
        std::cout << "  Min: " << min << " ms\n";
        std::cout << "  Max: " << max << " ms\n\n";
    }

    // Benchmark AVX2 MLP
    {
        std::vector<double> times;
        for (int i = 0; i < iterations; ++i) {
            Timer timer;
            Tensor output = avx2::qwen3_mlp_avx(hidden_states, gate_proj, up_proj, down_proj);
            times.push_back(timer.elapsed_ms());
        }
        double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        double min = *std::min_element(times.begin(), times.end());
        double max = *std::max_element(times.begin(), times.end());
        std::cout << "AVX2 MLP:\n";
        std::cout << "  Avg: " << std::fixed << std::setprecision(2) << avg << " ms\n";
        std::cout << "  Min: " << min << " ms\n";
        std::cout << "  Max: " << max << " ms\n\n";
    }

    // Benchmark individual linear operations
    std::cout << "Benchmarking individual linear operations:\n\n";

    // Reshape input for linear layer
    Tensor hidden_reshaped = hidden_states.view({static_cast<long>(batch * seq_len), static_cast<long>(hidden_size)});

    // Baseline linear
    {
        std::vector<double> times;
        for (int i = 0; i < iterations; ++i) {
            Timer timer;
            Tensor output = ops::linear(hidden_reshaped, gate_proj, nullptr);
            times.push_back(timer.elapsed_ms());
        }
        double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        std::cout << "Baseline linear (1024->4096): " << avg << " ms\n";
    }

    // AVX2 linear (matmul)
    {
        std::vector<double> times;
        for (int i = 0; i < iterations; ++i) {
            Timer timer;
            std::vector<float> output_data(batch * seq_len * intermediate_size);
            #pragma omp parallel for if(batch * seq_len * intermediate_size > 1000)
            for (size_t s = 0; s < batch * seq_len; ++s) {
                for (size_t i = 0; i < intermediate_size; ++i) {
                    float sum = 0.0f;
                    size_t weight_offset = i * hidden_size;
                    size_t j = 0;
                    __m256 sum_vec = _mm256_setzero_ps();
                    for (; j + 8 <= hidden_size; j += 8) {
                        __m256 hidden_vec = _mm256_loadu_ps(&hidden_reshaped[s * hidden_size + j]);
                        __m256 weight_vec = _mm256_loadu_ps(&gate_proj[weight_offset + j]);
                        sum_vec = _mm256_fmadd_ps(hidden_vec, weight_vec, sum_vec);
                    }
                    sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
                    sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
                    float temp[8];
                    _mm256_storeu_ps(temp, sum_vec);
                    sum = temp[0] + temp[4];
                    for (; j < hidden_size; ++j) {
                        sum += hidden_reshaped[s * hidden_size + j] * gate_proj[weight_offset + j];
                    }
                    output_data[s * intermediate_size + i] = sum;
                }
            }
            times.push_back(timer.elapsed_ms());
        }
        double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        std::cout << "AVX2 linear (1024->4096): " << avg << " ms\n";
    }

    std::cout << "\n========================================\n";
    std::cout << "Profiling completed!\n";
    std::cout << "========================================\n";

    return 0;
}
