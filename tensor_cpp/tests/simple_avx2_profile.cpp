/**
 * @file simple_avx2_profile.cpp
 * @brief Simple AVX2 matmul performance test
 */

#include <immintrin.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>

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
    std::cout << "Simple AVX2 Matmul Profiling\n";
    std::cout << "========================================\n\n";

    // Test with Qwen3-0.6B MLP dimensions
    const size_t num_samples = 4;  // batch * seq_len
    const size_t in_features = 1024;
    const size_t out_features = 4096;

    std::cout << "Problem size: " << num_samples << " x " << in_features << " -> " << num_samples << " x " << out_features << "\n\n";

    // Create input and weight
    std::vector<float> input(num_samples * in_features);
    std::vector<float> weight(out_features * in_features);

    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i % 100) / 100.0f;
    }
    for (size_t i = 0; i < weight.size(); ++i) {
        weight[i] = static_cast<float>(i % 100) / 100.0f;
    }

    const int warmup = 3;
    const int iterations = 20;

    std::cout << "Warming up...\n";

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        std::vector<float> output_baseline(num_samples * out_features, 0.0f);
        for (size_t s = 0; s < num_samples; ++s) {
            for (size_t o = 0; o < out_features; ++o) {
                float sum = 0.0f;
                for (size_t j = 0; j < in_features; ++j) {
                    sum += input[s * in_features + j] * weight[o * in_features + j];
                }
                output_baseline[s * out_features + o] = sum;
            }
        }
    }

    std::cout << "Benchmarking (" << iterations << " iterations)...\n\n";

    // Benchmark baseline (scalar)
    {
        std::vector<double> times;
        for (int i = 0; i < iterations; ++i) {
            std::vector<float> output(num_samples * out_features, 0.0f);
            Timer timer;
            #pragma omp parallel for if(num_samples * out_features > 100)
            for (size_t s = 0; s < num_samples; ++s) {
                for (size_t o = 0; o < out_features; ++o) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < in_features; ++j) {
                        sum += input[s * in_features + j] * weight[o * in_features + j];
                    }
                    output[s * out_features + o] = sum;
                }
            }
            times.push_back(timer.elapsed_ms());
        }
        double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        double min = *std::min_element(times.begin(), times.end());
        std::cout << "Baseline (scalar + OpenMP):\n";
        std::cout << "  Avg: " << std::fixed << std::setprecision(2) << avg << " ms\n";
        std::cout << "  Min: " << min << " ms\n";
        std::cout << "  Perf: " << (2.0 * num_samples * out_features * in_features / avg) / 1e9 << " GFLOPS\n\n";
    }

    // Benchmark AVX2
    {
        std::vector<double> times;
        for (int i = 0; i < iterations; ++i) {
            std::vector<float> output(num_samples * out_features, 0.0f);
            Timer timer;
            #pragma omp parallel for if(num_samples * out_features > 100)
            for (size_t s = 0; s < num_samples; ++s) {
                for (size_t o = 0; o < out_features; ++o) {
                    float sum = 0.0f;
                    size_t weight_offset = o * in_features;
                    size_t j = 0;

                    // AVX2 dot product
                    __m256 sum_vec = _mm256_setzero_ps();
                    for (; j + 8 <= in_features; j += 8) {
                        __m256 input_vec = _mm256_loadu_ps(&input[s * in_features + j]);
                        __m256 weight_vec = _mm256_loadu_ps(&weight[weight_offset + j]);
                        sum_vec = _mm256_fmadd_ps(input_vec, weight_vec, sum_vec);
                    }

                    // Horizontal sum
                    sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
                    sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
                    float temp[8];
                    _mm256_storeu_ps(temp, sum_vec);
                    sum = temp[0] + temp[4];

                    // Handle remaining elements
                    for (; j < in_features; ++j) {
                        sum += input[s * in_features + j] * weight[weight_offset + j];
                    }

                    output[s * out_features + o] = sum;
                }
            }
            times.push_back(timer.elapsed_ms());
        }
        double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        double min = *std::min_element(times.begin(), times.end());
        std::cout << "AVX2 (with OpenMP):\n";
        std::cout << "  Avg: " << std::fixed << std::setprecision(2) << avg << " ms\n";
        std::cout << "  Min: " << min << " ms\n";
        std::cout << "  Perf: " << (2.0 * num_samples * out_features * in_features / avg) / 1e9 << " GFLOPS\n\n";
    }

    // Benchmark improved AVX2 (better horizontal sum)
    {
        std::vector<double> times;
        for (int i = 0; i < iterations; ++i) {
            std::vector<float> output(num_samples * out_features, 0.0f);
            Timer timer;
            #pragma omp parallel for if(num_samples * out_features > 100)
            for (size_t s = 0; s < num_samples; ++s) {
                for (size_t o = 0; o < out_features; ++o) {
                    float sum = 0.0f;
                    size_t weight_offset = o * in_features;
                    size_t j = 0;

                    // AVX2 dot product with better horizontal sum
                    __m256 sum_vec = _mm256_setzero_ps();
                    for (; j + 8 <= in_features; j += 8) {
                        __m256 input_vec = _mm256_loadu_ps(&input[s * in_features + j]);
                        __m256 weight_vec = _mm256_loadu_ps(&weight[weight_offset + j]);
                        sum_vec = _mm256_fmadd_ps(input_vec, weight_vec, sum_vec);
                    }

                    // Better horizontal sum using shuffle
                    __m128 hi_quad = _mm256_extractf128_ps(sum_vec, 1);
                    __m128 lo_quad = _mm256_castps256_ps128(sum_vec);
                    __m128 sum_quad = _mm_add_ps(lo_quad, hi_quad);
                    __m128 lo_dual = sum_quad;
                    __m128 hi_dual = _mm_movehl_ps(sum_quad, sum_quad);
                    __m128 sum_dual = _mm_add_ps(lo_dual, hi_dual);
                    __m128 lo = sum_dual;
                    __m128 hi = _mm_shuffle_ps(sum_dual, sum_dual, 1);
                    __m128 sum_128 = _mm_add_ss(lo, hi);
                    sum = _mm_cvtss_f32(sum_128);

                    // Handle remaining elements
                    for (; j < in_features; ++j) {
                        sum += input[s * in_features + j] * weight[weight_offset + j];
                    }

                    output[s * out_features + o] = sum;
                }
            }
            times.push_back(timer.elapsed_ms());
        }
        double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        double min = *std::min_element(times.begin(), times.end());
        std::cout << "AVX2 (improved horizontal sum):\n";
        std::cout << "  Avg: " << std::fixed << std::setprecision(2) << avg << " ms\n";
        std::cout << "  Min: " << min << " ms\n";
        std::cout << "  Perf: " << (2.0 * num_samples * out_features * in_features / avg) / 1e9 << " GFLOPS\n\n";
    }

    std::cout << "========================================\n";
    std::cout << "Profiling completed!\n";
    std::cout << "========================================\n";

    return 0;
}
