/**
 * Correctness Test Suite for Phase 1
 *
 * Tests:
 * 1. Naive Attention basic functionality
 * 2. Streaming Attention basic functionality
 * 3. Numerical equivalence between Naive and Streaming
 * 4. Performance comparison
 */

#include "attention/attention.h"
#include "utils/timer.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>

// Test configuration
struct TestConfig {
    int T;          // Sequence length
    int d;          // Hidden dimension
    int block_size; // Block size for streaming
    int warmup;     // Warmup iterations
    int repeat;     // Timing iterations
};

// Generate random data
void generate_random_data(float* data, int size, float min = -1.0f, float max = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);

    for (int i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
}

// Run a single test case
void run_test(const TestConfig& config) {
    std::cout << "\n========== Test Case: T=" << config.T
              << ", d=" << config.d
              << ", block_size=" << config.block_size << " ==========\n";

    // Allocate and generate data
    std::vector<float> Q(config.d);
    std::vector<float> K(config.T * config.d);
    std::vector<float> V(config.T * config.d);

    generate_random_data(Q.data(), config.d, -0.1f, 0.1f);  // Small values for Q
    generate_random_data(K.data(), config.T * config.d);
    generate_random_data(V.data(), config.T * config.d);

    // Run naive attention (baseline)
    Timer timer_naive;
    for (int i = 0; i < config.warmup; ++i) {
        auto result = naive_attention_serial(Q.data(), K.data(), V.data(), config.T, config.d);
    }

    timer_naive.reset();
    timer_naive.start();
    for (int i = 0; i < config.repeat; ++i) {
        auto result = naive_attention_serial(Q.data(), K.data(), V.data(), config.T, config.d);
    }
    timer_naive.stop();

    auto output_naive = naive_attention_serial(Q.data(), K.data(), V.data(), config.T, config.d);

    // Run streaming attention
    Timer timer_streaming;
    for (int i = 0; i < config.warmup; ++i) {
        auto result = streaming_attention_serial(Q.data(), K.data(), V.data(),
                                                  config.T, config.d, config.block_size);
    }

    timer_streaming.reset();
    timer_streaming.start();
    for (int i = 0; i < config.repeat; ++i) {
        auto result = streaming_attention_serial(Q.data(), K.data(), V.data(),
                                                  config.T, config.d, config.block_size);
    }
    timer_streaming.stop();

    auto output_streaming = streaming_attention_serial(Q.data(), K.data(), V.data(),
                                                        config.T, config.d, config.block_size);

    // Verify correctness
    float l2_err = compute_l2_error(output_naive.data(), output_streaming.data(), config.d);
    float max_err = compute_max_error(output_naive.data(), output_streaming.data(), config.d);

    // Compute performance metrics
    double time_naive_ms = timer_naive.elapsed() / config.repeat * 1000.0;
    double time_streaming_ms = timer_streaming.elapsed() / config.repeat * 1000.0;

    // FLOPs: 2*T*d for QK^T, T*d for softmax@V
    double flops = 3.0 * config.T * config.d;
    double bytes = 2.0 * config.T * config.d * sizeof(float);  // Read K, V

    double bandwidth_naive = bytes / time_naive_ms / 1e6;
    double bandwidth_streaming = bytes / time_streaming_ms / 1e6;
    double gflops_naive = flops / time_naive_ms / 1e6;
    double gflops_streaming = flops / time_streaming_ms / 1e6;

    // Print results
    std::cout << std::scientific << std::setprecision(2);
    std::cout << "Correctness:\n";
    std::cout << "  L2 Error:  " << l2_err << "\n";
    std::cout << "  Max Error: " << max_err << "\n";

    if (l2_err < 1e-4 && max_err < 1e-5) {
        std::cout << "  ✓ PASSED: Outputs match within tolerance\n";
    } else {
        std::cout << "  ✗ FAILED: Outputs differ significantly\n";
    }

    Timer::print_header("Performance");
    Timer::print_row("Naive (Serial)", time_naive_ms, bandwidth_naive, gflops_naive);
    Timer::print_row("Streaming (Serial)", time_streaming_ms, bandwidth_streaming, gflops_streaming);

    double speedup = time_naive_ms / time_streaming_ms;
    std::cout << "\nSpeedup (Streaming vs Naive): " << std::fixed << std::setprecision(2) << speedup << "x\n";
}

int main(int argc, char** argv) {
    std::cout << "============================================================\n";
    std::cout << "  Phase 1: Serial Correctness Validation\n";
    std::cout << "============================================================\n";

    // Default configuration
    TestConfig config;
    config.T = 2048;
    config.d = 128;
    config.block_size = 64;
    config.warmup = 5;
    config.repeat = 20;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--T" && i + 1 < argc) {
            config.T = std::atoi(argv[++i]);
        } else if (arg == "--d" && i + 1 < argc) {
            config.d = std::atoi(argv[++i]);
        } else if (arg == "--block" && i + 1 < argc) {
            config.block_size = std::atoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --T N          Sequence length (default: 2048)\n"
                      << "  --d N          Hidden dimension (default: 128)\n"
                      << "  --block N      Block size (default: 64)\n";
            return 0;
        }
    }

    // Run test suite
    std::vector<TestConfig> test_configs = {
        {512, 64, 32, 5, 20},
        {1024, 128, 64, 5, 20},
        {2048, 256, 128, 5, 20},
        {4096, 128, 64, 5, 20},
        config,  // User-specified config
    };

    for (const auto& cfg : test_configs) {
        run_test(cfg);
    }

    std::cout << "\n============================================================\n";
    std::cout << "  Phase 1 Complete: All correctness tests finished\n";
    std::cout << "============================================================\n";

    return 0;
}
