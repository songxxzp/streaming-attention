/**
 * Phase 2: OpenMP Parallelization Test Suite
 *
 * Tests:
 * 1. Correctness: OMP vs Serial
 * 2. Thread scaling: 1, 2, 4, 8, ... threads
 * 3. Block size sensitivity
 * 4. NUMA-aware vs non-NUMA
 */

#include "attention/attention.h"
#include "utils/timer.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <omp.h>

struct TestConfig {
    int T;
    int d;
    int block_size;
    int warmup;
    int repeat;
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

/**
 * NUMA-aware data allocation using first-touch policy
 *
 * The key idea: each thread initializes the portion of data it will access.
 * This causes the OS to allocate pages on the NUMA node where the thread runs.
 */
void initialize_data_numa_aware(float* K, float* V, int T, int d) {
    // First, seed all random generators serially to avoid issues
    std::vector<std::mt19937::result_type> seeds;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int n_threads = omp_get_num_threads();

        #pragma omp single
        {
            seeds.resize(n_threads);
            std::random_device rd;
            for (int i = 0; i < n_threads; ++i) {
                seeds[i] = rd();
            }
        }

        #pragma omp barrier

        // Each thread initializes its chunk
        int chunk_size = (T + n_threads - 1) / n_threads;
        int start = tid * chunk_size;
        int end = std::min(start + chunk_size, T);

        // Use thread-local generator with pre-computed seed
        std::mt19937 gen(seeds[tid]);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (int i = start; i < end; ++i) {
            for (int j = 0; j < d; ++j) {
                K[i * d + j] = dist(gen);
                V[i * d + j] = dist(gen);
            }
        }
    }
}

// Run thread scaling experiment
void run_thread_scaling(const TestConfig& config, const float* Q, const float* K, const float* V) {
    std::cout << "\n========== Thread Scaling (T=" << config.T
              << ", d=" << config.d << ", block=" << config.block_size << ") ==========\n";

    // Get reference (serial) result
    auto ref_output = streaming_attention_serial(Q, K, V, config.T, config.d, config.block_size);

    // Test different thread counts
    std::vector<int> thread_counts = {1, 2, 4, 8, 16, 32};
    std::vector<std::string> labels = {"Serial", "OMP-1", "OMP-2", "OMP-4", "OMP-8", "OMP-16", "OMP-32"};

    Timer::print_header("Thread Scaling");

    // First, test serial version
    Timer timer_serial;
    for (int i = 0; i < config.warmup; ++i) {
        auto result = streaming_attention_serial(Q, K, V, config.T, config.d, config.block_size);
    }
    timer_serial.reset();
    timer_serial.start();
    for (int i = 0; i < config.repeat; ++i) {
        auto result = streaming_attention_serial(Q, K, V, config.T, config.d, config.block_size);
    }
    timer_serial.stop();
    double time_serial = timer_serial.elapsed() / config.repeat * 1000.0;
    double flops = 3.0 * config.T * config.d;
    double bandwidth = 2.0 * config.T * config.d * sizeof(float);
    Timer::print_row("Serial", time_serial, bandwidth / time_serial / 1e6, flops / time_serial / 1e6);

    // Now test OMP with different thread counts
    for (int n_threads : thread_counts) {
        Timer timer;

        for (int i = 0; i < config.warmup; ++i) {
            auto result = streaming_attention_omp(Q, K, V, config.T, config.d, config.block_size, n_threads);
        }

        timer.reset();
        timer.start();
        for (int i = 0; i < config.repeat; ++i) {
            auto result = streaming_attention_omp(Q, K, V, config.T, config.d, config.block_size, n_threads);
        }
        timer.stop();

        auto omp_output = streaming_attention_omp(Q, K, V, config.T, config.d, config.block_size, n_threads);

        double time_ms = timer.elapsed() / config.repeat * 1000.0;
        double speedup = time_serial / time_ms;

        // Check correctness
        float l2_err = compute_l2_error(ref_output.data(), omp_output.data(), config.d);

        std::string label = "OMP-" + std::to_string(n_threads);
        std::cout << std::setw(30) << std::left << label
                  << std::setw(15) << std::right << std::fixed << std::setprecision(3) << time_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << bandwidth / time_ms / 1e6
                  << std::setw(15) << std::fixed << std::setprecision(2) << flops / time_ms / 1e6
                  << " | " << std::setprecision(2) << speedup << "x"
                  << " (L2: " << std::scientific << l2_err << ")" << "\n";
    }
}

// Run block size experiment
void run_block_size_experiment(const TestConfig& base_config, const float* Q, const float* K, const float* V) {
    std::cout << "\n========== Block Size Experiment (T=" << base_config.T
              << ", d=" << base_config.d << ") ==========\n";

    auto ref_output = streaming_attention_serial(Q, K, V, base_config.T, base_config.d, base_config.block_size);

    std::vector<int> block_sizes = {16, 32, 64, 128, 256, 512};

    std::cout << std::left << std::setw(20) << "Block Size"
              << std::right << std::setw(15) << "Time (ms)"
              << std::setw(15) << "Speedup"
              << std::setw(15) << "L2 Error" << "\n";
    std::cout << std::string(65, '-') << "\n";

    // Baseline serial time
    Timer timer_base;
    for (int i = 0; i < base_config.warmup; ++i) {
        auto result = streaming_attention_serial(Q, K, V, base_config.T, base_config.d, base_config.block_size);
    }
    timer_base.reset();
    timer_base.start();
    for (int i = 0; i < base_config.repeat; ++i) {
        auto result = streaming_attention_serial(Q, K, V, base_config.T, base_config.d, base_config.block_size);
    }
    timer_base.stop();
    double time_base = timer_base.elapsed() / base_config.repeat * 1000.0;
    std::cout << std::left << std::setw(20) << "Serial"
              << std::right << std::setw(15) << std::fixed << std::setprecision(3) << time_base
              << std::setw(15) << "1.00"
              << std::setw(15) << "-" << "\n";

    // Test different block sizes with OMP
    int n_threads = omp_get_max_threads();

    for (int block_size : block_sizes) {
        Timer timer;

        for (int i = 0; i < base_config.warmup; ++i) {
            auto result = streaming_attention_omp(Q, K, V, base_config.T, base_config.d, block_size, n_threads);
        }

        timer.reset();
        timer.start();
        for (int i = 0; i < base_config.repeat; ++i) {
            auto result = streaming_attention_omp(Q, K, V, base_config.T, base_config.d, block_size, n_threads);
        }
        timer.stop();

        auto omp_output = streaming_attention_omp(Q, K, V, base_config.T, base_config.d, block_size, n_threads);

        double time_ms = timer.elapsed() / base_config.repeat * 1000.0;
        double speedup = time_base / time_ms;
        float l2_err = compute_l2_error(ref_output.data(), omp_output.data(), base_config.d);

        std::cout << std::left << std::setw(20) << ("OMP-" + std::to_string(block_size))
                  << std::right << std::setw(15) << std::fixed << std::setprecision(3) << time_ms
                  << std::setw(15) << std::setprecision(2) << speedup
                  << std::setw(15) << std::scientific << l2_err << "\n";
    }
}

// Run NUMA-aware vs non-NUMA comparison
void run_numa_comparison(const TestConfig& config) {
    std::cout << "\n========== NUMA-aware vs Regular Allocation ==========\n";

    // Allocate data
    std::vector<float> Q(config.d);
    std::vector<float> K(config.T * config.d);
    std::vector<float> V(config.T * config.d);

    // Generate Q (small, doesn't matter)
    generate_random_data(Q.data(), config.d, -0.1f, 0.1f);

    // Test 1: Regular allocation (initialized by single thread)
    generate_random_data(K.data(), config.T * config.d);
    generate_random_data(V.data(), config.T * config.d);

    Timer timer_regular;
    int n_threads = omp_get_max_threads();

    for (int i = 0; i < config.warmup; ++i) {
        auto result = streaming_attention_omp(Q.data(), K.data(), V.data(),
                                              config.T, config.d, config.block_size, n_threads);
    }

    timer_regular.reset();
    timer_regular.start();
    for (int i = 0; i < config.repeat; ++i) {
        auto result = streaming_attention_omp(Q.data(), K.data(), V.data(),
                                              config.T, config.d, config.block_size, n_threads);
    }
    timer_regular.stop();

    // Test 2: NUMA-aware allocation (first-touch)
    initialize_data_numa_aware(K.data(), V.data(), config.T, config.d);

    Timer timer_numa;
    for (int i = 0; i < config.warmup; ++i) {
        auto result = streaming_attention_omp(Q.data(), K.data(), V.data(),
                                              config.T, config.d, config.block_size, n_threads);
    }

    timer_numa.reset();
    timer_numa.start();
    for (int i = 0; i < config.repeat; ++i) {
        auto result = streaming_attention_omp(Q.data(), K.data(), V.data(),
                                              config.T, config.d, config.block_size, n_threads);
    }
    timer_numa.stop();

    double time_regular = timer_regular.elapsed() / config.repeat * 1000.0;
    double time_numa = timer_numa.elapsed() / config.repeat * 1000.0;

    std::cout << "Regular allocation:  " << std::fixed << std::setprecision(3) << time_regular << " ms\n";
    std::cout << "NUMA-aware alloc:    " << std::fixed << std::setprecision(3) << time_numa << " ms\n";
    std::cout << "Improvement:         " << std::setprecision(2) << (time_regular / time_numa) << "x\n";
}

void print_system_info() {
    std::cout << "\n========== System Information ==========\n";
    std::cout << "Max OpenMP threads: " << omp_get_max_threads() << "\n";
    std::cout << "Num procs: " << omp_get_num_procs() << "\n";

    // Try to get NUMA info
    #ifdef _NUMA_H
    std::cout << "NUMA available: Yes\n";
    #else
    std::cout << "NUMA available: Using first-touch policy\n";
    #endif
}

int main(int argc, char** argv) {
    std::cout << "============================================================\n";
    std::cout << "  Phase 2: OpenMP Parallelization\n";
    std::cout << "============================================================\n";

    print_system_info();

    // Default configuration
    TestConfig config;
    config.T = 4096;
    config.d = 128;
    config.block_size = 64;
    config.warmup = 5;
    config.repeat = 20;

    // Parse command line
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--T" && i + 1 < argc) config.T = std::atoi(argv[++i]);
        else if (arg == "--d" && i + 1 < argc) config.d = std::atoi(argv[++i]);
        else if (arg == "--block" && i + 1 < argc) config.block_size = std::atoi(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc) omp_set_num_threads(std::atoi(argv[++i]));
        else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --T N          Sequence length\n"
                      << "  --d N          Hidden dimension\n"
                      << "  --block N      Block size\n"
                      << "  --threads N    Set OpenMP thread count\n";
            return 0;
        }
    }

    // Run experiments
    std::vector<TestConfig> test_configs = {
        {2048, 128, 64, 5, 20},
        {4096, 128, 64, 5, 20},
        {8192, 256, 128, 5, 20},
        config,
    };

    // Find maximum dimensions needed
    int max_T = 0, max_d = 0;
    for (const auto& cfg : test_configs) {
        max_T = std::max(max_T, cfg.T);
        max_d = std::max(max_d, cfg.d);
    }
    // Also account for block size experiments
    max_d = std::max(max_d, config.d);

    // Allocate data with maximum size needed
    std::vector<float> Q(max_d);
    std::vector<float> K(max_T * max_d);
    std::vector<float> V(max_T * max_d);

    generate_random_data(Q.data(), max_d, -0.1f, 0.1f);
    initialize_data_numa_aware(K.data(), V.data(), max_T, max_d);

    for (const auto& cfg : test_configs) {
        run_thread_scaling(cfg, Q.data(), K.data(), V.data());
    }

    // Block size experiment
    run_block_size_experiment(config, Q.data(), K.data(), V.data());

    // NUMA comparison (only if meaningful)
    if (omp_get_max_threads() > 1) {
        run_numa_comparison(config);
    }

    std::cout << "\n============================================================\n";
    std::cout << "  Phase 2 Complete: All OpenMP tests finished\n";
    std::cout << "============================================================\n";

    return 0;
}
