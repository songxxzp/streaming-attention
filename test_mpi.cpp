/**
 * Phase 3: MPI + OpenMP Multi-Node Test Suite
 *
 * Tests:
 * 1. Correctness: MPI vs Serial (only rank 0 has reference)
 * 2. Strong scaling: Fixed T_total, varying number of ranks
 * 3. Communication vs Computation breakdown
 */

#include <mpi.h>
#include "attention/attention.h"
#include "utils/timer.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <omp.h>

struct TestConfig {
    int T;          // Total sequence length (across all ranks)
    int d;          // Hidden dimension
    int block_size; // Block size
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

// Distribute data across ranks from rank 0
void distribute_data(
    const float* full_data,
    float* local_data,
    int T_total,
    int d,
    MPI_Comm comm
) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int T_local = T_total / size;

    if (rank == 0) {
        // Scatter data to all ranks
        MPI_Scatter(full_data, T_local * d, MPI_FLOAT,
                    local_data, T_local * d, MPI_FLOAT,
                    0, comm);
    } else {
        // Receive scattered data
        MPI_Scatter(nullptr, T_local * d, MPI_FLOAT,
                    local_data, T_local * d, MPI_FLOAT,
                    0, comm);
    }
}

// Run correctness test
void run_correctness_test(const TestConfig& config, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        std::cout << "\n========== Correctness Test (MPI) ==========\n";
        std::cout << "T=" << config.T << ", d=" << config.d
                  << ", ranks=" << size << "\n";
    }

    // Each rank computes its local size
    int T_local = config.T / size;
    int T_offset = rank * T_local;

    // Allocate data
    std::vector<float> Q(config.d);
    std::vector<float> K_local(T_local * config.d);
    std::vector<float> V_local(T_local * config.d);

    // Generate Q on rank 0 and broadcast
    if (rank == 0) {
        generate_random_data(Q.data(), config.d, -0.1f, 0.1f);
    }
    MPI_Bcast(Q.data(), config.d, MPI_FLOAT, 0, comm);

    // Each rank generates its local K, V
    generate_random_data(K_local.data(), T_local * config.d);
    generate_random_data(V_local.data(), T_local * config.d);

    // Rank 0 also generates full data for reference
    std::vector<float> K_full, V_full;
    if (rank == 0) {
        K_full.resize(config.T * config.d);
        V_full.resize(config.T * config.d);
        generate_random_data(K_full.data(), config.T * config.d);
        generate_random_data(V_full.data(), config.T * config.d);

        // Copy rank 0's local portion
        std::copy(K_local.data(), K_local.data() + T_local * config.d, K_full.data());
        std::copy(V_local.data(), V_local.data() + T_local * config.d, V_full.data());

        // Gather from other ranks
        for (int r = 1; r < size; ++r) {
            MPI_Recv(K_full.data() + r * T_local * config.d, T_local * config.d,
                     MPI_FLOAT, r, 0, comm, MPI_STATUS_IGNORE);
            MPI_Recv(V_full.data() + r * T_local * config.d, T_local * config.d,
                     MPI_FLOAT, r, 1, comm, MPI_STATUS_IGNORE);
        }
    } else {
        // Send local data to rank 0
        MPI_Send(K_local.data(), T_local * config.d, MPI_FLOAT, 0, 0, comm);
        MPI_Send(V_local.data(), T_local * config.d, MPI_FLOAT, 0, 1, comm);
    }

    // Run MPI version
    auto mpi_output = streaming_attention_mpi_simple(
        Q.data(), K_local.data(), V_local.data(),
        T_local, config.T, config.d, config.block_size, comm
    );

    // Rank 0 compares with serial
    if (rank == 0) {
        auto ref_output = streaming_attention_serial(
            Q.data(), K_full.data(), V_full.data(),
            config.T, config.d, config.block_size
        );

        float l2_err = compute_l2_error(ref_output.data(), mpi_output.data(), config.d);
        float max_err = compute_max_error(ref_output.data(), mpi_output.data(), config.d);

        std::cout << "L2 Error:  " << std::scientific << l2_err << "\n";
        std::cout << "Max Error: " << std::scientific << max_err << "\n";

        if (l2_err < 1e-4 && max_err < 1e-5) {
            std::cout << "✓ PASSED: MPI output matches Serial\n";
        } else {
            std::cout << "✗ FAILED: MPI output differs from Serial\n";
        }
    }

    MPI_Barrier(comm);
}

// Run strong scaling experiment
void run_strong_scaling(const TestConfig& config, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        std::cout << "\n========== Strong Scaling (T=" << config.T
                  << ", d=" << config.d << ") ==========\n";
        std::cout << "Ranks: " << size << "\n";
    }

    int T_local = config.T / size;

    // Allocate data
    std::vector<float> Q(config.d);
    std::vector<float> K_local(T_local * config.d);
    std::vector<float> V_local(T_local * config.d);

    // Generate Q on rank 0 and broadcast
    if (rank == 0) {
        generate_random_data(Q.data(), config.d, -0.1f, 0.1f);
    }
    MPI_Bcast(Q.data(), config.d, MPI_FLOAT, 0, comm);

    // Generate local data
    generate_random_data(K_local.data(), T_local * config.d);
    generate_random_data(V_local.data(), T_local * config.d);

    // Timing
    double total_time = 0.0;

    for (int iter = 0; iter < config.warmup + config.repeat; ++iter) {
        MPI_Barrier(comm);

        Timer timer_total;
        timer_total.start();

        // Full MPI attention (includes both computation and communication)
        auto mpi_output = streaming_attention_mpi_simple(
            Q.data(), K_local.data(), V_local.data(),
            T_local, config.T, config.d, config.block_size, comm
        );

        timer_total.stop();

        if (iter >= config.warmup) {
            total_time += timer_total.elapsed();
        }
    }

    total_time /= config.repeat;

    // Report times from rank 0
    if (rank == 0) {
        std::cout << std::string(60, '-') << "\n";
        std::cout << std::left << std::setw(30) << "Total Time"
                  << std::right << std::setw(20) << std::fixed << std::setprecision(3) << total_time * 1000.0 << " ms\n";

        // Compute performance metrics
        double flops = 3.0 * config.T * config.d;
        double gflops = flops / total_time / 1e9;
        double bandwidth = 2.0 * config.T * config.d * sizeof(float) / total_time / 1e9;

        std::cout << std::left << std::setw(30) << "GFLOPS"
                  << std::right << std::setw(20) << std::setprecision(2) << gflops << "\n";
        std::cout << std::left << std::setw(30) << "Bandwidth"
                  << std::right << std::setw(20) << std::setprecision(2) << bandwidth << " GB/s\n";
    }

    MPI_Barrier(comm);
}

// Simple struct for POD (Plain Old Data) MPI operations
struct PartialResultPOD {
    float m;
    float l;
};

void print_system_info(MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        std::cout << "\n========== System Information ==========\n";
        std::cout << "MPI Ranks: " << size << "\n";
        std::cout << "OpenMP Max Threads: " << omp_get_max_threads() << "\n";
        std::cout << "Num_procs: " << omp_get_num_procs() << "\n";
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "============================================================\n";
        std::cout << "  Phase 3: MPI + OpenMP Multi-Node Parallelization\n";
        std::cout << "============================================================\n";
    }

    print_system_info(MPI_COMM_WORLD);

    // Default configuration
    TestConfig config;
    config.T = 8192;
    config.d = 128;
    config.block_size = 64;
    config.warmup = 5;
    config.repeat = 20;

    // Parse command line (only rank 0)
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--T" && i + 1 < argc) config.T = std::atoi(argv[++i]);
            else if (arg == "--d" && i + 1 < argc) config.d = std::atoi(argv[++i]);
            else if (arg == "--block" && i + 1 < argc) config.block_size = std::atoi(argv[++i]);
            else if (arg == "--help") {
                std::cout << "Usage: mpirun -np N " << argv[0] << " [options]\n"
                          << "Options:\n"
                          << "  --T N          Sequence length\n"
                          << "  --d N          Hidden dimension\n"
                          << "  --block N      Block size\n";
                MPI_Abort(MPI_COMM_WORLD, 0);
            }
        }
    }

    // Broadcast config from rank 0
    MPI_Bcast(&config.T, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&config.d, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&config.block_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&config.warmup, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&config.repeat, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Ensure T is divisible by size
    if (config.T % size != 0 && rank == 0) {
        std::cout << "Warning: T (" << config.T << ") not divisible by size (" << size << ")\n";
        std::cout << "Adjusting T to " << (config.T / size * size) << "\n";
        config.T = (config.T / size) * size;
    }
    MPI_Bcast(&config.T, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Run correctness test
    run_correctness_test(config, MPI_COMM_WORLD);

    // Run strong scaling experiments
    std::vector<TestConfig> test_configs = {
        {4096, 128, 64, 5, 20},
        {8192, 128, 64, 5, 20},
        {16384, 256, 128, 5, 20},
        config,
    };

    for (const auto& cfg : test_configs) {
        if (cfg.T >= size) {  // Only run if T is large enough
            run_strong_scaling(cfg, MPI_COMM_WORLD);
        }
    }

    if (rank == 0) {
        std::cout << "\n============================================================\n";
        std::cout << "  Phase 3 Complete: All MPI tests finished\n";
        std::cout << "============================================================\n";
    }

    MPI_Finalize();
    return 0;
}
