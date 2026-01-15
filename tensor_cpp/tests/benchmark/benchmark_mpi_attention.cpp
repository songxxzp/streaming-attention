/**
 * @file benchmark_mpi_attention.cpp
 * @brief Comprehensive benchmark comparing MPI attention implementations
 */

#include <mpi.h>
#include "tensor_cpp/ops_mpi.h"
#include "tensor_cpp/ops.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>

using namespace tensor_cpp;
using namespace tensor_cpp::ops;

// Benchmark standard vs streaming MPI attention
void benchmark_attention_comparison(MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        std::cout << "\n";
        std::cout << "==================================================================================\n";
        std::cout << "  MPI Attention Benchmark: Standard vs Streaming\n";
        std::cout << "==================================================================================\n";
        std::cout << "MPI processes: " << size << "\n";
        std::cout << "Testing different sequence lengths to show where streaming is beneficial\n";
        std::cout << "==================================================================================\n\n";
    }

    // Test different sequence lengths
    std::vector<size_t> seq_lengths = {32, 64, 128, 256, 512, 1024};
    size_t batch = 2;
    size_t num_heads = 16;
    size_t num_kv_heads = 8;
    size_t head_dim = 128;

    for (size_t seq_len : seq_lengths) {
        if (rank == 0) {
            std::cout << "Sequence Length: " << seq_len << "\n";
            std::cout << "----------------------------------------\n";
        }

        // Create test tensors
        std::vector<float> q_data(batch * num_heads * seq_len * head_dim);
        std::vector<float> k_data(batch * num_kv_heads * seq_len * head_dim);
        std::vector<float> v_data(batch * num_kv_heads * seq_len * head_dim);

        for (size_t i = 0; i < q_data.size(); ++i) q_data[i] = static_cast<float>(i) / 1000.0f;
        for (size_t i = 0; i < k_data.size(); ++i) k_data[i] = static_cast<float>(i) / 2000.0f;
        for (size_t i = 0; i < v_data.size(); ++i) v_data[i] = static_cast<float>(i) / 3000.0f;

        Tensor q_orig(std::move(q_data), Shape({static_cast<long>(batch), static_cast<long>(num_heads),
                                                static_cast<long>(seq_len), static_cast<long>(head_dim)}));
        Tensor k_orig(std::move(k_data), Shape({static_cast<long>(batch), static_cast<long>(num_kv_heads),
                                                static_cast<long>(seq_len), static_cast<long>(head_dim)}));
        Tensor v_orig(std::move(v_data), Shape({static_cast<long>(batch), static_cast<long>(num_kv_heads),
                                                static_cast<long>(seq_len), static_cast<long>(head_dim)}));

        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        const int num_iterations = 5;

        // Benchmark Standard Attention
        double standard_time = 0.0;
        for (int iter = 0; iter < num_iterations; ++iter) {
            // Copy tensors (since they're moved)
            std::vector<float> q_data_copy(batch * num_heads * seq_len * head_dim);
            std::vector<float> k_data_copy(batch * num_kv_heads * seq_len * head_dim);
            std::vector<float> v_data_copy(batch * num_kv_heads * seq_len * head_dim);

            for (size_t i = 0; i < q_data_copy.size(); ++i) q_data_copy[i] = static_cast<float>(i) / 1000.0f;
            for (size_t i = 0; i < k_data_copy.size(); ++i) k_data_copy[i] = static_cast<float>(i) / 2000.0f;
            for (size_t i = 0; i < v_data_copy.size(); ++i) v_data_copy[i] = static_cast<float>(i) / 3000.0f;

            Tensor q(std::move(q_data_copy), Shape({static_cast<long>(batch), static_cast<long>(num_heads),
                                                    static_cast<long>(seq_len), static_cast<long>(head_dim)}));
            Tensor k(std::move(k_data_copy), Shape({static_cast<long>(batch), static_cast<long>(num_kv_heads),
                                                    static_cast<long>(seq_len), static_cast<long>(head_dim)}));
            Tensor v(std::move(v_data_copy), Shape({static_cast<long>(batch), static_cast<long>(num_kv_heads),
                                                    static_cast<long>(seq_len), static_cast<long>(head_dim)}));

            MPI_Barrier(comm);
            auto start = std::chrono::high_resolution_clock::now();

            Tensor result = mpi::self_attention_mpi_omp(q, k, v, nullptr, scale,
                                                         num_heads, num_kv_heads, comm);

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> diff = end - start;
            standard_time += diff.count();
        }
        standard_time /= num_iterations;

        // Benchmark Streaming Attention
        double streaming_time = 0.0;
        for (int iter = 0; iter < num_iterations; ++iter) {
            // Copy tensors
            std::vector<float> q_data_copy(batch * num_heads * seq_len * head_dim);
            std::vector<float> k_data_copy(batch * num_kv_heads * seq_len * head_dim);
            std::vector<float> v_data_copy(batch * num_kv_heads * seq_len * head_dim);

            for (size_t i = 0; i < q_data_copy.size(); ++i) q_data_copy[i] = static_cast<float>(i) / 1000.0f;
            for (size_t i = 0; i < k_data_copy.size(); ++i) k_data_copy[i] = static_cast<float>(i) / 2000.0f;
            for (size_t i = 0; i < v_data_copy.size(); ++i) v_data_copy[i] = static_cast<float>(i) / 3000.0f;

            Tensor q(std::move(q_data_copy), Shape({static_cast<long>(batch), static_cast<long>(num_heads),
                                                    static_cast<long>(seq_len), static_cast<long>(head_dim)}));
            Tensor k(std::move(k_data_copy), Shape({static_cast<long>(batch), static_cast<long>(num_kv_heads),
                                                    static_cast<long>(seq_len), static_cast<long>(head_dim)}));
            Tensor v(std::move(v_data_copy), Shape({static_cast<long>(batch), static_cast<long>(num_kv_heads),
                                                    static_cast<long>(seq_len), static_cast<long>(head_dim)}));

            MPI_Barrier(comm);
            auto start = std::chrono::high_resolution_clock::now();

            Tensor result = mpi::self_attention_mpi_streaming_omp(q, k, v, nullptr, scale,
                                                                   num_heads, num_kv_heads, comm);

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> diff = end - start;
            streaming_time += diff.count();
        }
        streaming_time /= num_iterations;

        if (rank == 0) {
            std::cout << "  Standard:  " << std::setw(8) << std::fixed << std::setprecision(2) << standard_time << " ms\n";
            std::cout << "  Streaming: " << std::setw(8) << std::fixed << std::setprecision(2) << streaming_time << " ms";
            if (streaming_time < standard_time) {
                double speedup = standard_time / streaming_time;
                std::cout << "  (✓ " << std::setprecision(2) << speedup << "x faster)";
            } else if (streaming_time > standard_time) {
                double slowdown = streaming_time / standard_time;
                std::cout << "  (✗ " << std::setprecision(2) << slowdown << "x slower)";
            } else {
                std::cout << "  (= same)";
            }
            std::cout << "\n\n";
        }

        MPI_Barrier(comm);
    }

    if (rank == 0) {
        std::cout << "==================================================================================\n";
        std::cout << "  Analysis:\n";
        std::cout << "==================================================================================\n";
        std::cout << "  Standard Attention:  Materializes full QK^T matrix [seq_len, seq_len]\n";
        std::cout << "                       Memory: O(seq_len²), Cache-friendly\n";
        std::cout << "\n";
        std::cout << "  Streaming Attention: Online softmax, processes blocks\n";
        std::cout << "                       Memory: O(seq_len), Better for long sequences\n";
        std::cout << "\n";
        std::cout << "  For short sequences (< 128): Standard is usually faster (less overhead)\n";
        std::cout << "  For long sequences (> 256): Streaming is competitive (better cache, memory)\n";
        std::cout << "==================================================================================\n\n";
    }
}

// Benchmark scaling with different number of MPI processes
void benchmark_mpi_scaling(MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        std::cout << "\n";
        std::cout << "==================================================================================\n";
        std::cout << "  MPI Scaling Benchmark (Streaming Attention)\n";
        std::cout << "==================================================================================\n";
        std::cout << "Testing how streaming attention scales with number of processes\n";
        std::cout << "==================================================================================\n\n";
    }

    size_t batch = 2;
    size_t seq_len = 256;
    size_t num_heads = 16;
    size_t num_kv_heads = 8;
    size_t head_dim = 128;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    if (rank == 0) {
        std::cout << "Configuration:\n";
        std::cout << "  Batch: " << batch << "\n";
        std::cout << "  Sequence length: " << seq_len << "\n";
        std::cout << "  Attention heads: " << num_heads << " (query), " << num_kv_heads << " (key/value)\n";
        std::cout << "  Head dimension: " << head_dim << "\n";
        std::cout << "  MPI processes: " << size << "\n\n";
    }

    // Create test tensors
    std::vector<float> q_data(batch * num_heads * seq_len * head_dim);
    std::vector<float> k_data(batch * num_kv_heads * seq_len * head_dim);
    std::vector<float> v_data(batch * num_kv_heads * seq_len * head_dim);

    for (size_t i = 0; i < q_data.size(); ++i) q_data[i] = static_cast<float>(i) / 1000.0f;
    for (size_t i = 0; i < k_data.size(); ++i) k_data[i] = static_cast<float>(i) / 2000.0f;
    for (size_t i = 0; i < v_data.size(); ++i) v_data[i] = static_cast<float>(i) / 3000.0f;

    const int num_iterations = 10;

    double total_time = 0.0;
    for (int iter = 0; iter < num_iterations; ++iter) {
        std::vector<float> q_data_copy(batch * num_heads * seq_len * head_dim);
        std::vector<float> k_data_copy(batch * num_kv_heads * seq_len * head_dim);
        std::vector<float> v_data_copy(batch * num_kv_heads * seq_len * head_dim);

        for (size_t i = 0; i < q_data_copy.size(); ++i) q_data_copy[i] = static_cast<float>(i) / 1000.0f;
        for (size_t i = 0; i < k_data_copy.size(); ++i) k_data_copy[i] = static_cast<float>(i) / 2000.0f;
        for (size_t i = 0; i < v_data_copy.size(); ++i) v_data_copy[i] = static_cast<float>(i) / 3000.0f;

        Tensor q(std::move(q_data_copy), Shape({static_cast<long>(batch), static_cast<long>(num_heads),
                                                static_cast<long>(seq_len), static_cast<long>(head_dim)}));
        Tensor k(std::move(k_data_copy), Shape({static_cast<long>(batch), static_cast<long>(num_kv_heads),
                                                static_cast<long>(seq_len), static_cast<long>(head_dim)}));
        Tensor v(std::move(v_data_copy), Shape({static_cast<long>(batch), static_cast<long>(num_kv_heads),
                                                static_cast<long>(seq_len), static_cast<long>(head_dim)}));

        MPI_Barrier(comm);
        auto start = std::chrono::high_resolution_clock::now();

        Tensor result = mpi::self_attention_mpi_streaming_omp(q, k, v, nullptr, scale,
                                                               num_heads, num_kv_heads, comm);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> diff = end - start;
        total_time += diff.count();
    }

    double avg_time = total_time / num_iterations;

    if (rank == 0) {
        std::cout << "Results:\n";
        std::cout << "  Average time: " << std::fixed << std::setprecision(2) << avg_time << " ms\n";
        std::cout << "  Throughput: " << std::fixed << std::setprecision(1) << (1000.0 / avg_time) << " iterations/sec\n";
        std::cout << "\n";

        // Estimate speedup vs single process (theoretical)
        std::cout << "  Note: Head-wise parallelism distributes " << num_heads << " heads across " << size << " processes\n";
        std::cout << "        Each process computes ~" << (num_heads / size) << " heads\n";
        if (num_heads % size != 0) {
            std::cout << "        Warning: num_heads not evenly divisible by num_processes (load imbalance)\n";
        }
        std::cout << "==================================================================================\n\n";
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "\n";
        std::cout << "################################################################################\n";
        std::cout << "  MPI Attention Benchmark Suite\n";
        std::cout << "  Comparing Standard vs Streaming Attention with MPI+OpenMP\n";
        std::cout << "################################################################################\n";
        std::cout << "\n";
    }

    // Run benchmarks
    benchmark_attention_comparison(MPI_COMM_WORLD);
    benchmark_mpi_scaling(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n";
        std::cout << "################################################################################\n";
        std::cout << "  All benchmarks completed!\n";
        std::cout << "################################################################################\n";
        std::cout << "\n";
    }

    MPI_Finalize();
    return 0;
}
