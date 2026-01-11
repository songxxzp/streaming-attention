/**
 * @file test_qwen3_mpi_simple.cpp
 * @brief Simple test for Qwen3 MPI functions
 */

#include "tensor_cpp/qwen3_ops_mpi.h"
#include "tensor_cpp/tensor.h"
#include <mpi.h>
#include <iostream>
#include <vector>

using namespace tensor_cpp;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "\n========================================\n";
        std::cout << "Qwen3 MPI Simple Test\n";
        std::cout << "========================================\n";
        std::cout << "MPI processes: " << size << "\n";
        std::cout << "========================================\n\n";
    }

    try {
        // Test 1: qwen3_mlp_mpi_omp
        if (rank == 0) std::cout << "Test 1: qwen3_mlp_mpi_omp\n";

        // hidden_states: [batch=2, seq_len=4, hidden_size=1024]
        std::vector<float> hidden_data(2 * 4 * 1024, 0.1f);
        for (size_t i = 0; i < hidden_data.size(); ++i) {
            hidden_data[i] = static_cast<float>(i % 100) / 100.0f;
        }
        Tensor hidden_states(std::move(hidden_data), Shape({2, 4, 1024}));

        // gate_proj: [intermediate_size=4096, hidden_size=1024]
        std::vector<float> gate_data(4096 * 1024, 0.1f);
        Tensor gate_proj(std::move(gate_data), Shape({4096, 1024}));

        // up_proj: [intermediate_size=4096, hidden_size=1024]
        std::vector<float> up_data(4096 * 1024, 0.1f);
        Tensor up_proj(std::move(up_data), Shape({4096, 1024}));

        // down_proj: [hidden_size=1024, intermediate_size=4096]
        std::vector<float> down_data(1024 * 4096, 0.1f);
        Tensor down_proj(std::move(down_data), Shape({1024, 4096}));

        Tensor output = qwen3::mpi::qwen3_mlp_mpi_omp(
            hidden_states, gate_proj, up_proj, down_proj, MPI_COMM_WORLD
        );

        if (rank == 0) {
            std::cout << "  Input shape: [" << hidden_states.shape()[0] << ", "
                      << hidden_states.shape()[1] << ", " << hidden_states.shape()[2] << "]\n";
            std::cout << "  Output shape: [" << output.shape()[0] << ", "
                      << output.shape()[1] << ", " << output.shape()[2] << "]\n";
            std::cout << "  Status: PASSED ✅\n\n";
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // Test 2: qwen3_attention_mpi_omp
        if (rank == 0) std::cout << "Test 2: qwen3_attention_mpi_omp\n";

        // hidden_states: [batch=1, seq_len=4, hidden_size=1024]
        std::vector<float> attn_hidden_data(1 * 4 * 1024, 0.1f);
        Tensor attn_hidden(std::move(attn_hidden_data), Shape({1, 4, 1024}));

        // qkv_projs: [q_dim + 2*kv_dim, hidden_size] = [2048 + 2*1024, 1024] = [4096, 1024]
        size_t num_heads = 16, kv_heads = 8, head_dim = 64;
        size_t q_dim = num_heads * head_dim;
        size_t kv_dim = kv_heads * head_dim;
        size_t total_qkv = q_dim + 2 * kv_dim;

        std::vector<float> qkv_data(total_qkv * 1024, 0.1f);
        Tensor qkv_projs(std::move(qkv_data), Shape({static_cast<long>(total_qkv), 1024}));

        // o_proj: [hidden_size=1024, hidden_size=1024]
        std::vector<float> o_data(1024 * 1024, 0.1f);
        Tensor o_proj(std::move(o_data), Shape({1024, 1024}));

        // q_norm, k_norm: [head_dim=64]
        std::vector<float> q_norm_data(64, 1.0f);
        Tensor q_norm(std::move(q_norm_data), Shape({64}));

        std::vector<float> k_norm_data(64, 1.0f);
        Tensor k_norm(std::move(k_norm_data), Shape({64}));

        // RoPE cos/sin: [seq_len=4, head_dim/2=32]
        std::vector<float> cos_data(4 * 32, 1.0f);
        Tensor cos(std::move(cos_data), Shape({4, 32}));

        std::vector<float> sin_data(4 * 32, 0.0f);
        Tensor sin(std::move(sin_data), Shape({4, 32}));

        Tensor attn_output = qwen3::mpi::qwen3_attention_mpi_omp(
            attn_hidden, num_heads, kv_heads, head_dim,
            qkv_projs, o_proj, q_norm, k_norm, cos, sin, MPI_COMM_WORLD
        );

        if (rank == 0) {
            std::cout << "  Input shape: [" << attn_hidden.shape()[0] << ", "
                      << attn_hidden.shape()[1] << ", " << attn_hidden.shape()[2] << "]\n";
            std::cout << "  Output shape: [" << attn_output.shape()[0] << ", "
                      << attn_output.shape()[1] << ", " << attn_output.shape()[2] << "]\n";
            std::cout << "  Status: PASSED ✅\n\n";
        }

        if (rank == 0) {
            std::cout << "========================================\n";
            std::cout << "All tests passed!\n";
            std::cout << "========================================\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Rank " << rank << " error: " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    MPI_Finalize();
    return 0;
}
