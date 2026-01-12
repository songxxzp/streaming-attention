/**
 * @file test_weights_broadcast.cpp
 * @brief Test if weights need to be broadcast to all MPI ranks
 */

#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops_mpi.h"
#include <mpi.h>
#include <iostream>

using namespace tensor_cpp;
using namespace tensor_cpp::qwen3;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "\n========================================\n";
        std::cout << "Weights Broadcast Test\n";
        std::cout << "========================================\n";
        std::cout << "MPI processes: " << size << "\n";
        std::cout << "========================================\n\n";
    }

    try {
        std::string model_path = "/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors";

        // Load weights only on rank 0
        Qwen3Weights weights;
        if (rank == 0) {
            weights = load_qwen3_weights(model_path);
            std::cout << "Rank 0: Weights loaded\n";
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // Try to access token embedding on all ranks
        if (rank == 0) std::cout << "\nTest 1: Access token_embedding\n";

        size_t vocab_size = weights.embed_tokens.shape()[0];
        size_t hidden_size = weights.embed_tokens.shape()[1];

        if (rank == 0) {
            std::cout << "  Token embedding shape: [" << vocab_size << ", " << hidden_size << "]\n";
            std::cout << "  Rank " << rank << ": PASSED ✅\n";
        } else {
            // This will likely segfault if weights are not available
            std::cout << "  Rank " << rank << ": Accessing weights...\n";
            std::cout << "  Shape: [" << vocab_size << ", " << hidden_size << "]\n";
            std::cout << "  Rank " << rank << ": PASSED ✅\n";
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // Test 2: Try a simple forward pass
        if (rank == 0) std::cout << "\nTest 2: Simple forward pass\n";

        std::vector<long> input_ids_data;
        for (int i = 0; i < 4; ++i) {
            input_ids_data.push_back(static_cast<long>(i % 1000));
        }
        TensorL input_ids(input_ids_data, Shape({1, 4}));

        Tensor output = mpi::qwen3_forward_mpi_omp(
            input_ids,
            weights.embed_tokens,
            weights.layers,
            weights.norm_weight,
            1,  // num_layers (use just 1 layer)
            weights.num_attention_heads,
            weights.num_key_value_heads,
            weights.head_dim,
            1e-6f,
            MPI_COMM_WORLD
        );

        if (rank == 0) {
            std::cout << "  Output shape: [" << output.shape()[0] << ", "
                      << output.shape()[1] << ", " << output.shape()[2] << "]\n";
            std::cout << "  Status: PASSED ✅\n";
        }

        if (rank == 0) {
            std::cout << "\n========================================\n";
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
