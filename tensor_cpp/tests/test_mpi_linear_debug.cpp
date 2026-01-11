/**
 * @file test_mpi_linear_debug.cpp
 * @brief Debug MPI linear layer buffer issues
 */

#include <mpi.h>
#include "tensor_cpp/ops_mpi.h"
#include "tensor_cpp/tensor.h"
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
        std::cout << "MPI Linear Layer Debug Test\n";
        std::cout << "========================================\n";
        std::cout << "MPI processes: " << size << "\n";
        std::cout << "========================================\n\n";
    }

    try {
        // Test 1: Simple case that should work
        if (rank == 0) std::cout << "Test 1: Simple case (128x128)\n";

        std::vector<float> input_data(128 * 64, 1.0f);
        Tensor input(std::move(input_data), Shape({128, 64}));

        std::vector<float> weight_data(128 * 64, 0.1f);
        Tensor weight(std::move(weight_data), Shape({128, 64}));

        Tensor output = ops::mpi::linear_mpi_omp(input, weight, nullptr, MPI_COMM_WORLD);

        if (rank == 0) {
            std::cout << "  Input shape: [" << input.shape()[0] << ", " << input.shape()[1] << "]\n";
            std::cout << "  Weight shape: [" << weight.shape()[0] << ", " << weight.shape()[1] << "]\n";
            std::cout << "  Output shape: [" << output.shape()[0] << ", " << output.shape()[1] << "]\n";
            std::cout << "  Status: PASSED ✅\n\n";
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // Test 2: Qwen3-like dimensions
        if (rank == 0) std::cout << "Test 2: Qwen3 MLP dimensions (4x1024 -> 4x4096)\n";

        // hidden_states reshaped: [seq_len=4, hidden_size=1024]
        std::vector<float> input2_data(4 * 1024, 1.0f);
        Tensor input2(std::move(input2_data), Shape({4, 1024}));

        // gate_proj: [intermediate_size=4096, hidden_size=1024]
        std::vector<float> weight2_data(4096 * 1024, 0.1f);
        for (size_t i = 0; i < weight2_data.size(); ++i) {
            weight2_data[i] = static_cast<float>(i % 100) / 100.0f;
        }
        Tensor weight2(std::move(weight2_data), Shape({4096, 1024}));

        if (rank == 0) {
            std::cout << "  Input shape: [" << input2.shape()[0] << ", " << input2.shape()[1] << "]\n";
            std::cout << "  Weight shape: [" << weight2.shape()[0] << ", " << weight2.shape()[1] << "]\n";
        }

        Tensor output2 = ops::mpi::linear_mpi_omp(input2, weight2, nullptr, MPI_COMM_WORLD);

        if (rank == 0) {
            std::cout << "  Output shape: [" << output2.shape()[0] << ", " << output2.shape()[1] << "]\n";
            std::cout << "  Status: PASSED ✅\n\n";
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // Test 3: Larger sequence
        if (rank == 0) std::cout << "Test 3: Larger sequence (16x1024 -> 16x4096)\n";

        std::vector<float> input3_data(16 * 1024, 1.0f);
        Tensor input3(std::move(input3_data), Shape({16, 1024}));

        Tensor output3 = ops::mpi::linear_mpi_omp(input3, weight2, nullptr, MPI_COMM_WORLD);

        if (rank == 0) {
            std::cout << "  Input shape: [" << input3.shape()[0] << ", " << input3.shape()[1] << "]\n";
            std::cout << "  Output shape: [" << output3.shape()[0] << ", " << output3.shape()[1] << "]\n";
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
