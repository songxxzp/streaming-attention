// Simple test for OMP functionality
#include "attention/attention.h"
#include "utils/timer.h"
#include <iostream>
#include <random>
#include <vector>
#include <omp.h>

void generate_random_data(float* data, int size, float min = -1.0f, float max = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);

    for (int i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
}

int main() {
    std::cout << "Testing OMP implementation\n";
    std::cout << "Max threads: " << omp_get_max_threads() << "\n";

    int T = 512;
    int d = 64;
    int block_size = 32;

    std::vector<float> Q(d);
    std::vector<float> K(T * d);
    std::vector<float> V(T * d);

    generate_random_data(Q.data(), d, -0.1f, 0.1f);
    generate_random_data(K.data(), T * d);
    generate_random_data(V.data(), T * d);

    std::cout << "Data generated\n";

    // Test serial version
    std::cout << "Testing serial version...\n";
    auto ref_output = streaming_attention_serial(Q.data(), K.data(), V.data(), T, d, block_size);
    std::cout << "Serial done, output[0]=" << ref_output[0] << "\n";

    // Test OMP version with 1 thread
    std::cout << "Testing OMP version with 1 thread...\n";
    auto omp1_output = streaming_attention_omp(Q.data(), K.data(), V.data(), T, d, block_size, 1);
    std::cout << "OMP-1 done, output[0]=" << omp1_output[0] << "\n";

    // Test OMP version with 2 threads
    std::cout << "Testing OMP version with 2 threads...\n";
    auto omp2_output = streaming_attention_omp(Q.data(), K.data(), V.data(), T, d, block_size, 2);
    std::cout << "OMP-2 done, output[0]=" << omp2_output[0] << "\n";

    // Compare
    float l2_err1 = compute_l2_error(ref_output.data(), omp1_output.data(), d);
    float l2_err2 = compute_l2_error(ref_output.data(), omp2_output.data(), d);

    std::cout << "L2 error (OMP-1 vs Serial): " << l2_err1 << "\n";
    std::cout << "L2 error (OMP-2 vs Serial): " << l2_err2 << "\n";

    if (l2_err1 < 1e-4 && l2_err2 < 1e-4) {
        std::cout << "All tests PASSED!\n";
        return 0;
    } else {
        std::cout << "Tests FAILED!\n";
        return 1;
    }
}
