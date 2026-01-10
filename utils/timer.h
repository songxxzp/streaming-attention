#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <iostream>
#include <iomanip>

class Timer {
public:
    Timer() : elapsed_time(0.0), running(false) {}

    void start() {
        if (!running) {
            start_time = std::chrono::high_resolution_clock::now();
            running = true;
        }
    }

    void stop() {
        if (running) {
            auto end_time = std::chrono::high_resolution_clock::now();
            elapsed_time += std::chrono::duration<double>(end_time - start_time).count();
            running = false;
        }
    }

    void reset() {
        elapsed_time = 0.0;
        running = false;
    }

    double elapsed() const {
        return elapsed_time;
    }

    static void print_header(const std::string& title = "Performance Results") {
        std::cout << "\n========== " << title << " ==========\n";
        std::cout << std::left << std::setw(30) << "Implementation"
                  << std::right << std::setw(15) << "Time (ms)"
                  << std::setw(15) << "GB/s"
                  << std::setw(15) << "GFLOPS"
                  << "\n";
        std::cout << std::string(75, '-') << "\n";
    }

    static void print_row(const std::string& name, double time_ms, double bandwidth, double gflops) {
        std::cout << std::left << std::setw(30) << name
                  << std::right << std::setw(15) << std::fixed << std::setprecision(3) << time_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << bandwidth
                  << std::setw(15) << std::fixed << std::setprecision(2) << gflops
                  << "\n";
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;
    double elapsed_time;
    bool running;
};

#endif // TIMER_H
