#include "fdd/uav_simulator.hpp"
#include <chrono>
#include <iostream>
int main()
{

    UavSimulator::Simulator uav_simulator[10000];
    auto start_time = std::chrono::high_resolution_clock::now();
    for(int cnt = 0; cnt < 300; cnt++){
        #pragma omp parallel for schedule(dynamic)
        for(int i = 0; i < 10000; i++)
            uav_simulator[i].test();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "solve time: " << duration.count() << " microseconds" << std::endl;

    return 0;
}