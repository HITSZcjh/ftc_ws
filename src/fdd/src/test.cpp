#include "fdd/uav_simulator.hpp"
#include "fdd/particle_filter.hpp"
#include <random>
#include <iostream>
#include <chrono>
using namespace FDD;
int main()
{
    int num_particles = 250;
    int num_threads = 25;
    double ts = 0.01;
    ParticleFilter pf(num_particles, num_threads, ts);
    Eigen::Matrix<double, Nobs, 1> obs;
    obs.setZero();
    pf.SetInitState(obs);
    Eigen::VectorXd action(4);
    action.setOnes();
    Eigen::VectorXd state_est(NX);
    Eigen::VectorXd k_est(NK);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; i++)
    {
        pf.Loop(obs, action, state_est, k_est);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    return 0;
}