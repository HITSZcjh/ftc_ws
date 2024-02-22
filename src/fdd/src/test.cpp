#include "fdd/uav_simulator.hpp"
#include "fdd/particle_filter.hpp"
#include <random>
#include <iostream>
#include <chrono>
int main()
{

    UavSimulator::Simulator uav_simulator;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> u_dis(1.0, 2.0);
    std::normal_distribution<double> z_dis(0, 0.01);
    std::normal_distribution<double> x_dis(0, 0.01);

    Eigen::Matrix<double, UavSimulator::Nobs, 1> z_noise;
    for (int i = 0; i < UavSimulator::Nobs; i++)
    {
        z_noise(i) = z_dis(gen);
    }
    uav_simulator.get_obs(z_noise);
    ParticleFilter::Filter PF(uav_simulator.obs);

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int cnt = 0; cnt < 600; cnt++)
    {
        Eigen::Vector4d k = 0.9 * Eigen::Vector4d::Ones();
        if (cnt > 200)
            k(0) = 0.5;
        if (cnt > 400)
            k(1) = 0.5;
        if (cnt > 600)
            k(2) = 0.5;
        if (cnt > 800)
            k(3) = 0.5;
        uav_simulator.noise = Eigen::Matrix<double, UavSimulator::NX, 1>::Ones();
        for (int i = 0; i < UavSimulator::NX; i++)
        {
            uav_simulator.noise(i) = uav_simulator.noise(i) + x_dis(gen);
        }
        Eigen::Matrix<double, UavSimulator::Nobs, 1> z_noise;
        for (int i = 0; i < UavSimulator::Nobs; i++)
        {
            z_noise(i) = z_dis(gen);
        }
        Eigen::Matrix<double, UavSimulator::NU, 1> u;
        for (int i = 0; i < UavSimulator::NU; i++)
        {
            u(i) = u_dis(gen);
        }
        uav_simulator.u = u;
        uav_simulator.k = k;
        uav_simulator.step();
        uav_simulator.get_obs(z_noise);

        PF.Loop(uav_simulator.obs, u);

        uav_simulator.x0 = uav_simulator.x1;
        // std::cout << PF.x_est.transpose() << std::endl;
        std::cout << cnt << " " << PF.k_est.transpose() << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "solve time: " << duration.count() << " milliseconds" << std::endl;

    return 0;

    // UavSimulator::Simulator uav_simulator;
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_real_distribution<double> dis(1.0, 2.0);
    // for(int j=0; j< UavSimulator::NU; j++)
    //     uav_simulator.u(j) = -1;
    // auto start_time = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < 1000; i++)
    // {
    //     // for(int j=0; j< Simulator::NU; j++)
    //     //     uav_simulator.u_map(j) = dis(gen);
    //     uav_simulator.step();
    //     // std::cout << uav_simulator.u_map.transpose() << std::endl;
    //     uav_simulator.x0 = uav_simulator.x1;
    //     // std::cout << uav_simulator.x1_map.transpose() << std::endl;
    // }
    // auto end_time = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    // std::cout << "MPC solve time: " << duration.count() << " microseconds" << std::endl;
    // return 0;
}