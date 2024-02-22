#pragma once

#include "fdd/uav_simulator.hpp"
#include <random>
#include <memory>
// #include <cmath>
namespace ParticleFilter
{
    constexpr int num_particles = 1000;
    constexpr double predict_k_sigma = 0.05;
    constexpr double predict_x_sigma = 0.05;
    constexpr double predict_z_sigma = 0.05;
    constexpr double calculate_temp_constance = 1.0 / std::sqrt(std::pow(2 * M_PI * std::pow(predict_z_sigma, 2), UavSimulator::Nobs));
    //5.31154e+11;
    // 1.0 / std::sqrt(std::pow(2 * M_PI * std::pow(predict_z_sigma, 2), UavSimulator::Nobs));
    class Particle
    {
    public:
        Particle();
        ~Particle()
        {}
        void SetInitState(Eigen::Matrix<double, UavSimulator::Nobs, 1> &obs);
        void Update(Eigen::Matrix<double, UavSimulator::Nobs, 1> &obs, Eigen::Matrix<double, UavSimulator::NU,1> &u);
        UavSimulator::Simulator uav_simulator;
        double weights;
        Eigen::Matrix<double, UavSimulator::NX, 1> x;
        Eigen::Matrix<double, UavSimulator::NK, 1> k;
        
        // // 重载等号运算符
        // Particle& operator=(const Particle& other) {
        //     if (this != &other) {
        //         // 使用 Eigen 的 = 运算符
        //         x = other.x;
        //         k = other.k;
        //         weights = other.weights;
        //     }
        //     return *this;
        // }

    private:
        std::random_device rd;
        std::mt19937 gen;
        std::normal_distribution<double> k_dis;
        std::normal_distribution<double> x_dis;

    };

    class Filter
    {
    public:
        Filter(Eigen::Matrix<double, UavSimulator::Nobs, 1> obs);
        ~Filter()
        {}
        void SimpleResample();
        void GetEstimate();
        void Loop(Eigen::Matrix<double, UavSimulator::Nobs, 1> &obs, Eigen::Matrix<double, UavSimulator::NU,1> &u);
        Eigen::Matrix<double, UavSimulator::NX, 1> x_est;
        Eigen::Matrix<double, UavSimulator::NK, 1> k_est;
    private:
        std::random_device rd;
        std::mt19937 gen;
        std::uniform_real_distribution<double> dis;
        std::vector<std::shared_ptr<Particle>> particles_ptr;
        Eigen::MatrixXd x_matrix;
        Eigen::MatrixXd k_matrix;
    };
}