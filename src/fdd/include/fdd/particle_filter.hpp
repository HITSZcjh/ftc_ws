#pragma once

#include "fdd/uav_simulator.hpp"
#include <random>
#include <memory>
// #include <cmath>
namespace FDD
{
    constexpr double action_range[2] = {0, 6};
    constexpr double k_range[2] = {0, 1};

    // constexpr double state_noise_std[FDD::NX] = {1.5, 1.5, 1.5,
    //                                              2.5, 2.5, 2.5,
    //                                              0.0, 0.0, 0.0, 0.0,
    //                                              1.5, 1.5, 1.5,
    //                                              0, 0, 0, 0};

    // constexpr double obs_noise_std[FDD::Nobs] = {0.02, 0.02, 0.02,
    //                                              0.1, 0.1, 0.1,
    //                                              0.017, 0.017, 0.017, 0.017,
    //                                              0.1, 0.1, 0.1};

    // constexpr double k_std[FDD::NK] = {0.05, 0.05, 0.05, 0.05};

    constexpr double state_noise_std[FDD::NX] = {2, 2, 2,
                                                 3, 3, 3,
                                                 0.0, 0.0, 0.0, 0.0,
                                                 3, 3, 3,
                                                 0.5, 0.5, 0.5, 0.5};

    constexpr double obs_noise_std[FDD::Nobs] = {0.3, 0.3, 0.3,
                                                 0.3, 0.3, 0.3,
                                                 0.05, 0.05, 0.05, 0.05,
                                                 0.3, 0.3, 0.3};

    constexpr double k_std[FDD::NK] = {0.05, 0.05, 0.05, 0.05};


    class Particle
    {
    public:
        Particle(double *state_data, double *k_data, double *weights, double ts);
        ~Particle()
        {
        }
        void SetInitState(Eigen::Ref<Eigen::Matrix<double, FDD::Nobs, 1>> obs);
        void Update(double *action, Eigen::Ref<Eigen::Matrix<double, FDD::Nobs, 1>> obs);
        Eigen::VectorXd GetSamples(const double *std, int num);
        double GetPDF(Eigen::Ref<Eigen::VectorXd> x,
                      Eigen::Ref<Eigen::VectorXd> mean,
                      Eigen::Ref<Eigen::MatrixXd> cov_inv,
                      double &coef);
        Simulator simulator;
        double *state_data;
        double *k_data;
        double *weights;
        double p_data[FDD::NP];
        Eigen::Map<Eigen::Matrix<double, FDD::NX, 1>> state;
        Eigen::Map<Eigen::Matrix<double, FDD::NK, 1>> k;

    private:
        std::normal_distribution<double> dist;
        std::random_device rd_;
        std::mt19937 random_gen_{rd_()};
        Eigen::MatrixXd obs_noise_cov_inv;
        double obs_noise_coef;
    };

    class ParticleFilter
    {
    public:
        ParticleFilter(int num_particles, int num_threads, double ts);
        ~ParticleFilter()
        {
            delete[] state_matrix_data;
            delete[] k_matrix_data;
        }
        void SimpleResample();
        void GetEstimate();
        void SetInitState(Eigen::Ref<Eigen::Matrix<double, -1, 1>> obs);
        void Loop(Eigen::Ref<Eigen::Matrix<double, -1, 1>> obs,
                  Eigen::Ref<Eigen::Matrix<double, -1, 1>> action,
                  Eigen::Ref<Eigen::Matrix<double, -1, 1>> state_est,
                  Eigen::Ref<Eigen::Matrix<double, -1, 1>> k_est);
        void GetStateMatrix(Eigen::Ref<Eigen::Matrix<double, -1, -1, 1>> state_matrix);
        Eigen::Matrix<double, NX, 1> state_est;
        Eigen::Matrix<double, NK, 1> k_est;

    private:
        int num_particles;
        std::random_device rd_;
        std::mt19937 random_gen_{rd_()};
        std::uniform_real_distribution<double> dist;
        std::vector<std::unique_ptr<Particle>> particles_list;
        double *state_matrix_data;
        double *k_matrix_data;
        Eigen::Map<Eigen::MatrixXd> state_matrix;
        Eigen::Map<Eigen::MatrixXd> k_matrix;
        Eigen::VectorXd weights;
    };
}