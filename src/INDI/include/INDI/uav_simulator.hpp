#pragma once

// standard
#include <stdio.h>
#include <stdlib.h>
// acados
#include "acados/utils/print.h"
#include "acados/utils/math.h"
#include "acados_c/sim_interface.h"
#include "acados_sim_solver_UAVModel.h"

#include <Eigen/Eigen>
#include <random>

#include <yaml-cpp/yaml.h>
namespace QuadrotorEnv
{

    constexpr int NX = UAVMODEL_NX;
    constexpr int NU = UAVMODEL_NU;
    constexpr int NP = UAVMODEL_NP;
    constexpr int NK = 4;
    constexpr int Nobs = 21;
    constexpr double u_range[2] = {0, 6};
    constexpr double delta_u_range[2] = {-50, 50};
    constexpr double mass_inv = 1 / 0.73;
    // constexpr double omega_range[2] = {-5, 5};
    // constexpr double velocity_range[2] = {-5, 5};

    constexpr double state_noise_std[QuadrotorEnv::NX] = {0.05, 0.05, 0.05,
                                                 0.5, 0.5, 0.5,
                                                 0.0, 0.0, 0.0, 0.0,
                                                 0.5, 0.5, 0.5,
                                                 0, 0, 0, 0};
    constexpr double obs_noise_std[QuadrotorEnv::Nobs] = {0.02, 0.02, 0.02,
                                                 0.1, 0.1, 0.1,
                                                 0.017, 0.017, 0.017, 0.017,
                                                 0.1, 0.1, 0.1,
                                                 0,0,0,0,
                                                 0.1,
                                                 0.1,0.1,0.1,
                                                 };

    // constexpr double state_noise_std[QuadrotorEnv::NX] = {0.25, 0.25, 0.25,
    //                                              2.5, 2.5, 2.5,
    //                                              0.0, 0.0, 0.0, 0.0,
    //                                              2.5, 2.5, 2.5,
    //                                              0, 0, 0, 0};
    // constexpr double obs_noise_std[QuadrotorEnv::Nobs] = {0.1, 0.1, 0.1,
    //                                              0.5, 0.5, 0.5,
    //                                              0.085, 0.085, 0.085, 0.085,
    //                                              0.5, 0.5, 0.5,
    //                                              0,0,0,0,
    //                                              0.5,
    //                                              0.5,0.5,0.5,
    //                                              };


    class LPF_t
    {
    public:
        double ts;
        double cutoff_freq; // rad/s
        Eigen::Matrix<double, -1, 1> last_output;
        Eigen::Matrix<double, -1, 1> output;
        LPF_t(double cutoff_freq, double ts, Eigen::Matrix<double, -1, 1> input) : cutoff_freq(cutoff_freq), ts(ts), last_output(input){};
        void calculate_derivative(Eigen::Ref<Eigen::Matrix<double, -1, 1>> input,
                           Eigen::Ref<Eigen::Matrix<double, -1, 1>> output);
    };

    class Simulator
    {
    public:
        Simulator(double ts, YAML::Node cfg);
        ~Simulator();
        void step(int agent_id, Eigen::Ref<Eigen::Matrix<double, -1, -1, 1>> actions,
                  Eigen::Ref<Eigen::Matrix<double, -1, -1, 1>> obs,
                  Eigen::Ref<Eigen::Matrix<double, -1, 1>> rewards,
                  Eigen::Ref<Eigen::Matrix<bool, -1, 1>> dones);
        void get_obs(Eigen::Ref<Eigen::Matrix<double, -1, 1>> obs);
        void get_obs_with_noise(Eigen::Ref<Eigen::Matrix<double, -1, 1>> obs);
        void get_reward(double &reward);
        void get_done(bool &done);
        void get_acc(double &acc);
        void reset();
        void test();
        double x_data[NX];
        double u_data[NU];
        double p_data[NP];
        Eigen::Map<Eigen::Matrix<double, NX, 1>> x;
        Eigen::Map<Eigen::Matrix<double, NU, 1>> u;
        Eigen::Matrix<double, NU, 1> delta_u;
        Eigen::Map<Eigen::Matrix<double, NX, 1>> state_noise;
        Eigen::Matrix<double, Nobs, 1> obs_noise;
        Eigen::Map<Eigen::Matrix<double, NK, 1>> k;
        LPF_t omega_lpf;

    private:
        void get_noise(const double *std, int num, Eigen::Ref<Eigen::VectorXd> noise);
        UAVModel_sim_solver_capsule *capsule;
        sim_config *config;
        sim_in *in;
        sim_out *out;
        void *dims;
        int status;
        std::normal_distribution<double> normal_dist_{0.0, 1.0};
        std::uniform_real_distribution<double> uniform_dist_{-1.0, 1.0};
        std::random_device rd_;
        std::mt19937 random_gen_{rd_()};

        double ts;
        Eigen::Matrix<double, 3, 2> world_box;
        Eigen::Matrix<double, NX, 1> goal_state;
        double pos_coeff;
        double ori_coeff;
        double lin_vel_coeff;
        double ang_vel_coeff;
        double act_coeff;
        int max_ep_len;
        int cnt;
    };
}