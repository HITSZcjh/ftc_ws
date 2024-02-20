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
    constexpr int Nobs = 17; // 带电机转速反馈
    constexpr double u_range[2] = {0, 6};

    // constexpr double omega_range[2] = {-5, 5};
    // constexpr double velocity_range[2] = {-5, 5};
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
        void get_obs(Eigen::Ref<Eigen::Matrix<double, -1, 1>> obs, Eigen::Matrix<double, Nobs, 1> &noise);
        void get_reward(double &reward);
        void get_done(bool &done);
        void reset();
        void test();
        double x_data[NX];
        double u_data[NU];
        double p_data[NP];
        Eigen::Map<Eigen::Matrix<double, NX, 1>> x;
        Eigen::Map<Eigen::Matrix<double, NU, 1>> u;
        Eigen::Map<Eigen::Matrix<double, NX, 1>> noise;
        Eigen::Map<Eigen::Matrix<double, NK, 1>> k;
        Eigen::Map<Eigen::Matrix<double, Nobs, 1>> obs_map;

    private:
        UAVModel_sim_solver_capsule *capsule;
        sim_config *config;
        sim_in *in;
        sim_out *out;
        void *dims;
        int status;

        std::uniform_real_distribution<double> uniform_dist_{-1.0, 1.0};
        std::random_device rd_;
        std::mt19937 random_gen_{rd_()};

        Eigen::Matrix<double, 3, 2> world_box;
        Eigen::Matrix<double, NX, 1> goal_state;
        double pos_coeff;
        double lin_vel_coeff;
        double ang_vel_coeff;
        int max_ep_len;
        int cnt;
    };
}