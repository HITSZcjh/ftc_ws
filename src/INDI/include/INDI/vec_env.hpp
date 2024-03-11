#pragma once

#include "INDI/uav_simulator.hpp"
#include <memory>
#include <yaml-cpp/yaml.h>

namespace QuadrotorEnv
{

    class VecEnv
    {
    public:
        VecEnv();
        void init();
        void step(Eigen::Ref<Eigen::Matrix<double, -1, -1, 1>> actions,
                  Eigen::Ref<Eigen::Matrix<double, -1, -1, 1>> obs,
                  Eigen::Ref<Eigen::Matrix<double, -1, 1>> rewards,
                  Eigen::Ref<Eigen::Matrix<bool, -1, 1>> dones);
        void reset(Eigen::Ref<Eigen::Matrix<double, -1, -1, 1>> obs);
        inline int get_num_envs() { return num_envs; }
        inline int get_obs_dim() { return Nobs; }
        inline int get_action_dim() { return NU; }
        inline int get_state_dim() { return NX; }
        void set_k(Eigen::Ref<Eigen::Matrix<double, -1, -1, 1>> k);
        void set_state(Eigen::Ref<Eigen::Matrix<double, -1, -1, 1>> state,
                       Eigen::Ref<Eigen::Matrix<double, -1, -1, 1>> obs);
        void get_state(Eigen::Ref<Eigen::Matrix<double, -1, -1, 1>> state);

    private:
        YAML::Node cfg;
        int num_envs;
        std::vector<std::unique_ptr<Simulator>> envs_list;
    };
}