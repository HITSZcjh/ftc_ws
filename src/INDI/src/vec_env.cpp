#include "INDI/vec_env.hpp"
#include <omp.h>
#include <iostream>
namespace QuadrotorEnv
{
    VecEnv::VecEnv()
    {
        cfg = YAML::LoadFile("/home/jiao/ftc_ws/src/INDI/configs/vec_env.yaml");
        init();
    }

    void VecEnv::init()
    {
        double ts = cfg["env"]["ts"].as<double>();
        num_envs = cfg["env"]["num_envs"].as<int>();
        int num_threads = cfg["env"]["num_threads"].as<int>();

        omp_set_num_threads(num_threads);
        for (int i = 0; i < num_envs; i++)
        {
            envs_list.push_back(std::make_unique<Simulator>(ts, cfg));
        }
    }

    void VecEnv::step(Eigen::Ref<Eigen::Matrix<double, -1, -1, 1>> actions,
                      Eigen::Ref<Eigen::Matrix<double, -1, -1, 1>> obs,
                      Eigen::Ref<Eigen::Matrix<double, -1, 1>> rewards,
                      Eigen::Ref<Eigen::Matrix<bool, -1, 1>> dones)
    {
        if(actions.rows()!=num_envs||actions.cols()!=NU||
        obs.rows()!=num_envs||obs.cols()!=Nobs||
        rewards.rows()!=num_envs||rewards.cols()!=1||
        dones.rows()!=num_envs||dones.cols()!=1)
        {
            std::cerr<<"Invalid input size!"<<std::endl;
            return;
        }

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num_envs; i++)
        {
            envs_list[i]->step(i, actions, obs, rewards, dones);
        }
    }

    void VecEnv::reset(Eigen::Ref<Eigen::Matrix<double, -1, -1, 1>> obs)
    {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num_envs; i++)
        {
            envs_list[i]->reset();
            envs_list[i]->get_obs(obs.row(i));
        }
    }

    void VecEnv::set_k(Eigen::Ref<Eigen::Matrix<double, -1, -1, 1>> k)
    {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num_envs; i++)
        {
            envs_list[i]->set_k(i, k);
        }
    }

    void VecEnv::set_state(Eigen::Ref<Eigen::Matrix<double, -1, -1, 1>> state, Eigen::Ref<Eigen::Matrix<double, -1, -1, 1>> obs)
    {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num_envs; i++)
        {
            envs_list[i]->set_state(i, state, obs);
        }
    }
}