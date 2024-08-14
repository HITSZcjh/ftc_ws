#include "quadrotors_sim/vec_env.hpp"

namespace quadrotors
{
    VecEnv::VecEnv()
    {
        YAML::Node cfg = YAML::LoadFile("/home/jiao/rl_quad_ws/ftc_ws/src/quadrotors_sim/configs/vec_env.yaml");
        num_envs = cfg["env"]["num_envs"].as<int>();
        omp_set_num_threads(cfg["env"]["num_threads"].as<int>());
        for (int i = 0; i < num_envs; i++)
        {
            envs_list.push_back(std::make_unique<Simulator>(cfg));
        }
        for (auto &re : envs_list[0]->extra_info)
        {
            extra_info_names.push_back(re.first);
        }
    }
    void VecEnv::step(TensorsRef<Scalar> actions,
                      TensorsRef<Scalar> obs,
                      VectorRef<Scalar> rewards,
                      VectorRef<bool> dones,
                      TensorsRef<Scalar> extra_infos)
    {
        if (actions.rows() != num_envs || actions.cols() != NU ||
            obs.rows() != num_envs || obs.cols() != NOBS ||
            rewards.rows() != num_envs || rewards.cols() != 1 ||
            dones.rows() != num_envs || dones.cols() != 1)
        {
            std::cerr << "Invalid input size!" << std::endl;
            return;
        }

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num_envs; i++)
        {
            envs_list[i]->step(actions.row(i), obs.row(i), rewards(i), dones(i), extra_infos.row(i));
        }
    }
    void VecEnv::reset(TensorsRef<Scalar> obs)
    {
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num_envs; i++)
        {
            envs_list[i]->reset(obs.row(i));
        }
    }
    void VecEnv::get_k(TensorsRef<Scalar> k)
    {
        if (k.rows() != num_envs || k.cols() != 4)
        {
            std::cerr << "Invalid input K size!" << std::endl;
            return;
        }
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num_envs; i++)
        {
            envs_list[i]->get_k(k.row(i));
        }
    }

    void VecEnv::set_k(TensorsRef<Scalar> k)
    {
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num_envs; i++)
        {
            envs_list[i]->set_k(k.row(i));
        }
    }

    void VecEnv::set_state(TensorsRef<Scalar> state, TensorsRef<Scalar> obs)
    {
        if (state.rows() != num_envs || state.cols() != NX)
        {
            std::cerr << "Invalid input X size!" << std::endl;
            return;
        }

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num_envs; i++)
        {
            envs_list[i]->set_state(state.row(i), obs.row(i));
        }
    }

    void VecEnv::get_state(TensorsRef<Scalar> state)
    {
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num_envs; i++)
        {
            envs_list[i]->get_state(state.row(i));
        }
    }

    void VecEnv::print_quad_param()
    {
        for (int i = 0; i < num_envs; i++)
        {
            envs_list[i]->print_quad_param();
        }
    }

}
