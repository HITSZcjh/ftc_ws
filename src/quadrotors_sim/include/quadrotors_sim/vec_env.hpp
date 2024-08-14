#pragma once
#include <quadrotors_sim/simulator.hpp>
#include <omp.h>
#include <memory>



namespace quadrotors
{
    class VecEnv
    {
    public:
        VecEnv();
        void step(TensorsRef<Scalar> actions,
                  TensorsRef<Scalar> obs,
                  VectorRef<Scalar> rewards,
                  VectorRef<bool> dones,
                  TensorsRef<Scalar> extra_infos);
        void reset(TensorsRef<Scalar> obs);
        void get_k(TensorsRef<Scalar> k);
        void set_k(TensorsRef<Scalar> k);
        void set_state(TensorsRef<Scalar> state, TensorsRef<Scalar> obs);
        void get_state(TensorsRef<Scalar> state);

        inline int get_num_envs() { return num_envs; }
        inline int get_obs_dim() { return NOBS; }
        inline int get_action_dim() { return NU; }
        inline int get_state_dim() { return NX; }
        inline std::vector<std::string> &getExtraInfoNames() { return extra_info_names; }
        void print_quad_param();

    private:
        int num_envs;
        std::vector<std::unique_ptr<Simulator>> envs_list;
        std::vector<std::string> extra_info_names;
    };
}