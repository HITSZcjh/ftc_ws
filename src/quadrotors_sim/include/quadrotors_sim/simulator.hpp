#pragma once
#include "quadrotors_sim/integrator.hpp"
#include "yaml-cpp/yaml.h"
#include <unordered_map>


namespace quadrotors
{
    // CONST
    constexpr Scalar obs_noise_std[NOBS] = {
        0.02,
        0.02,
        0.02,
        0.1,
        0.1,
        0.1,
        0.017,
        0.017,
        0.017,
        0.017,
        0.1,
        0.1,
        0.1,
        0,
        0,
        0,
        0,
        0.1,
        0.1,
        0.1,
        0.0,
        0.0,
        0.0,
    };

    constexpr Scalar world_box[3][2] = {{-5, 5}, {-5, 5}, {0, 6}};
    // const Vector<NX> goal_state = (Vector<NX>() << 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).finished();
    const Vector<NP> goal_pos = (Vector<NP>() << 0, 0, 3).finished();

    const Vector<NOBS> obs_normalized_max = (Vector<NOBS>() << 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 5, 5, 30, thrust_range[1], thrust_range[1], thrust_range[1], thrust_range[1], 5, 5, 30, 30, 30, 30).finished();

    class Simulator;
    class HyperParam
    {
    private:
        Scalar dt, pos_coeff, ori_coeff, lin_vel_coeff, ang_vel_coeff, act_coeff;
        int max_ep_len;
        friend class Simulator;

    public:
        HyperParam(YAML::Node &cfg);
    };

    class Simulator
    {
    public:
        std::unordered_map<std::string, Scalar> extra_info;
        Simulator(YAML::Node &cfg);
        void set_k(VectorRef<Scalar> k);
        void get_k(VectorRef<Scalar> k);
        void step(Vector<NU> action);
        void step(VectorRef<Scalar> actions,
                  VectorRef<Scalar> obs,
                  Scalar &rewards,
                  bool &dones,
                  VectorRef<Scalar> extra_infos);
        Scalar calc_reward();
        inline bool check_done();
        void reset(VectorRef<Scalar> obs);
        void get_obs(VectorRef<Scalar> obs);
        void random_state();
        inline Vector<3> get_acc();
        void set_state(VectorRef<Scalar> state, VectorRef<Scalar> obs);
        void get_state(VectorRef<Scalar> state);
        void print_quad_param();

    private:
        int itr;
        Scalar reward, total_reward;

        Vector<NX> x;
        Vector<NU> u;
        VectorRef<Scalar, NP> p;
        VectorRef<Scalar, NV> v;
        VectorRef<Scalar, NQ> q;
        VectorRef<Scalar, NW> w;
        VectorRef<Scalar, NTHRUSTS> thrusts_real;
        VectorRef<Scalar, NU> u_lpf;
        Vector<NU> last_u_lpf;

        Vector<NX> ode_func_noise;
        Vector<NOBS> obs_noise;
        QuadParam quad_param;
        HyperParam hyper_param;
        Integrator integrator;

        std::uniform_real_distribution<Scalar> uniform_dist_{-1.0, 1.0};
        std::normal_distribution<Scalar> normal_dist_{0.0, 1.0};
        std::random_device rd_;
        std::mt19937 random_gen_{rd_()};

        void get_noise_vector(const Scalar *std, VectorRef<Scalar> noise);

        LPF_t omega_dot_lpf;
        bool first_reset;
        Scalar max_ep_len_copy;
    };

}