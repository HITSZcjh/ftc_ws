#include "INDI/uav_simulator.hpp"
#include <iostream>
namespace QuadrotorEnv
{
    Simulator::Simulator(double ts, YAML::Node cfg) : x(x_data, NX), u(u_data, NU), noise(p_data, NX), k(p_data + NX, NK), omega_lpf(50, ts, Eigen::Matrix<double,3,1>::Zero())
    {
        Simulator::ts = ts;
        world_box << -5, 5, -5, 5, 0, 6;
        goal_state << 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

        pos_coeff = cfg["rl"]["pos_coeff"].as<double>();
        ori_coeff = cfg["rl"]["ori_coeff"].as<double>();
        lin_vel_coeff = cfg["rl"]["lin_vel_coeff"].as<double>();
        ang_vel_coeff = cfg["rl"]["ang_vel_coeff"].as<double>();
        act_coeff = cfg["rl"]["act_coeff"].as<double>();
        max_ep_len = cfg["rl"]["max_ep_len"].as<int>();

        capsule = UAVModel_acados_sim_solver_create_capsule();
        status = UAVModel_acados_sim_create(capsule);
        if (status)
        {
            printf("UAVModel_acados_sim_create() returned status %d. Exiting.\n", status);
            exit(1);
        }
        config = UAVModel_acados_get_sim_config(capsule);
        in = UAVModel_acados_get_sim_in(capsule);
        out = UAVModel_acados_get_sim_out(capsule);
        dims = UAVModel_acados_get_sim_dims(capsule);

        // initial
        cnt = 0;
        x.setZero();
        x(2) = 2;
        x(6) = 1;
        u.setZero();
        noise.setZero();
        k.setOnes();

        sim_in_set(config, dims, in, "x", x_data);
        sim_in_set(config, dims, in, "T", &ts);
        UAVModel_acados_sim_update_params(capsule, p_data, NP);
    }

    void Simulator::step(int agent_id, Eigen::Ref<Eigen::Matrix<double, -1, -1, 1>> actions,
                         Eigen::Ref<Eigen::Matrix<double, -1, -1, 1>> obs,
                         Eigen::Ref<Eigen::Matrix<double, -1, 1>> rewards,
                         Eigen::Ref<Eigen::Matrix<bool, -1, 1>> dones)
    {
        delta_u = actions.row(agent_id);
        // set boundary
        delta_u = delta_u.cwiseMax(delta_u_range[0]).cwiseMin(delta_u_range[1]);
        u += delta_u * ts;
        u = u.cwiseMax(u_range[0]).cwiseMin(u_range[1]);
        UAVModel_acados_sim_update_params(capsule, p_data, NP);
        sim_in_set(config, dims, in, "x", x_data);
        sim_in_set(config, dims, in, "u", u_data);
        status = UAVModel_acados_sim_solve(capsule);
        if (status != ACADOS_SUCCESS)
        {
            printf("UAVModel_acados_sim_solve() returned status %d. Exiting.\n", status);
            exit(1);
        }
        sim_out_get(config, dims, out, "x", x_data);
        // x1.segment(3,3) = x1.segment(3,3).cwiseMax(velocity_range[0]).cwiseMin(velocity_range[1]);
        // x1.segment(10,3) = x1.segment(10,3).cwiseMax(omega_range[0]).cwiseMin(omega_range[1]);
        x.segment(6, 4) = x.segment(6, 4) / x.segment(6, 4).norm();

        get_reward(rewards(agent_id));
        get_done(dones(agent_id));
        cnt++;

        if (dones(agent_id))
            reset();
        get_obs(obs.row(agent_id));
    }

    void Simulator::reset()
    {
        cnt = 0;

        x.setZero();

        x(0) = uniform_dist_(random_gen_);
        x(1) = uniform_dist_(random_gen_);
        x(2) = uniform_dist_(random_gen_) + 3;

        x(3) = uniform_dist_(random_gen_);
        x(4) = uniform_dist_(random_gen_);
        x(5) = uniform_dist_(random_gen_);

        x(6) = uniform_dist_(random_gen_);
        x(7) = uniform_dist_(random_gen_);
        x(8) = uniform_dist_(random_gen_);
        x(9) = uniform_dist_(random_gen_);
        x.segment(6, 4) = x.segment(6, 4) / x.segment(6, 4).norm();

        x(10) = uniform_dist_(random_gen_);
        x(11) = uniform_dist_(random_gen_);
        x(12) = uniform_dist_(random_gen_);

        x(13) = (uniform_dist_(random_gen_) + 1) * u_range[1] / 2;
        x(14) = (uniform_dist_(random_gen_) + 1) * u_range[1] / 2;
        x(15) = (uniform_dist_(random_gen_) + 1) * u_range[1] / 2;
        x(16) = (uniform_dist_(random_gen_) + 1) * u_range[1] / 2;

        u.setZero();
        omega_lpf.last_output = x.segment(10, 3);

        k.setOnes();

        double random = (uniform_dist_(random_gen_) + 1) / 2;
        double random_k;
        if(random < 0.3)
            random_k = 0;
        else if(random < 0.8)
            random_k = (uniform_dist_(random_gen_) + 1) / 4;
        else
            random_k = 1-(uniform_dist_(random_gen_) + 1) / 4;

        random = (uniform_dist_(random_gen_) + 1) / 2;
        if(random < 0.2)
            k(0) = random_k;
        else if(random < 0.4)
            k(1) = random_k;
        else if(random < 0.6)
            k(2) = random_k;
        else if(random < 0.8)
            k(3) = random_k;

    }

    void Simulator::get_obs(Eigen::Ref<Eigen::Matrix<double, -1, 1>> obs)
    {
        obs.segment(0, 13) = x.segment(0, 13);
        obs.segment(13, 4) = u;
        get_acc(obs(17));
        omega_lpf.calculate_derivative(x.segment(10, 3), obs.segment(18, 3));
    }

    void Simulator::get_obs(Eigen::Ref<Eigen::Matrix<double, -1, 1>> obs, Eigen::Matrix<double, Nobs, 1> &noise)
    {
        // obs.segment(0, NX) = obs_map + noise;
        obs.segment(6, 4) = obs.segment(6, 4) / obs.segment(6, 4).norm();
    }

    void Simulator::get_reward(double &reward)
    {
        double pos_reward = pos_coeff * (x.segment(0, 3) - goal_state.segment(0, 3)).squaredNorm();
        double lin_vel_reward = lin_vel_coeff * (x.segment(3, 3)).squaredNorm();
        double ori_reward = ori_coeff * (x.segment(7, 2)).squaredNorm();
        double ang_vel_reward = ang_vel_coeff * (x.segment(10, 2)).squaredNorm();
        double act_reward = act_coeff * delta_u.squaredNorm();
        reward = pos_reward + lin_vel_reward + ori_reward + ang_vel_reward + act_reward + 10.0;
        // if(cnt == max_ep_len)
        //     reward += 10;
    }

    void Simulator::get_done(bool &done)
    {
        if (x(0) < world_box(0, 0) || x(0) > world_box(0, 1) || x(1) < world_box(1, 0) || x(1) > world_box(1, 1) || x(2) < world_box(2, 0) || x(2) > world_box(2, 1) || cnt == max_ep_len)
        {
            done = true;
        }
        else
            done = false;
    }

    void Simulator::get_acc(double &acc)
    {
        acc = x.segment(13, 4).sum() * mass_inv;
    }

    void Simulator::test()
    {
        u.setOnes();
        sim_in_set(config, dims, in, "x", x_data);
        sim_in_set(config, dims, in, "u", u_data);
        status = UAVModel_acados_sim_solve(capsule);
        if (status != ACADOS_SUCCESS)
        {
            printf("UAVModel_acados_sim_solve() returned status %d. Exiting.\n", status);
            exit(1);
        }
        sim_out_get(config, dims, out, "x", x_data);
        std::cout << x.transpose() << std::endl;
    }

    Simulator::~Simulator()
    {
        status = UAVModel_acados_sim_free(capsule);
        if (status)
        {
            printf("UAVModel_acados_sim_free() returned status %d. \n", status);
        }

        UAVModel_acados_sim_solver_free_capsule(capsule);
    }

    void LPF_t::calculate_derivative(Eigen::Ref<Eigen::Matrix<double, -1, 1>> input,
                                     Eigen::Ref<Eigen::Matrix<double, -1, 1>> derivative)
    {
        output = (cutoff_freq * ts * input + last_output) / (1 + cutoff_freq * ts);
        derivative = (output - last_output) / ts;
        last_output = output;
    }
}
