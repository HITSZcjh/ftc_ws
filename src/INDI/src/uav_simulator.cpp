#include "INDI/uav_simulator.hpp"
#include <iostream>
namespace QuadrotorEnv
{
    void Simulator::get_noise(const double *std, int num, Eigen::Ref<Eigen::VectorXd> noise)
    {
        for (int i = 0; i < num; i++)
        {
            noise(i) = normal_dist_(random_gen_) * std[i];
        }
    }

    Simulator::Simulator(double ts, YAML::Node cfg) : x(x_data, NX), u(u_data, NU), state_noise(p_data, NX), k(p_data + NX, NK), omega_lpf(4, ts, Eigen::Matrix<double, 3, 1>::Zero())
    {
        Simulator::ts = ts;
        world_box << -5, 5, -5, 5, 0, 6;
        goal_state << 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

        obs_normalized_max << 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 5, 5, 30, u_range[1], u_range[1], u_range[1], u_range[1], 35, 30, 30, 30;

        pos_coeff = cfg["rl"]["pos_coeff"].as<double>();
        ori_coeff = cfg["rl"]["ori_coeff"].as<double>();
        lin_vel_coeff = cfg["rl"]["lin_vel_coeff"].as<double>();
        ang_vel_coeff = cfg["rl"]["ang_vel_coeff"].as<double>();
        act_coeff = cfg["rl"]["act_coeff"].as<double>();
        max_ep_len = cfg["rl"]["max_ep_len"].as<int>();
        add_noise = cfg["env"]["add_noise"].as<int>();
        delay_time = cfg["env"]["delay_time"].as<double>();
        failed_repeat = cfg["env"]["failed_repeat"].as<int>();
        delay_step = (int)(delay_time / ts) + 1;
        delay_obs_list.resize(delay_step);

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
        x(2) = 3;
        x(6) = 1;
        u.setZero();
        state_noise.setZero();
        obs_noise.setZero();
        k.setOnes();
        if (failed_repeat)
            failed_k_index = -1;

        sim_in_set(config, dims, in, "x", x_data);
        sim_in_set(config, dims, in, "T", &ts);
        UAVModel_acados_sim_update_params(capsule, p_data, NP);
    }

    void Simulator::step(int agent_id, Eigen::Ref<Eigen::Matrix<double, -1, -1, 1>> actions,
                         Eigen::Ref<Eigen::Matrix<double, -1, -1, 1>> obs,
                         Eigen::Ref<Eigen::Matrix<double, -1, 1>> rewards,
                         Eigen::Ref<Eigen::Matrix<bool, -1, 1>> dones)
    {
        u = actions.row(agent_id).cwiseMax(u_range[0]).cwiseMin(u_range[1]);
        sim_in_set(config, dims, in, "u", u_data);
        if (add_noise)
            get_noise(state_noise_std, NX, state_noise);
        UAVModel_acados_sim_update_params(capsule, p_data, NP);

        sim_in_set(config, dims, in, "x", x_data);
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
        get_done(dones(agent_id), rewards(agent_id));
        cnt++;

        if (dones(agent_id))
            reset(obs.row(agent_id));
        else
        {
            if (add_noise)
                get_obs_with_noise(obs.row(agent_id));
            else
                get_obs(obs.row(agent_id));
        }
    }

    void Simulator::reset(Eigen::Ref<Eigen::Matrix<double, -1, 1>> obs)
    {
        cnt = 0;

        x.setZero();

        x(0) = uniform_dist_(random_gen_);
        x(1) = uniform_dist_(random_gen_);
        x(2) = uniform_dist_(random_gen_) + 3;

        x(3) = uniform_dist_(random_gen_);
        x(4) = uniform_dist_(random_gen_);
        x(5) = uniform_dist_(random_gen_);

        double roll = uniform_dist_(random_gen_) * 0.15 * M_PI;
        double pitch = uniform_dist_(random_gen_) * 0.15 * M_PI;
        double yaw = uniform_dist_(random_gen_) * M_PI;
        Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());
        // 将三个 AngleAxis 对象相乘，得到旋转四元数
        Eigen::Quaterniond q = yawAngle * pitchAngle * rollAngle;
        x(6) = q.w();
        x(7) = q.x();
        x(8) = q.y();
        x(9) = q.z();

        // x(6) = uniform_dist_(random_gen_);
        // x(7) = uniform_dist_(random_gen_);
        // x(8) = uniform_dist_(random_gen_);
        // x(9) = uniform_dist_(random_gen_);
        // x.segment(6, 4) = x.segment(6, 4) / x.segment(6, 4).norm();

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

        if (failed_repeat && random < 0.3 && failed_k_list.size() > 0)
        {
            random = (uniform_dist_(random_gen_) + 1) / 2;
            failed_k_index = (int)(random * max_failed_k_list_size) % failed_k_list.size();
            k = failed_k_list[failed_k_index];
        }
        else
        {
            failed_k_index = -1;
            random = (uniform_dist_(random_gen_) + 1) / 2;
            double random_k;
            if (random < 0.3)
                random_k = 0;
            else if (random < 0.8)
                random_k = (uniform_dist_(random_gen_) + 1) / 4;
            else
                random_k = 1 - (uniform_dist_(random_gen_) + 1) / 4;

            random = (uniform_dist_(random_gen_) + 1) / 2;
            if (random < 0.2)
                k(0) = random_k;
            else if (random < 0.4)
                k(1) = random_k;
            else if (random < 0.6)
                k(2) = random_k;
            else if (random < 0.8)
                k(3) = random_k;
        }

        if (add_noise)
        {
            obs.segment(0, 13) = x.segment(0, 13);
            obs.segment(13, 4) = u;
            get_acc(obs(17));
            omega_lpf.calc_derivative(x.segment(10, 3), obs.segment(18, 3));
            get_noise(obs_noise_std, singal_obs_num, obs_noise);
            obs.segment(0, singal_obs_num) += obs_noise;
            obs.segment(6, 4) = obs.segment(6, 4) / obs.segment(6, 4).norm();
        }
        else
        {
            obs.segment(0, 13) = x.segment(0, 13);
            obs.segment(13, 4) = u;
            get_acc(obs(17));
            omega_lpf.calc_derivative(x.segment(10, 3), obs.segment(18, 3));
        }
        for (int i = 1; i < obs_list_num; i++)
        {
            obs.segment(singal_obs_num * i, singal_obs_num) = obs.segment(0, singal_obs_num);
        }

        for (int i = 0; i < obs_list_num; i++)
        {
            obs.segment(singal_obs_num * i, singal_obs_num) =
                obs.segment(singal_obs_num * i, singal_obs_num).cwiseQuotient(obs_normalized_max);
        }

        // for (int i = 0; i < delay_step; i++)
        // {
        //     delay_obs_list[i] = obs;
        // }

        // obs.segment(singal_obs_num * (obs_list_num - 1), singal_obs_num) =
        //     obs.segment(singal_obs_num * (obs_list_num - 1), singal_obs_num).cwiseQuotient(obs_normalized_max);
    }

    void Simulator::get_obs(Eigen::Ref<Eigen::Matrix<double, -1, 1>> obs)
    {
        for (int i = 0; i < obs_list_num - 1; i++)
        {
            obs.segment(singal_obs_num * i, singal_obs_num) = obs.segment(singal_obs_num * (i + 1), singal_obs_num);
        }

        int index_bias = (obs_list_num - 1) * singal_obs_num;
        obs.segment(index_bias + 0, 13) = x.segment(0, 13);
        obs.segment(index_bias + 13, 4) = u;
        get_acc(obs(index_bias + 17));
        omega_lpf.calc_derivative(x.segment(10, 3), obs.segment(index_bias + 18, 3));

        // delay_obs_list.pop_front();
        // delay_obs_list.push_back(obs);
        // obs = delay_obs_list[0];
        // obs.segment(index_bias + 13, 4) = u;

        obs.segment(index_bias, singal_obs_num) = obs.segment(index_bias, singal_obs_num).cwiseQuotient(obs_normalized_max);
    }

    void Simulator::get_obs_with_noise(Eigen::Ref<Eigen::Matrix<double, -1, 1>> obs)
    {
        for (int i = 0; i < obs_list_num - 1; i++)
        {
            obs.segment(singal_obs_num * i, singal_obs_num) = obs.segment(singal_obs_num * (i + 1), singal_obs_num);
        }

        int index_bias = (obs_list_num - 1) * singal_obs_num;
        obs.segment(index_bias + 0, 13) = x.segment(0, 13);
        obs.segment(index_bias + 13, 4) = u;
        get_acc(obs(index_bias + 17));
        omega_lpf.calc_derivative(x.segment(10, 3), obs.segment(index_bias + 18, 3));
        get_noise(obs_noise_std, singal_obs_num, obs_noise);
        obs.segment(index_bias + 0, singal_obs_num) += obs_noise;
        obs.segment(index_bias + 6, 4) = obs.segment(index_bias + 6, 4) / obs.segment(index_bias + 6, 4).norm();

        // delay_obs_list.pop_front();
        // delay_obs_list.push_back(obs);
        // obs = delay_obs_list[0];

        // obs.segment(index_bias + 13, 4) = u;

        obs.segment(index_bias, singal_obs_num) = obs.segment(index_bias, singal_obs_num).cwiseQuotient(obs_normalized_max);
    }

    void Simulator::get_reward(double &reward)
    {
        // double pos_reward = pos_coeff * exp(-(x.segment(0, 3) - goal_state.segment(0, 3)).squaredNorm());
        // double lin_vel_reward = lin_vel_coeff * exp(-(x.segment(3, 3)).squaredNorm());
        // double ori_reward = ori_coeff * exp(-(x.segment(7, 2)).squaredNorm());
        // double ang_vel_reward = ang_vel_coeff * exp(-(x.segment(10, 2)).squaredNorm());
        // double act_reward = act_coeff * exp(-u.squaredNorm());
        // reward = pos_reward + lin_vel_reward + ori_reward + ang_vel_reward + act_reward;

        double pos_reward = pos_coeff * (x.segment(0, 3) - goal_state.segment(0, 3)).squaredNorm();
        double lin_vel_reward = lin_vel_coeff * (x.segment(3, 3)).squaredNorm();
        double ori_reward = ori_coeff * (x.segment(7, 2)).squaredNorm();
        double ang_vel_reward = ang_vel_coeff * (x.segment(10, 2)).squaredNorm();
        double act_reward = act_coeff * u.squaredNorm();
        reward = pos_reward + lin_vel_reward + ori_reward + ang_vel_reward + act_reward + 10.0;

        // if(cnt == max_ep_len)
        //     reward += 10;
    }

    void Simulator::get_done(bool &done, double &reward)
    {
        if (cnt == max_ep_len)
        {
            if(failed_repeat && failed_k_index != -1)
            {
                auto it = failed_k_list.begin()+failed_k_index;
                failed_k_list.erase(it);
                failed_k_index = -1;
            }
            done = true;
            reward += 0.0;
        }
        else if (x(0) < world_box(0, 0) || x(0) > world_box(0, 1) || x(1) < world_box(1, 0) || x(1) > world_box(1, 1) || x(2) < world_box(2, 0) || x(2) > world_box(2, 1))
        {
            if(failed_repeat)
            {
                if(failed_k_index == -1 && failed_k_list.size() < max_failed_k_list_size)
                {
                    failed_k_list.push_back(k);
                }
            }
            done = true;
            reward -= 0.0;
        }
        else
        {
            done = false;
        }
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

    void Simulator::set_k(int agent_id, Eigen::Ref<Eigen::Matrix<double, -1, -1, 1>> k)
    {
        Simulator::k = k.row(agent_id);
    }

    void Simulator::set_state(int agent_id, Eigen::Ref<Eigen::Matrix<double, -1, -1, 1>> state, Eigen::Ref<Eigen::Matrix<double, -1, -1, 1>> obs)
    {
        Simulator::x = state.row(agent_id);
        Eigen::Ref<Eigen::Matrix<double, -1, 1>> single_obs(obs.row(agent_id));
        omega_lpf.last_output = x.segment(10, 3);
        if (add_noise)
        {
            single_obs.segment(0, 13) = x.segment(0, 13);
            single_obs.segment(13, 4) = u;
            get_acc(single_obs(17));
            omega_lpf.calc_derivative(x.segment(10, 3), single_obs.segment(18, 3));
            get_noise(obs_noise_std, singal_obs_num, obs_noise);
            single_obs.segment(0, singal_obs_num) += obs_noise;
            single_obs.segment(6, 4) = single_obs.segment(6, 4) / single_obs.segment(6, 4).norm();
        }
        else
        {
            single_obs.segment(0, 13) = x.segment(0, 13);
            single_obs.segment(13, 4) = u;
            get_acc(single_obs(17));
            omega_lpf.calc_derivative(x.segment(10, 3), single_obs.segment(18, 3));
        }
        for (int i = 1; i < obs_list_num; i++)
        {
            single_obs.segment(singal_obs_num * i, singal_obs_num) = single_obs.segment(0, singal_obs_num);
        }

        for (int i = 0; i < obs_list_num; i++)
        {
            single_obs.segment(singal_obs_num * i, singal_obs_num) =
                single_obs.segment(singal_obs_num * i, singal_obs_num).cwiseQuotient(obs_normalized_max);
        }
    }

    void Simulator::get_state(int agent_id, Eigen::Ref<Eigen::Matrix<double, -1, -1, 1>> state)
    {
        state.row(agent_id) = x;
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

    void LPF_t::calc(Eigen::Ref<Eigen::Matrix<double, -1, 1>> input,
                     Eigen::Ref<Eigen::Matrix<double, -1, 1>> output)
    {
        output = (cutoff_freq * ts * input + last_output) / (1 + cutoff_freq * ts);
        last_output = output;
    }

    void LPF_t::calc_derivative(Eigen::Ref<Eigen::Matrix<double, -1, 1>> input,
                                Eigen::Ref<Eigen::Matrix<double, -1, 1>> derivative)
    {
        output = (cutoff_freq * ts * input + last_output) / (1 + cutoff_freq * ts);
        derivative = (output - last_output) / ts;
        last_output = output;
    }
}
