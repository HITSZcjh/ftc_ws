#include "quadrotors_sim/simulator.hpp"

namespace quadrotors
{
    HyperParam::HyperParam(YAML::Node &cfg)
    {
        dt = cfg["env"]["dt"].as<Scalar>();
        pos_coeff = cfg["rl"]["pos_coeff"].as<Scalar>();
        ori_coeff = cfg["rl"]["ori_coeff"].as<Scalar>();
        lin_vel_coeff = cfg["rl"]["lin_vel_coeff"].as<Scalar>();
        ang_vel_coeff = cfg["rl"]["ang_vel_coeff"].as<Scalar>();
        act_coeff = cfg["rl"]["act_coeff"].as<Scalar>();
        max_ep_len = cfg["rl"]["max_ep_len"].as<int>();
    }
    Simulator::Simulator(YAML::Node &cfg) : p(x.segment<NP>(P)),
                                            v(x.segment<NV>(V)),
                                            q(x.segment<NQ>(Q)),
                                            w(x.segment<NW>(W)),
                                            thrusts_real(x.segment<NTHRUSTS>(THRUSTS_REAL)),
                                            u_lpf(x.segment<NU>(U_LPF)),
                                            hyper_param(cfg),
                                            omega_dot_lpf(0.12, hyper_param.dt)
    {
        // init
        x.setZero();
        u.setZero();
        q(0) = 1;
        itr = 0;
        ode_func_noise.setZero();
        obs_noise.setZero();

        extra_info.emplace("pos_reward", 0);
        extra_info.emplace("lin_vel_reward", 0);
        extra_info.emplace("ori_reward", 0);
        extra_info.emplace("ang_vel_reward", 0);
        extra_info.emplace("act_reward", 0);
        extra_info.emplace("live_reward", 0.0);

        extra_info.emplace("pos_err", 0);
        extra_info.emplace("lin_vel_err", 0);
        extra_info.emplace("ori_err", 0);
        extra_info.emplace("ang_vel_err", 0);

        first_reset = true;
    }

    void Simulator::step(Vector<NU> action)
    {
        u = action.cwiseMax(thrust_range[0]).cwiseMin(thrust_range[1]);
        x = integrator.integrate(x, u, hyper_param.dt, quad_param, ode_func_noise);
        q = q / q.norm();
    }

    void Simulator::step(VectorRef<Scalar> actions,
                         VectorRef<Scalar> obs,
                         Scalar &rewards,
                         bool &dones,
                         VectorRef<Scalar> extra_infos)
    {
        u = actions.cwiseMax(thrust_range[0]).cwiseMin(thrust_range[1]);
        x = integrator.integrate(x, u, hyper_param.dt, quad_param, ode_func_noise);
        q = q / q.norm();

        if (p(0) < world_box[0][0])
            p(0) = world_box[0][0];
        if (p(0) > world_box[0][1])
            p(0) = world_box[0][1];
        if (p(1) < world_box[1][0])
            p(1) = world_box[1][0];
        if (p(1) > world_box[1][1])
            p(1) = world_box[1][1];
        if (p(2) < world_box[2][0])
            p(2) = world_box[2][0];
        if (p(2) > world_box[2][1])
            p(2) = world_box[2][1];

        rewards = calc_reward();
        dones = check_done();
        itr++;

        if (dones)
            reset(obs);
        else
        {
            // calc obs
            get_obs(obs);
            Vector<3> omega_dot;
            omega_dot_lpf.calc_derivative(obs.segment<3>(10), omega_dot);
            obs.segment<3>(20) += omega_dot; // add noise
            obs = obs.cwiseQuotient(obs_normalized_max);
        }

        int index = 0;
        for (const auto &info : extra_info)
        {
            extra_infos(index) = info.second;
            index++;
        }
    }

    Scalar Simulator::calc_reward()
    {

        Scalar factor = pow(quad_param.get_k().sum() - 3, 12);
        extra_info["pos_err"] = (p - goal_pos).squaredNorm();
        extra_info["lin_vel_err"] = v.squaredNorm();
        extra_info["ori_err"] = (q.segment<2>(0)).squaredNorm();
        extra_info["ang_vel_err"] = (w.segment<2>(0)).squaredNorm() + factor * w(2) * w(2);

        extra_info["pos_reward"] = hyper_param.pos_coeff * extra_info["pos_err"];
        extra_info["lin_vel_reward"] = hyper_param.lin_vel_coeff * extra_info["lin_vel_err"];
        extra_info["ori_reward"] = hyper_param.ori_coeff * extra_info["ori_err"];
        // Scalar itr_discount = itr / (2.f / hyper_param.dt) < 1.f ? itr / (2.f / hyper_param.dt) : 1.f;
        // itr_discount = 1;
        // Scalar factor = pow(quad_param.get_k().sum() - 3, 12) * itr_discount;
        extra_info["ang_vel_reward"] = hyper_param.ang_vel_coeff * extra_info["ang_vel_err"];
        extra_info["act_reward"] = hyper_param.act_coeff * u.squaredNorm();

        // extra_info["pos_reward"] = hyper_param.pos_coeff * exp(-(x.segment(0, 3) - goal_pos).squaredNorm()/30);
        // extra_info["lin_vel_reward"] = -hyper_param.lin_vel_coeff * (x.segment(3, 3)).squaredNorm();
        // extra_info["ori_reward"] = -hyper_param.ori_coeff * (x.segment(7, 2)).squaredNorm();
        // extra_info["ang_vel_reward"] = hyper_param.ang_vel_coeff * ((w.segment<2>(0)).squaredNorm() + factor * w(2) * w(2));
        // extra_info["act_reward"] = -hyper_param.act_coeff * u.squaredNorm();


        reward = extra_info["pos_reward"] + extra_info["lin_vel_reward"] + extra_info["ori_reward"] +
                 extra_info["ang_vel_reward"] + extra_info["act_reward"] + extra_info["live_reward"];
        total_reward += reward;
        return reward;
    }

    inline bool Simulator::check_done()
    {
        return itr == hyper_param.max_ep_len;
    }

    void Simulator::random_state()
    {
        p(0) = uniform_dist_(random_gen_) * 1;
        p(1) = uniform_dist_(random_gen_) * 1;
        p(2) = uniform_dist_(random_gen_) * 1 + 3;

        v(0) = uniform_dist_(random_gen_) * 1;
        v(1) = uniform_dist_(random_gen_) * 1;
        v(2) = uniform_dist_(random_gen_) * 1;

        Scalar roll = uniform_dist_(random_gen_) * 0.15 * M_PI;
        Scalar pitch = uniform_dist_(random_gen_) * 0.15 * M_PI;
        Scalar yaw = uniform_dist_(random_gen_) * M_PI;
        Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());
        // 将三个 AngleAxis 对象相乘，得到旋转四元数
        Eigen::Quaterniond quaternion = yawAngle * pitchAngle * rollAngle;
        q(0) = quaternion.w();
        q(1) = quaternion.x();
        q(2) = quaternion.y();
        q(3) = quaternion.z();

        w(0) = uniform_dist_(random_gen_) * 1;
        w(1) = uniform_dist_(random_gen_) * 1;
        w(2) = uniform_dist_(random_gen_) * 1;

        thrusts_real(0) = (uniform_dist_(random_gen_) + 1) * thrust_range[1] / 2;
        thrusts_real(1) = (uniform_dist_(random_gen_) + 1) * thrust_range[1] / 2;
        thrusts_real(2) = (uniform_dist_(random_gen_) + 1) * thrust_range[1] / 2;
        thrusts_real(3) = (uniform_dist_(random_gen_) + 1) * thrust_range[1] / 2;

        thrusts_real.setZero();
        u_lpf.setZero();
    }

    void Simulator::reset(VectorRef<Scalar> obs)
    {
        // randomize max_ep_len
        if (first_reset)
        {
            max_ep_len_copy = hyper_param.max_ep_len;
            hyper_param.max_ep_len *= (uniform_dist_(random_gen_) + 1) / 2;
            first_reset = false;
        }
        else
            hyper_param.max_ep_len = max_ep_len_copy;

        itr = 0;
        random_state();
        // quad_param.random_only_k();
        quad_param.random_all();
        // quad_param.set_k((Vector<4>() << 0,1,1,1).finished());

        get_obs(obs);

        omega_dot_lpf.last_input = obs.segment<3>(10);
        omega_dot_lpf.last_output = Vector<3>::Zero();
        obs = obs.cwiseQuotient(obs_normalized_max);
    }

    void Simulator::get_noise_vector(const Scalar *std, VectorRef<Scalar> noise)
    {
        for (int i = 0; i < noise.rows(); i++)
        {
            noise(i) = normal_dist_(random_gen_) * std[i];
        }
    }

    void Simulator::get_obs(VectorRef<Scalar> obs)
    {
        obs.setZero();
        obs.segment<13>(0) = x.segment<13>(0);
        // obs.segment<4>(13) = x.segment<NU>(U_LPF);
        obs.segment<4>(13) = u;
        obs.segment<3>(17) = get_acc();
        get_noise_vector(obs_noise_std, obs_noise);
        obs += obs_noise;
    }

    inline Vector<3> Simulator::get_acc()
    {
        return integrator.get_acc(x, quad_param);
    }

    void Simulator::set_k(VectorRef<Scalar> k)
    {
        quad_param.set_k(k);
    }

    void Simulator::get_k(VectorRef<Scalar> k)
    {
        k = quad_param.get_k();
    }

    void Simulator::set_state(VectorRef<Scalar> state, VectorRef<Scalar> obs)
    {
        x = state;
        get_obs(obs);
        omega_dot_lpf.last_input = obs.segment<3>(10);
        omega_dot_lpf.last_output = Matrix<3, 1>::Zero();
        obs = obs.cwiseQuotient(obs_normalized_max);
    }

    void Simulator::get_state(VectorRef<Scalar> state)
    {
        state = x;
    }

    void Simulator::print_quad_param()
    {
        std::cout << quad_param;
    }
}