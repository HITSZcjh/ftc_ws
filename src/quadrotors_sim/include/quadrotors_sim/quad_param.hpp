#pragma once
#include "quadrotors_sim/utils.hpp"
#include <iostream>
#include <algorithm> // std::clamp 所在的头文件

namespace quadrotors
{
    // const
    // constexpr Scalar g_real = 9.81;
    // constexpr Scalar g_range[2] = {g_real * 0.999, g_real * 1.001};
    // constexpr Scalar rotor_time_constant_up_real = 0.0125;
    // constexpr Scalar rotor_time_constant_up_range[2] = {rotor_time_constant_up_real * 0.9, rotor_time_constant_up_real * 1.1};
    // constexpr Scalar rotor_time_constant_down_real = 0.025;
    // constexpr Scalar rotor_time_constant_down_range[2] = {rotor_time_constant_down_real * 0.9, rotor_time_constant_down_real * 1.1};
    // constexpr Scalar rotor_moment_constant_real = 0.016;
    // constexpr Scalar rotor_moment_constant_range[2] = {rotor_moment_constant_real * 0.8, rotor_moment_constant_real * 1.2};
    // constexpr Scalar body_length_real = 0.17;
    // constexpr Scalar body_length_range[2] = {body_length_real * 0.95, body_length_real * 1.05};
    // constexpr Scalar mass_real = 0.73;
    // constexpr Scalar mass_range[2] = {mass_real * 0.95, mass_real * 1.05};
    // constexpr Scalar inertia_real[3] = {0.007, 0.007, 0.012};
    // constexpr Scalar inertia_range[3][2] = {{inertia_real[0] * 0.9, inertia_real[0] * 1.1},
    //                                         {inertia_real[1] * 0.9, inertia_real[1] * 1.1},
    //                                         {inertia_real[2] * 0.9, inertia_real[2] * 1.1}};
    // constexpr Scalar vel_drag_factor_real = 0.1;
    // constexpr Scalar vel_drag_factor_range[2] = {vel_drag_factor_real * 0.8, vel_drag_factor_real * 1.2};
    // constexpr Scalar omega_drag_factor_real = 4e-4;
    // constexpr Scalar omega_drag_factor_range[2] = {omega_drag_factor_real * 0.8, omega_drag_factor_real * 1.2};


    constexpr Scalar g_real = 9.81;
    constexpr Scalar g_range[2] = {g_real * 0.999, g_real * 1.001};
    constexpr Scalar rotor_time_constant_up_real = 0.0125;
    constexpr Scalar rotor_time_constant_up_range[2] = {rotor_time_constant_up_real * 0.9, rotor_time_constant_up_real * 1.1};
    constexpr Scalar rotor_time_constant_down_real = 0.025;
    constexpr Scalar rotor_time_constant_down_range[2] = {rotor_time_constant_down_real * 0.9, rotor_time_constant_down_real * 1.1};
    constexpr Scalar rotor_moment_constant_real = 0.016;
    constexpr Scalar rotor_moment_constant_range[2] = {rotor_moment_constant_real * 0.8, rotor_moment_constant_real * 1.2};
    constexpr Scalar body_length_real = 0.17;
    constexpr Scalar body_length_range[2] = {body_length_real * 0.95, body_length_real * 1.05};
    constexpr Scalar mass_real = 0.73;
    constexpr Scalar mass_range[2] = {mass_real * 0.95, mass_real * 1.05};
    constexpr Scalar inertia_real[3] = {0.007, 0.007, 0.012};
    constexpr Scalar inertia_range[3][2] = {{inertia_real[0] * 0.95, inertia_real[0] * 1.05},
                                            {inertia_real[1] * 0.95, inertia_real[1] * 1.05},
                                            {inertia_real[2] * 0.95, inertia_real[2] * 1.05}};
    constexpr Scalar vel_drag_factor_real = 0.1;
    constexpr Scalar vel_drag_factor_range[2] = {vel_drag_factor_real * 0.8, vel_drag_factor_real * 1.2};
    constexpr Scalar omega_drag_factor_real = 4e-4;
    constexpr Scalar omega_drag_factor_range[2] = {omega_drag_factor_real * 0.5, omega_drag_factor_real * 1.5};

    // constexpr Scalar g_real = 9.81;
    // constexpr Scalar g_range[2] = {g_real * 0.95, g_real * 1.05};
    // constexpr Scalar rotor_time_constant_up_real = 0.0125;
    // constexpr Scalar rotor_time_constant_up_range[2] = {rotor_time_constant_up_real * 0.7, rotor_time_constant_up_real * 1.3};
    // constexpr Scalar rotor_time_constant_down_real = 0.025;
    // constexpr Scalar rotor_time_constant_down_range[2] = {rotor_time_constant_down_real * 0.7, rotor_time_constant_down_real * 1.3};
    // constexpr Scalar rotor_moment_constant_real = 0.016;
    // constexpr Scalar rotor_moment_constant_range[2] = {rotor_moment_constant_real * 0.5, rotor_moment_constant_real * 1.5};
    // constexpr Scalar body_length_real = 0.17;
    // constexpr Scalar body_length_range[2] = {body_length_real * 0.8, body_length_real * 1.2};
    // constexpr Scalar mass_real = 0.73;
    // constexpr Scalar mass_range[2] = {mass_real * 0.8, mass_real * 1.2};
    // constexpr Scalar inertia_real[3] = {0.007, 0.007, 0.012};
    // constexpr Scalar inertia_range[3][2] = {{inertia_real[0] * 0.7, inertia_real[0] * 1.3},
    //                                         {inertia_real[1] * 0.7, inertia_real[1] * 1.3},
    //                                         {inertia_real[2] * 0.7, inertia_real[2] * 1.3}};
    // constexpr Scalar vel_drag_factor_real = 0.1;
    // constexpr Scalar vel_drag_factor_range[2] = {vel_drag_factor_real * 0.7, vel_drag_factor_real * 1.3};
    // constexpr Scalar omega_drag_factor_real = 4e-4;
    // constexpr Scalar omega_drag_factor_range[2] = {omega_drag_factor_real * 0.7, omega_drag_factor_real * 1.3};


    constexpr Scalar thrust_range[2] = {0, 6};

    class Integrator;
    class QuadParam
    {
    private:
        Vector<3> g;
        Scalar rotor_time_constant_up;
        Scalar rotor_time_constant_down;
        Scalar rotor_moment_constant;
        Scalar body_length;
        Scalar mass;
        Matrix<3, 3> inertia;
        Scalar vel_drag_factor;
        Scalar omega_drag_factor;
        Vector<4> k;

        Scalar mass_inv;
        Matrix<4, 4> alloc_mat;
        Matrix<3, 3> inertia_inv;
        Scalar rotor_time_constant_up_inv;
        Scalar rotor_time_constant_down_inv;
        friend class Integrator;

    public:
        std::normal_distribution<Scalar> normal_dist_{0.0, 1.0};
        std::uniform_real_distribution<Scalar> uniform_dist_{0, 1};
        std::random_device rd_;
        std::mt19937 random_gen_{rd_()};
        // void random_all()
        // {
        //     g.setZero();
        //     g(2) = -(uniform_dist_(random_gen_) * (g_range[1] - g_range[0]) + g_range[0]);
        //     rotor_time_constant_up = uniform_dist_(random_gen_) * (rotor_time_constant_up_range[1] - rotor_time_constant_up_range[0]) + rotor_time_constant_up_range[0];
        //     rotor_time_constant_down = uniform_dist_(random_gen_) * (rotor_time_constant_down_range[1] - rotor_time_constant_down_range[0]) + rotor_time_constant_down_range[0];
        //     rotor_moment_constant = uniform_dist_(random_gen_) * (rotor_moment_constant_range[1] - rotor_moment_constant_range[0]) + rotor_moment_constant_range[0];
        //     body_length = uniform_dist_(random_gen_) * (body_length_range[1] - body_length_range[0]) + body_length_range[0];
        //     mass = uniform_dist_(random_gen_) * (mass_range[1] - mass_range[0]) + mass_range[0];

        //     Scalar inertia_xx = uniform_dist_(random_gen_) * (inertia_range[0][1] - inertia_range[0][0]) + inertia_range[0][0];
        //     Scalar inertia_yy = uniform_dist_(random_gen_) * (inertia_range[1][1] - inertia_range[1][0]) + inertia_range[1][0];
        //     Scalar inertia_zz = uniform_dist_(random_gen_) * (inertia_range[2][1] - inertia_range[2][0]) + inertia_range[2][0];
        //     inertia << inertia_xx, 0, 0,
        //         0, inertia_yy, 0,
        //         0, 0, inertia_zz;

        //     vel_drag_factor = uniform_dist_(random_gen_) * (vel_drag_factor_range[1] - vel_drag_factor_range[0]) + vel_drag_factor_range[0];
        //     omega_drag_factor = uniform_dist_(random_gen_) * (omega_drag_factor_range[1] - omega_drag_factor_range[0]) + omega_drag_factor_range[0];

        //     k.setOnes();
        //     Scalar random = uniform_dist_(random_gen_);
        //     Scalar random_k;
        //     if (random < 0.3)
        //         random_k = 0;
        //     else if (random < 0.8)
        //         random_k = uniform_dist_(random_gen_) / 2;
        //     else
        //         random_k = 1 - uniform_dist_(random_gen_) / 2;

        //     random = uniform_dist_(random_gen_);
        //     if (random < 0.2)
        //         k(0) = random_k;
        //     else if (random < 0.4)
        //         k(1) = random_k;
        //     else if (random < 0.6)
        //         k(2) = random_k;
        //     else if (random < 0.8)
        //         k(3) = random_k;

        //     // due with intermediate term
        //     mass_inv = 1 / mass;
        //     alloc_mat << 0, body_length, 0, -body_length,
        //         -body_length, 0, body_length, 0,
        //         rotor_moment_constant, -rotor_moment_constant, rotor_moment_constant, -rotor_moment_constant,
        //         1, 1, 1, 1;
        //     inertia_inv = inertia.inverse();

        //     rotor_time_constant_up_inv = 1 / rotor_time_constant_up;
        //     rotor_time_constant_down_inv = 1 / rotor_time_constant_down;
        // }

        void random_all()
        {
            g.setZero();
            g(2) = -gaussian_random(g_range[0],g_range[1]);
            rotor_time_constant_up = gaussian_random(rotor_time_constant_up_range[0],rotor_time_constant_up_range[1]);
            rotor_time_constant_down = gaussian_random(rotor_time_constant_down_range[0],rotor_time_constant_down_range[1]);
            rotor_moment_constant = gaussian_random(rotor_moment_constant_range[0],rotor_moment_constant_range[1]);
            body_length = gaussian_random(body_length_range[0],body_length_range[1]);
            mass = gaussian_random(mass_range[0],mass_range[1]);

            Scalar inertia_xx = gaussian_random(inertia_range[0][0],inertia_range[0][1]);
            Scalar inertia_yy = gaussian_random(inertia_range[1][0],inertia_range[1][1]);
            Scalar inertia_zz = gaussian_random(inertia_range[2][0],inertia_range[2][1]);
            inertia << inertia_xx, 0, 0,
                0, inertia_yy, 0,
                0, 0, inertia_zz;

            vel_drag_factor = gaussian_random(vel_drag_factor_range[0],vel_drag_factor_range[1]);
            omega_drag_factor = gaussian_random(omega_drag_factor_range[0],omega_drag_factor_range[1]);

            k.setOnes();
            Scalar random = uniform_dist_(random_gen_);
            Scalar random_k;
            if (random < 0.3)
                random_k = 0;
            else if (random < 0.8)
                random_k = uniform_dist_(random_gen_) / 2;
            else
                random_k = 1 - uniform_dist_(random_gen_) / 2;

            random = uniform_dist_(random_gen_);
            if (random < 0.2)
                k(0) = random_k;
            else if (random < 0.4)
                k(1) = random_k;
            else if (random < 0.6)
                k(2) = random_k;
            else if (random < 0.8)
                k(3) = random_k;

            // due with intermediate term
            mass_inv = 1 / mass;
            alloc_mat << 0, body_length, 0, -body_length,
                -body_length, 0, body_length, 0,
                rotor_moment_constant, -rotor_moment_constant, rotor_moment_constant, -rotor_moment_constant,
                1, 1, 1, 1;
            inertia_inv = inertia.inverse();

            rotor_time_constant_up_inv = 1 / rotor_time_constant_up;
            rotor_time_constant_down_inv = 1 / rotor_time_constant_down;
        }

        void random_only_k()
        {
            k.setOnes();
            Scalar random = uniform_dist_(random_gen_);
            Scalar random_k;
            if (random < 0.3)
                random_k = 0;
            else if (random < 0.8)
                random_k = uniform_dist_(random_gen_) / 2;
            else
                random_k = 1 - uniform_dist_(random_gen_) / 2;

            random = uniform_dist_(random_gen_);
            if (random < 0.2)
                k(0) = random_k;
            else if (random < 0.4)
                k(1) = random_k;
            else if (random < 0.6)
                k(2) = random_k;
            else if (random < 0.8)
                k(3) = random_k;
        }

        void no_random()
        {
            g << 0, 0, -g_real;
            rotor_time_constant_up = rotor_time_constant_up_real;
            rotor_time_constant_down = rotor_time_constant_down_real;
            rotor_moment_constant = rotor_moment_constant_real;
            body_length = body_length_real;
            mass = mass_real;
            inertia << inertia_real[0], 0, 0,
                0, inertia_real[1], 0,
                0, 0, inertia_real[2];
            vel_drag_factor = vel_drag_factor_real;
            omega_drag_factor = omega_drag_factor_real;
            k.setOnes();

            // due with intermediate term
            mass_inv = 1 / mass;
            alloc_mat << 0, body_length, 0, -body_length,
                -body_length, 0, body_length, 0,
                rotor_moment_constant, -rotor_moment_constant, rotor_moment_constant, -rotor_moment_constant,
                1, 1, 1, 1;
            inertia_inv = inertia.inverse();

            rotor_time_constant_up_inv = 1 / rotor_time_constant_up;
            rotor_time_constant_down_inv = 1 / rotor_time_constant_down;
        }

        Scalar gaussian_random(const Scalar &min, const Scalar &max)
        {
            return clamp(normal_dist_(random_gen_) * (max - min) / 6 + (min + max) / 2, min, max);
        }

        inline void set_k(VectorRef<Scalar> k)
        {
            QuadParam::k = k;
        }

        inline Vector<4> get_k() const
        {
            return k;
        }

        QuadParam()
        {
            no_random();
        }

        friend std::ostream &operator<<(std::ostream &os, const QuadParam &quad_param)
        {
            os.precision(4);
            os << "QUadrotor Param:\n"
               << "g =                        [" << quad_param.g.transpose() << "]\n"
               << "rotor_time_constant_up =   [" << quad_param.rotor_time_constant_up << "]\n"
               << "rotor_time_constant_down = [" << quad_param.rotor_time_constant_down << "]\n"
               << "rotor_moment_constant =    [" << quad_param.rotor_moment_constant << "]\n"
               << "body_length =              [" << quad_param.body_length << "]\n"
               << "mass =                     [" << quad_param.mass << "]\n"
               << "inertia =                  [" << quad_param.inertia.row(0) << "]\n"
               << "                           [" << quad_param.inertia.row(1) << "]\n"
               << "                           [" << quad_param.inertia.row(2) << "]\n"
               << "vel_drag_factor =          [" << quad_param.vel_drag_factor << "]\n"
               << "omega_drag_factor =        [" << quad_param.omega_drag_factor << "]\n"
               << "k =                        [" << quad_param.k.transpose() << "]\n"
               << std::endl;
            os.precision();
            return os;
        }
    };

}