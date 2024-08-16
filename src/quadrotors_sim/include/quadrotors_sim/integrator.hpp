#pragma once

#include "quadrotors_sim/utils.hpp"
#include "quadrotors_sim/quad_param.hpp"

namespace quadrotors
{
    constexpr Scalar rk_max_dt = 2.5e-3;
    constexpr Scalar lpf_tau_inv = 1/0.06;

    class Integrator
    {
    public:
        Vector<NX> ode_func(const Vector<NX> &x, const Vector<NU> &u, const QuadParam &quad_param, const Vector<NX> &ode_func_noise)
        {
            Vector<NX> x_dot;
            x_dot.segment<NP>(P) = x.segment<NV>(V);
            Vector<4> torques_force = quad_param.alloc_mat * x.segment<NTHRUSTS>(THRUSTS_REAL);
            Vector<3> total_force(0, 0, torques_force(3));
            Vector<3> vel_drag = -quad_param.vel_drag_factor * x.segment<NV>(V);
            x_dot.segment<NV>(V) = quad_param.g + quad_param.mass_inv * (R(x.segment<NQ>(Q)) * total_force + vel_drag);

            x_dot.segment<NQ>(Q) << -x(QX) * x(WX) - x(QY) * x(WY) - x(QZ) * x(WZ),
                x(QW) * x(WX) + x(QY) * x(WZ) - x(QZ) * x(WY),
                x(QW) * x(WY) - x(QX) * x(WZ) + x(QZ) * x(WX),
                x(QW) * x(WZ) + x(QX) * x(WY) - x(QY) * x(WX);
            x_dot.segment<NQ>(Q) /= 2;

            Vector<3> omega_sign;
            for (int i = 0; i < 3; i++)
            {
                if (x(W + i) > 0)
                    omega_sign(i) = 1;
                else
                    omega_sign(i) = -1;
            }

            Vector<3> omega_drag = -quad_param.omega_drag_factor * omega_sign.array() * x.segment<NW>(W).array().square();
            x_dot.segment<NW>(W) = quad_param.inertia_inv * (torques_force.segment<3>(0) + omega_drag - x.segment<NW>(W).cross(quad_param.inertia * x.segment<NW>(W)));

            for (int i = 0; i < NTHRUSTS; i++)
            {
                Scalar dot = quad_param.k(i) * x(i + U_LPF) - x(i + THRUSTS_REAL);
                if (dot)
                    x_dot(i + THRUSTS_REAL) = quad_param.rotor_time_constant_up_inv * dot;
                else
                    x_dot(i + THRUSTS_REAL) = quad_param.rotor_time_constant_down_inv * dot;
            }

            x_dot.segment<NU>(U_LPF) = lpf_tau_inv * (u - x.segment<NU>(U_LPF));

            x_dot += ode_func_noise;
            return x_dot;
        }

        Vector<NX> rk4(const Vector<NX> &x0, const Vector<NU> &u, Scalar dt, const QuadParam &quad_param, const Vector<NX> &ode_func_noise)
        {
            Vector<NX> k[4];
            k[0] = ode_func(x0, u, quad_param, ode_func_noise);
            k[1] = ode_func(x0 + dt / 2 * k[0], u, quad_param, ode_func_noise);
            k[2] = ode_func(x0 + dt / 2 * k[1], u, quad_param, ode_func_noise);
            k[3] = ode_func(x0 + dt * k[2], u, quad_param, ode_func_noise);
            return x0 + dt / 6 * (k[0] + 2 * k[1] + 2 * k[2] + k[3]);
        }

        Vector<NX> integrate(const Vector<NX> &x0, const Vector<NU> &u, Scalar dt, const QuadParam &quad_param, const Vector<NX> &ode_func_noise)
        {
            Vector<NX> xf = x0;
            Scalar remain_rk_dt = dt;
            while (remain_rk_dt > 0)
            {
                Scalar dt = std::min(remain_rk_dt, rk_max_dt);
                remain_rk_dt -= dt;
                xf = rk4(xf, u, dt, quad_param, ode_func_noise);
            }
            return xf;
        }

        Vector<3> get_acc(const Vector<NX> &x, const QuadParam &quad_param)
        {
            Vector<4> torques_force = quad_param.alloc_mat * x.segment<NTHRUSTS>(THRUSTS_REAL);
            Vector<3> total_force(0, 0, torques_force(3));
            Vector<3> vel_drag = -quad_param.vel_drag_factor * x.segment<NV>(V);
            return (quad_param.mass_inv * (total_force + R(x.segment<NQ>(Q)).transpose() * vel_drag));
        }

        Integrator()
        {
        }
    };
}
