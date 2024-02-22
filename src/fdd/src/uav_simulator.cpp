#include "fdd/uav_simulator.hpp"

namespace UavSimulator
{
    Simulator::Simulator():
    x0(x0_data, NX),x1(x1_data, NX), u(u0_data, NU), noise(p_data, NX), k(p_data + NX, NK), obs_map(x1_data, Nobs)
    {
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
        x0.setZero();
        x0(2) = 3;
        x0(6) = 1;
        x1 = x0;
        u.setZero();
        noise.setOnes();
        k.setOnes();
    }

    void Simulator::test()
    {
        status = UAVModel_acados_sim_solve(capsule);
        if (status != ACADOS_SUCCESS)
        {
            printf("UAVModel_acados_sim_solve() returned status %d. Exiting.\n", status);
            exit(1);
        }
    }

    void Simulator::step()
    {
        // set boundary
        u = u.cwiseMax(u_range[0]).cwiseMin(u_range[1]);
        UAVModel_acados_sim_update_params(capsule, p_data, NP);
        sim_in_set(config, dims, in, "x", x0_data);
        sim_in_set(config, dims, in, "u", u0_data);
        status = UAVModel_acados_sim_solve(capsule);
        if (status != ACADOS_SUCCESS)
        {
            printf("UAVModel_acados_sim_solve() returned status %d. Exiting.\n", status);
            exit(1);
        }
        sim_out_get(config, dims, out, "x", x1_data);
        x1.segment(3,3) = x1.segment(3,3).cwiseMax(velocity_range[0]).cwiseMin(velocity_range[1]);
        x1.segment(10,3) = x1.segment(10,3).cwiseMax(omega_range[0]).cwiseMin(omega_range[1]);
        x1.segment(6,4) = x1.segment(6,4)/x1.segment(6,4).norm();
    }

    void Simulator::get_obs(Eigen::Matrix<double, Nobs, 1> &noise)
    {
        obs = obs_map + noise;
        obs.segment(6,4) = obs.segment(6,4)/obs.segment(6,4).norm();
    }

    Simulator::~Simulator()
    {
        status = UAVModel_acados_sim_free(capsule);
        if (status) {
            printf("UAVModel_acados_sim_free() returned status %d. \n", status);
        }

        UAVModel_acados_sim_solver_free_capsule(capsule);
    }
}

