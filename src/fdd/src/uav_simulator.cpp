#include "fdd/uav_simulator.hpp"

namespace FDD
{
    Simulator::Simulator(double ts)
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
        sim_in_set(config, dims, in, "T", &ts);
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

    void Simulator::step(double *action, double *p, double *state)
    {
        // set boundary
        UAVModel_acados_sim_update_params(capsule, p, NP);
        sim_in_set(config, dims, in, "x", state);
        sim_in_set(config, dims, in, "u", action);
        status = UAVModel_acados_sim_solve(capsule);
        if (status != ACADOS_SUCCESS)
        {
            printf("UAVModel_acados_sim_solve() returned status %d. Exiting.\n", status);
            exit(1);
        }
        sim_out_get(config, dims, out, "x", state);
    }

    // void Simulator::get_obs(Eigen::Matrix<double, Nobs, 1> &noise)
    // {
    //     obs = obs_map + noise;
    //     obs.segment(6,4) = obs.segment(6,4)/obs.segment(6,4).norm();
    // }

    Simulator::~Simulator()
    {
        status = UAVModel_acados_sim_free(capsule);
        if (status)
        {
            printf("UAVModel_acados_sim_free() returned status %d. \n", status);
        }

        UAVModel_acados_sim_solver_free_capsule(capsule);
    }
}
