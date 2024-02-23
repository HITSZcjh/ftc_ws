#pragma once

// standard
#include <stdio.h>
#include <stdlib.h>
// acados
#include "acados/utils/print.h"
#include "acados/utils/math.h"
#include "acados_c/sim_interface.h"
#include "acados_sim_solver_UAVModel.h"

#include <Eigen/Eigen>

namespace FDD
{

    constexpr int NX = UAVMODEL_NX;
    constexpr int NU = UAVMODEL_NU;
    constexpr int NP = UAVMODEL_NP;
    constexpr int NK = 4;
    constexpr int Nobs = 13;
    class Simulator
    {
    public:
        Simulator(double ts);
        ~Simulator();
        void step(double *action, double *p, double *state);
        void test();
        // void get_obs(Eigen::Matrix<double, Nobs, 1> &noise);
    private:
        UAVModel_sim_solver_capsule *capsule;
        sim_config *config;
        sim_in *in;
        sim_out *out;
        void *dims;
        int status;
    };
}