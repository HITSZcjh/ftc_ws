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

namespace UavSimulator
{
    constexpr int NX = UAVMODEL_NX;
    constexpr int NU = UAVMODEL_NU;
    constexpr int NP = UAVMODEL_NP;
    constexpr int NK = 4;
    constexpr int Nobs = 13;
    constexpr double u_range[2] = {0, 6};
    constexpr double omega_range[2] = {-5, 5};
    constexpr double velocity_range[2] = {-5, 5};
    class Simulator
    {
    public:
        Simulator();
        ~Simulator();
        void step();
        void test();
        void get_obs(Eigen::Matrix<double, Nobs, 1> &noise);
        double x0_data[NX];
        double u0_data[NU];
        double x1_data[NX];
        double p_data[NP];
        Eigen::Map<Eigen::Matrix<double, NX, 1>> x0;
        Eigen::Map<Eigen::Matrix<double, NU, 1>> u;
        Eigen::Map<Eigen::Matrix<double, NX, 1>> x1;
        Eigen::Map<Eigen::Matrix<double, NX, 1>> noise;
        Eigen::Map<Eigen::Matrix<double, NK, 1>> k;
        Eigen::Map<Eigen::Matrix<double, Nobs, 1>> obs_map;
        Eigen::Matrix<double, Nobs, 1> obs;
    private:
        UAVModel_sim_solver_capsule *capsule;
        sim_config *config;
        sim_in *in;
        sim_out *out;
        void *dims;
        int status;
    };
}