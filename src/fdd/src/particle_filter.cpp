#include "fdd/particle_filter.hpp"
#include <chrono>
#include <iostream>
#include <omp.h>

namespace ParticleFilter
{
    Particle::Particle():
    uav_simulator(), gen(rd()), k_dis(0, predict_k_sigma), x_dis(0, predict_x_sigma), x(Eigen::Matrix<double, UavSimulator::NX, 1>::Zero()), k(Eigen::Matrix<double, UavSimulator::NK, 1>::Zero()), weights(0)
    {
    }
    void Particle::SetInitState(Eigen::Matrix<double, UavSimulator::Nobs, 1> &obs)
    {
        x.segment(0,UavSimulator::Nobs) = obs;
        x.segment(0,UavSimulator::Nobs) = x.segment(0,UavSimulator::Nobs) + Eigen::Matrix<double, UavSimulator::Nobs, 1>::Random()*predict_x_sigma;
        x.segment(6,4) = x.segment(6,4)/x.segment(6,4).norm();
        k = (Eigen::Matrix<double, UavSimulator::NK, 1>::Random()+Eigen::Matrix<double, UavSimulator::NK, 1>::Ones())/2;
    }
    void Particle::Update(Eigen::Matrix<double, UavSimulator::Nobs, 1> &obs, Eigen::Matrix<double, UavSimulator::NU,1> &u)
    {
        for(int i = 0;i<UavSimulator::NK; i++)
        {
            k(i) = k(i) + k_dis(gen);
            k(i) = k(i) > 1 ? 1 : k(i);
            k(i) = k(i) < 0 ? 0 : k(i);
        }
        uav_simulator.noise = Eigen::Matrix<double, UavSimulator::NX, 1>::Ones();
        for(int i = 0;i<UavSimulator::NX; i++)
        {
            uav_simulator.noise(i) = uav_simulator.noise(i) + x_dis(gen);
        }
        uav_simulator.x0 = x;
        uav_simulator.k = k;
        uav_simulator.u = u;
        uav_simulator.step();
        x = uav_simulator.x1;
        weights = calculate_temp_constance * std::exp(-0.5 * (obs - x.segment(0,UavSimulator::Nobs)).squaredNorm() / std::pow(predict_z_sigma, 2));
    }

    Filter::Filter(Eigen::Matrix<double, UavSimulator::Nobs, 1> obs): gen(rd()), dis(0.0, 1.0), particles_ptr(num_particles), 
    x_est(Eigen::Matrix<double, UavSimulator::NX, 1>::Zero()), k_est(Eigen::Matrix<double, UavSimulator::NK, 1>::Zero())
    {
        for(int i = 0; i < num_particles; i++)
        {
            particles_ptr[i] = std::make_unique<Particle>();
            particles_ptr[i]->SetInitState(obs);
        }
        x_matrix = Eigen::MatrixXd::Zero(UavSimulator::NX, num_particles);
        k_matrix = Eigen::MatrixXd::Zero(UavSimulator::NK, num_particles);
    }

    void Filter::SimpleResample()
    {
        std::vector<double> weights;
        std::vector<double> cumulative_sum(num_particles);
        for(int i = 0; i < num_particles; i++)
            weights.push_back(particles_ptr[i]->weights);
        std::partial_sum(weights.begin(), weights.end(), cumulative_sum.begin());
        

        for(int i = 0; i < num_particles; i++)
        {
            cumulative_sum[i] /= cumulative_sum.back();
        }

        std::vector<double> rn(num_particles);
        std::generate(rn.begin(), rn.end(), [&]() { return dis(gen); });

        std::vector<Eigen::Matrix<double, UavSimulator::NX, 1>> x_list(num_particles);
        std::vector<Eigen::Matrix<double, UavSimulator::NK, 1>> k_list(num_particles);
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num_particles; i++) {
            auto index = std::lower_bound(cumulative_sum.begin(), cumulative_sum.end(), rn[i]) - cumulative_sum.begin();
            x_list[i] = particles_ptr[index]->x;
            k_list[i] = particles_ptr[index]->k;
        }
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num_particles; i++) {
            particles_ptr[i]->x = x_list[i];
            particles_ptr[i]->k = k_list[i];
        }
    }

    void Filter::GetEstimate()
    {
        #pragma omp parallel for schedule(dynamic)
        for(int i = 0; i < num_particles; i++)
        {
            x_matrix.col(i) = particles_ptr[i]->x;
            k_matrix.col(i) = particles_ptr[i]->k;
        }
        x_est = x_matrix.rowwise().mean();
        k_est = k_matrix.rowwise().mean();
    }

    void Filter::Loop(Eigen::Matrix<double, UavSimulator::Nobs, 1> &obs, Eigen::Matrix<double, UavSimulator::NU,1> &u)
    {
        // auto start_time = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(dynamic)
        for(int i = 0; i < num_particles; i++)
        {
            particles_ptr[i]->Update(obs, u);
        }
        SimpleResample();
        GetEstimate();
        // auto end_time = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        // std::cout << "solve time: " << duration.count() << " microseconds" << std::endl;
    }
}