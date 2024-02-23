#include "fdd/particle_filter.hpp"
#include <chrono>
#include <iostream>
#include <omp.h>

namespace FDD
{

    Particle::Particle(double *state_data, double *k_data, double *weights, double ts) : simulator(ts), state_data(state_data), k_data(k_data), weights(weights),
                                                                                         state(state_data, NX), k(k_data, NK), dist(0, 1)
    {
        Eigen::Map<const Eigen::VectorXd> obs_noise_std_map(obs_noise_std, Nobs);
        Eigen::MatrixXd cov = obs_noise_std_map.array().square().matrix().asDiagonal();
        obs_noise_cov_inv = cov.inverse();
        obs_noise_coef = 1.0 / std::sqrt(std::pow(2 * M_PI, Nobs) * cov.determinant());
    };

    Eigen::VectorXd Particle::GetSamples(const double *std, int num)
    {
        Eigen::VectorXd samples(num);
        for (int i = 0; i < num; i++)
        {
            samples(i) = dist(random_gen_) * std[i];
        }
        return samples;
    }

    double Particle::GetPDF(Eigen::Ref<Eigen::VectorXd> x,
                            Eigen::Ref<Eigen::VectorXd> mean,
                            Eigen::Ref<Eigen::MatrixXd> cov_inv,
                            double &coef)
    {
        Eigen::VectorXd diff = x - mean;
        double exp_val = -0.5 * diff.transpose() * cov_inv * diff;
        return coef * std::exp(exp_val);
    }

    void Particle::SetInitState(Eigen::Ref<Eigen::Matrix<double, Nobs, 1>> obs)
    {
        state.setZero();
        state.segment(0, Nobs) = obs + GetSamples(obs_noise_std, Nobs);
        state.segment(6, 4) /= state.segment(6, 4).norm();

        k.setOnes();
    }
    void Particle::Update(double *action, Eigen::Ref<Eigen::Matrix<double, Nobs, 1>> obs)
    {
        k += GetSamples(k_std, NK);
        k = k.cwiseMax(k_range[0]).cwiseMin(k_range[1]);

        Eigen::VectorXd state_noise = GetSamples(state_noise_std, NX);
        for (int i = 0; i < NX; i++)
        {
            p_data[i] = state_noise(i);
        }
        for (int i = 0; i < NK; i++)
        {
            p_data[i + NX] = k(i);
        }
        simulator.step(action, p_data, state_data);
        state.segment(6, 4) /= state.segment(6, 4).norm();
        *weights = GetPDF(state.segment(0, Nobs), obs, obs_noise_cov_inv, obs_noise_coef);
    }

    ParticleFilter::ParticleFilter(int num_particles, int num_threads, double ts) : dist(0.0, 1.0), particles_list(num_particles), state_matrix_data(new double[num_particles * NX]), k_matrix_data(new double[num_particles * NK]),
                                                                                    state_matrix(state_matrix_data, NX, num_particles), k_matrix(k_matrix_data, NK, num_particles), weights(Eigen::VectorXd::Zero(num_particles)), num_particles(num_particles)
    {
        for (int i = 0; i < num_particles; i++)
        {
            particles_list[i] = std::make_unique<Particle>(&state_matrix_data[i * NX], &k_matrix_data[i * NK], &weights[i], ts);
        }
        omp_set_num_threads(num_threads);
    }
    void ParticleFilter::SimpleResample()
    {
        double sum = weights.sum();
        if (sum < 1e-8)
        {
            weights = Eigen::VectorXd::Ones(num_particles) / num_particles;
        }
        else
        {
            weights /= sum;
        }
        Eigen::VectorXd cumulative_sum = weights;
        for (int i = 1; i < num_particles; i++)
        {
            cumulative_sum(i) += cumulative_sum(i - 1);
        }

        Eigen::VectorXd rn(num_particles);
        for (int i = 0; i < num_particles; i++)
        {
            rn(i) = dist(random_gen_);
        }
        Eigen::VectorXi index(num_particles);
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num_particles; i++)
        {
            index[i] = std::distance(cumulative_sum.data(), std::upper_bound(cumulative_sum.data(), cumulative_sum.data() + cumulative_sum.size(), rn[i]));
        }
        Eigen::MatrixXd state_matrix_temp(state_matrix);
        Eigen::MatrixXd k_matrix_temp(k_matrix);
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num_particles; i++)
        {
            state_matrix.col(i) = state_matrix_temp.col(index[i]);
            k_matrix.col(i) = k_matrix_temp.col(index[i]);
        }
    }

    void ParticleFilter::GetEstimate()
    {
        state_est = state_matrix.rowwise().mean();
        k_est = k_matrix.rowwise().mean();
    }

    void ParticleFilter::SetInitState(Eigen::Ref<Eigen::Matrix<double, -1, 1>> obs)
    {
        if(obs.rows()!=Nobs)
        {
            std::cerr<<"Invalid input size!"<<std::endl;
            return;
        }
        for (int i = 0; i < num_particles; i++)
        {
            particles_list[i]->SetInitState(obs);
        }
    }

    void ParticleFilter::GetStateMatrix(Eigen::Ref<Eigen::Matrix<double, -1, -1, 1>> state_matrix)
    {
        state_matrix = ParticleFilter::state_matrix;
    }

    void ParticleFilter::Loop(Eigen::Ref<Eigen::Matrix<double, -1, 1>> obs,
                              Eigen::Ref<Eigen::Matrix<double, -1, 1>> action,
                              Eigen::Ref<Eigen::Matrix<double, -1, 1>> state_est,
                              Eigen::Ref<Eigen::Matrix<double, -1, 1>> k_est)
    {
        if(obs.rows()!=Nobs||action.rows()!=NU)
        {
            std::cerr<<"Invalid input size!"<<std::endl;
            return;
        }
        action = action.cwiseMax(action_range[0]).cwiseMin(action_range[1]);
// auto start_time = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num_particles; i++)
        {
            particles_list[i]->Update(action.data(), obs);
        }
        SimpleResample();
        GetEstimate();
        state_est = ParticleFilter::state_est;
        k_est = ParticleFilter::k_est;
        // auto end_time = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        // std::cout << "solve time: " << duration.count() << " microseconds" << std::endl;
    }
}