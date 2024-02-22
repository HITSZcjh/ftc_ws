import sys
sys.path.append("/home/jiao/ftc_ws")
from src.INDI.scripts.uav_model import SimpleUAVModel
import numpy as np
from scipy.stats import multivariate_normal

class ParticleFilter(object):
    def __init__(self, particle_num, model, ts, log=False) -> None:
        self.particle_num = particle_num
        self.model = model

        self.x_p_extra = np.hstack((self.model.state, self.model.k)) \
                    + np.random.normal(0, 0.01, (self.particle_num, self.model.state.shape[0]+self.model.k.shape[0]))
        # self.x_p_extra[:,-4:] = np.random.uniform(0, 1, (self.particle_num, 4))

        self.k_sigma = 0.05

        self.state_noise_cov = np.diagflat([1.5,1.5,1.5,
                                            2.5,2.5,2.5,
                                            0.0,0.0,0.0,0.0,
                                            1.5,1.5,1.5,
                                            0,0,0,0])**2
        
        self.obs_noise_cov = np.diagflat([0.02,0.02,0.02,
                                          0.1,0.1,0.1,
                                          0.017,0.017,0.017,0.017,
                                          0.1,0.1,0.1])**2
        self.ts = ts
        self.log = log
        if self.log:
            self.log_x_est_list = []

    def loop(self, action, obs):
        weights = []
        obs_dist = multivariate_normal(mean=obs, cov=self.obs_noise_cov)
        state_nosie = multivariate_normal.rvs(mean=np.zeros(self.state_noise_cov.shape[0]), cov=self.state_noise_cov, size=self.particle_num)
        for i in range(self.particle_num):
            self.x_p_extra[i,-4:] += np.random.normal(0,self.k_sigma,4)
            self.x_p_extra[i,-4:] = np.clip(self.x_p_extra[i,-4:], 0, 1)
            self.x_p_extra[i,:-4] = self.model.step(action, state_nosie[i,:], self.x_p_extra[i,-4:], self.x_p_extra[i,:-4])
            weights.append(obs_dist.pdf(self.x_p_extra[i,:13]))

        weights = np.array(weights)
        sum = np.sum(weights)
        if sum == 0:
            weights = np.ones(self.particle_num)/self.particle_num
        else:
            weights /= np.sum(weights)

        x_est, _ = self.estimate(weights)
        self.simple_resample(weights)

        if self.log:
            self.log_x_est_list.append(x_est.copy())
        return x_est

    def estimate(self, weights):
        mean = np.average(self.x_p_extra, weights=weights, axis=0)
        var = np.average((self.x_p_extra - mean) ** 2, weights=weights, axis=0)
        return mean, var

    def simple_resample(self, weights):
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.  # 避免摄入误差
        rn = np.random.rand(self.particle_num)
        indexes = np.searchsorted(cumulative_sum, rn)
        # 根据索引采样
        self.x_p_extra[:,:] = self.x_p_extra[indexes,:]

    def log_show(self):
        if self.log:
            import matplotlib.pyplot as plt
            t = np.arange(0, len(self.log_x_est_list)*self.ts, self.ts)
            self.log_x_est_list = np.array(self.log_x_est_list)

            self.fig, self.axs = plt.subplots(2, 2)
            self.axs[0,0].plot(t, self.log_x_est_list[:, 0], label="px")
            self.axs[0,0].plot(t, self.log_x_est_list[:, 1], label="py")
            self.axs[0,0].plot(t, self.log_x_est_list[:, 2], label="pz")
            self.axs[0,0].legend()
            self.axs[0,1].plot(t, self.log_x_est_list[:, 3], label="vx")
            self.axs[0,1].plot(t, self.log_x_est_list[:, 4], label="vy")
            self.axs[0,1].plot(t, self.log_x_est_list[:, 5], label="vz")
            self.axs[0,1].legend()
            self.axs[1,0].plot(t, self.log_x_est_list[:, 6], label="w")
            self.axs[1,0].plot(t, self.log_x_est_list[:, 7], label="x")
            self.axs[1,0].plot(t, self.log_x_est_list[:, 8], label="y")
            self.axs[1,0].plot(t, self.log_x_est_list[:, 9], label="z")
            self.axs[1,0].legend()
            self.axs[1,1].plot(t, self.log_x_est_list[:, 10], label="wx")
            self.axs[1,1].plot(t, self.log_x_est_list[:, 11], label="wy")
            self.axs[1,1].plot(t, self.log_x_est_list[:, 12], label="wz")
            self.axs[1,1].legend()


            self.fig, self.axs = plt.subplots(2,2)
            self.axs[0,0].plot(t, self.log_x_est_list[:,13], label="f1")
            self.axs[0,0].legend()
            self.axs[0,1].plot(t, self.log_x_est_list[:,14], label="f2")
            self.axs[0,1].legend()
            self.axs[1,0].plot(t, self.log_x_est_list[:,15], label="f3")
            self.axs[1,0].legend()
            self.axs[1,1].plot(t, self.log_x_est_list[:,16], label="f4")
            self.axs[1,1].legend()

            self.fig, self.axs = plt.subplots(2,2)
            self.axs[0,0].plot(t, self.log_x_est_list[:,17], label="k1")
            self.axs[0,0].set_ylim(0,1)
            self.axs[0,0].legend()
            self.axs[0,1].plot(t, self.log_x_est_list[:,18], label="k2")
            self.axs[0,1].set_ylim(0,1)
            self.axs[0,1].legend()
            self.axs[1,0].plot(t, self.log_x_est_list[:,19], label="k3")
            self.axs[1,0].set_ylim(0,1)
            self.axs[1,0].legend()
            self.axs[1,1].plot(t, self.log_x_est_list[:,20], label="k4")
            self.axs[1,1].set_ylim(0,1)
            self.axs[1,1].legend()

            self.fig, self.axs = plt.subplots()
            self.axs.plot(t, self.log_x_est_list[:,17], label="k1")
            self.axs.plot(t, self.log_x_est_list[:,18], label="k2")
            self.axs.plot(t, self.log_x_est_list[:,19], label="k3")
            self.axs.plot(t, self.log_x_est_list[:,20], label="k4")
            self.axs.legend()
            self.axs.set_ylim(0,1)
            # plt.show()
    # def mcl_resample(self):
    #     global w_fast, w_slow
    #     p = max(0.00, 1 - w_fast/w_slow)
    #     # p = min(0.05, p)
    #     global z, action, N, x_est_extra
    #     num_mcl = int(p*N)
    #     print('num_mcl: ', num_mcl)
    #     particles_result = particles.copy()
    #     part1 = z * np.random.normal(1,0.01,(num_mcl,13))
    #     # if x_est_extra is not None:
    #     part2 = np.tile(x_est_extra[-8:-4], (num_mcl,1)) * np.random.normal(1,0.1,(num_mcl,4))
    #     # else:
    #     # part2 = action*np.random.uniform(0,1,(num_mcl,4))
    #     part3 = np.random.uniform(0,1,(num_mcl,4))
    #     particles_result[:num_mcl,:] = np.concatenate((part1,part2,part3), axis=1)

    #     cumulative_sum = np.cumsum(weights)
    #     cumulative_sum[-1] = 1.  # 避免摄入误差
    #     indexes = np.searchsorted(cumulative_sum, np.random.rand(N-num_mcl))
    #     # 根据索引采样
    #     particles_result[num_mcl:,:] = particles[indexes,:]
    #     return particles_result

if __name__=="__main__":
    ParticleFilter(500, SimpleUAVModel())
