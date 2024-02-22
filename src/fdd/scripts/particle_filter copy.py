# particle filter

import numpy as np
import matplotlib.pyplot as plt
from uav_model import UAVModel
import matplotlib.pyplot as plt
import timeit
from fdd.scripts.mpc_without_fault import UAV_MPC

np.set_printoptions(precision=2)
def estimate(particles, weights):
    mean = np.average(particles, weights=weights, axis=0)
    var = np.average((particles - mean) ** 2, weights=weights, axis=0)
    return mean, var

def simple_resample(particles, weights):
    global N
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.  # 避免摄入误差
    rn = np.random.rand(N)
    indexes = np.searchsorted(cumulative_sum, rn)
    # 根据索引采样
    particles[:,:] = particles[indexes,:]
    return particles

def mcl_resample(particles, weights):
    global w_fast, w_slow
    p = max(0.00, 1 - w_fast/w_slow)
    # p = min(0.05, p)
    global z, action, N, x_est_extra
    num_mcl = int(p*N)
    print('num_mcl: ', num_mcl)
    particles_result = particles.copy()
    part1 = z * np.random.normal(1,0.01,(num_mcl,13))
    # if x_est_extra is not None:
    part2 = np.tile(x_est_extra[-8:-4], (num_mcl,1)) * np.random.normal(1,0.1,(num_mcl,4))
    # else:
    # part2 = action*np.random.uniform(0,1,(num_mcl,4))
    part3 = np.random.uniform(0,1,(num_mcl,4))
    particles_result[:num_mcl,:] = np.concatenate((part1,part2,part3), axis=1)

    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.  # 避免摄入误差
    indexes = np.searchsorted(cumulative_sum, np.random.rand(N-num_mcl))
    # 根据索引采样
    particles_result[num_mcl:,:] = particles[indexes,:]
    return particles_result

class CircleTrajectory(object):
    def __init__(self, origin, radius=5, omega=0.5):
        self.radius = radius
        self.origin = origin
        self.omega = omega

    def step(self, t):
        x = self.radius*np.cos(self.omega*t) + self.origin[0]
        y = self.radius*np.sin(self.omega*t) + self.origin[1]
        z = self.origin[2]
        return np.array([x,y,z])

if __name__ == "__main__":
    model = UAVModel(dt=0.01)
    x_real = model.state
    k_real = np.ones(4)

    N = 200 # 粒子数
    x_real_extra = np.hstack((x_real, k_real))
    x_p_extra = x_real_extra * np.random.normal(1,0.01,(N,x_real_extra.shape[0])) # 初始化粒子分布
    x_p_extra[:,-4:] = np.random.uniform(0,1,(N,4))
    

    real_x_sigma = 0.01
    real_z_sigma = 0.02

    predict_x_sigma = 0.05
    predict_k_sigma = 0.05
    predict_z_sigma = 0.05
    
    alpha_fast = 0.05
    alpha_slow = 0.005
    w_slow = 0
    w_fast = 0


    x_est_extra = None

    mpc = UAV_MPC(dt=0.02)
    traj = CircleTrajectory([-5,0,3], radius=5, omega=0.1)
    mpc.yref[:mpc.nx] = x_real
    mpc.yref_e = x_real

    start = timeit.default_timer()
    t = 0
    times = 600
    k_est = []
    k_real_list = []
    x_est = []
    x_real_list = []
    k_p = np.zeros((times, N, 4))
    x_p = np.zeros((times, N, 17))
    for i in range(times):
        t += model.dt
        
        noise_x_real = np.ones(17) + np.random.normal(0,real_x_sigma,17)
        noise_z_real = np.random.normal(0,real_z_sigma,13)
        k_real = np.ones(4) * 0.9
        if(i>200 and i < 300):
            k_real[0] = 0.05
        if(i>400 and i < 500):
            k_real[1] = 0.05

        # if(i>600 and i<700):
        #     k_real[2] = 0.05
        # if(i>800):
        #     k_real[3] = 0.05
        # if(i>500):
        #     k_real[0] = 0.1
        # action = 1.75*np.ones(4)

        # MPC
        if(i<=200):
            p = np.concatenate((noise_x_real, k_real))
            for j in range(mpc.N):
                pos = traj.step(t+j*mpc.dt)
                mpc.yref[0:3] = pos
                mpc.solver.set(j, "yref", mpc.yref)
                mpc.solver.set(j, "p", p)
            pos = traj.step(t+mpc.N*mpc.dt)
            mpc.yref_e[0:3] = pos
            mpc.solver.set(mpc.N, "p", p)
            mpc.solver.set(mpc.N, "yref", mpc.yref_e)
            mpc.solver.solve_for_x0(x_real)
            action = mpc.solver.get(0, "u")

        if(i>200):
            action = 1.75*np.ones(4)
        # if(i>500 and i<600):
        #     action = 1.75*np.ones(4)

        print("action:", action)
        x_real = model.step(x_real, action, noise_x_real, k_real)
        z = model.get_obs(noise_z_real)
        print('x_real:', x_real)

        weights = []
        norms = []
        # fig, ax = plt.subplots(2,1)
        for j in range(N):
            x_p_extra[j,-4:] = x_p_extra[j,-4:] + np.random.normal(0,predict_k_sigma,4)
            x_p_extra[j,-4:] = np.clip(x_p_extra[j,-4:], 0, 1)
            noise_x_predict = np.ones(17) + np.random.normal(0,predict_x_sigma,17)
            x_p_extra[j,:-4] = model.step(x_p_extra[j,:-4], action, noise_x_predict, x_p_extra[j,-4:])
            weights.append(1/np.sqrt((2*np.pi*predict_z_sigma**2)**13)*np.exp(-np.linalg.norm(z-x_p_extra[j,:13])**2/(2*predict_z_sigma**2)))
            
            # norms.append(np.linalg.norm(x_real-x_p_extra[j,:17]))
        
        k_p[i,:,:] = x_p_extra[:,-4:].copy()
        x_p[i,:,:] = x_p_extra[:,:17].copy()
        # ax[0].plot(norms)
        # ax[1].plot(weights)
        # plt.show()
        # print("x_p:", x_p_extra[0,:-4])
        # print("norm:", np.linalg.norm(z-x_p_extra[i,:13]))
        weights = np.array(weights)
        weights_average = np.average(weights)
        w_fast += alpha_fast * (weights_average - w_fast)
        w_slow += alpha_slow * (weights_average - w_slow)

        if np.sum(weights) == 0:
            weights = np.ones(N)
        weights /= np.sum(weights)
        
        # print("x_real", x_real)
        x_est_extra, x_est_var_extra = estimate(x_p_extra, weights)
        x_p_extra = simple_resample(x_p_extra, weights)
        print("weight_slow", w_slow)
        print("weight_fast", w_fast)
        # print("x_est:", x_est_extra[:-4])
        # print(i, "norm:", np.linalg.norm(x_real-x_est_extra[:-4]))


        # print("w_fast: ", w_fast, "w_slow: ", w_slow)
        k_real_list.append(k_real)
        if(i>0):
            k_est.append(0.9*x_est_extra[-4:].copy()+0.1*k_est[-1])
        else:
            k_est.append(x_est_extra[-4:].copy())

        x_real_list.append(x_real)
        x_est.append(x_est_extra[:-4])
        
        print(i, "k_est: ", k_est[-1])
    print("time: ", timeit.default_timer()-start)

    x_real_list = np.array(x_real_list)
    x_est = np.array(x_est)
    k_real_list = np.array(k_real_list)
    k_est = np.array(k_est)
    fig1, ax1 = plt.subplots(4,1)
    for i in range(4):
        for j in range(N):
            ax1[i].plot(k_p[:,j,i], color='gray', linewidth=0.5)
        ax1[i].plot(k_real_list[:,i], color='red', linewidth=2)
        ax1[i].plot(k_est[:,i], color='green', linewidth=2)
        ax1[i].set_ylim(0,1)
        ax1[i].set_title('k'+str(i))

    fig2, ax2 = plt.subplots(17,1)
    for i in range(17):
        for j in range(N):
            ax2[i].plot(x_p[:,j,i], color='gray', linewidth=0.5)
        ax2[i].plot(x_real_list[:,i], color='red', linewidth=2)
        ax2[i].plot(x_est[:,i], color='green', linewidth=2)
        # ax2[i].set_ylim(0,1)

    plt.show()