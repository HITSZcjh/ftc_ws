from particle_filter_cpp import ParticleFilterCPP
from particle_filter import ParticleFilter
import sys
sys.path.append("/home/jiao/ftc_ws")
from src.INDI.scripts.uav_model import SimpleUAVModel
from mpc_without_fault import UAV_MPC_Without_Fault
from src.traj.script.traj import CircleTrajectory
from src.INDI.scripts.mpc import UAV_MPC
import numpy as np
import matplotlib.pyplot as plt
import time
if __name__ == "__main__":
    ts = 0.005
    model = SimpleUAVModel(ts, log=True)
    mpc1 = UAV_MPC_Without_Fault(dt=0.05)
    pf1 = ParticleFilterCPP(1000, 25, ts, log=True)
    pf1.set_init_state(model.get_obs()[0])
    traj = CircleTrajectory([-3,0,3], 3, 1)

    mpc2 = UAV_MPC(dt=0.05)
    model.k = np.array([0.9, 0.9, 0.9, 0.9])

    t = 0
    fault = False
    k_est = None
    real_k_list = []
    for i in range(2000):
        print("**** step: ", i)
        t += ts
        if i<1000:
            model.k = np.array([0.9, 0.9, 0.9, 0.9])
        elif i<2000:
            model.k = np.array([0.9, 0.9, 0.9, 0.02])
        elif i<3000:
            model.k = np.array([0.9, 0.3, 0.9, 0.9])
        elif i<4000:
            model.k = np.array([0.9, 0.9, 0.5, 0.9])
        elif i<5000:
            model.k = np.array([0.9, 0.9, 0.9, 0.3])
        elif i<6000:
            model.k = np.array([0.9, 0.9, 0.9, 0.01])
        real_k_list.append(model.k.copy())
        if i<1000:
            for j in range(mpc1.N):
                pos = traj.step(t+j*mpc1.dt, i)
                mpc1.yref[0:3] = pos
                mpc1.solver.set(j, "yref", mpc1.yref)
            pos = traj.step(t+mpc1.N*mpc1.dt, i)
            mpc1.yref_e[0:3] = pos
            mpc1.solver.set(mpc1.N, "yref", mpc1.yref_e)
            mpc1.solver.solve_for_x0(model.state)
            u = mpc1.solver.get(0, "u")

            f_target = model.state[-4:]+u*0.03
            f_target = f_target/model.k
        else:
            for j in range(mpc2.N):
                pos = traj.step(t+j*mpc2.dt, i)
                mpc2.yref[0:3] = pos
                mpc2.solver.set(j, "yref", mpc2.yref)
            pos = traj.step(t+mpc2.N*mpc2.dt, i)
            mpc2.yref_e[0:3] = pos
            mpc2.solver.set(mpc2.N, "yref", mpc2.yref_e)
            mpc2.solver.solve_for_x0(model.state)
            u = mpc2.solver.get(0, "u")
            
            f_target = model.state[-4:]+u*0.03
            f_target[np.where(f_target<0.2)] = 0
            f_target = (f_target)/model.k

        model.step(f_target)
        obs = model.get_obs()[0]
        start_time = time.perf_counter()
        k_est = pf1.loop(f_target, obs)[1]
        end_time = time.perf_counter()
        print("time: ", end_time-start_time)
        # k_est = pf2.loop(f_target, obs)[-4:]
    pf1.log_show()
    real_k_list = np.array(real_k_list)
    t = np.arange(0, len(pf1.log_state_est_list)*pf1.ts, pf1.ts)
    pf1.fig, pf1.axs = plt.subplots(2,2)
    pf1.axs[0,0].plot(t, pf1.log_k_est_list[:,0], label="k1_est")
    pf1.axs[0,0].plot(t, real_k_list[:,0], label="k1")
    pf1.axs[0,0].set_ylim(0,1)
    pf1.axs[0,0].legend()
    pf1.axs[0,1].plot(t, pf1.log_k_est_list[:,1], label="k2_est")
    pf1.axs[0,1].plot(t, real_k_list[:,1], label="k2")
    pf1.axs[0,1].set_ylim(0,1)
    pf1.axs[0,1].legend()
    pf1.axs[1,0].plot(t, pf1.log_k_est_list[:,2], label="k3_est")
    pf1.axs[1,0].plot(t, real_k_list[:,2], label="k3")
    pf1.axs[1,0].set_ylim(0,1)
    pf1.axs[1,0].legend()
    pf1.axs[1,1].plot(t, pf1.log_k_est_list[:,3], label="k4_est")
    pf1.axs[1,1].plot(t, real_k_list[:,3], label="k4")
    pf1.axs[1,1].set_ylim(0,1)
    pf1.axs[1,1].legend()
    plt.show()

    # pf2.log_show()
    # model.log_show()