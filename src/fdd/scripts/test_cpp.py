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
    model1 = SimpleUAVModel(ts, name="model1")
    mpc1 = UAV_MPC_Without_Fault(dt=0.05)
    pf1 = ParticleFilterCPP(250, 25, ts, log=True)
    pf1.set_init_state(model.get_obs()[0])
    # pf2 = ParticleFilter(250, model1, ts, log=True)
    traj = CircleTrajectory([0,0,2], 3, 1)

    mpc2 = UAV_MPC(dt=0.05)
    model.k = np.array([0.9, 0.9, 0.9, 0.9])

    t = 0
    fault = False
    k_est = None
    for i in range(4000):
        start_time = time.perf_counter()
        print("**** step: ", i)
        t += ts
        if i==2000:
            model.k = np.array([0.9, 0.9, 0.9, 0.05])
        
        if k_est is not None:
            if k_est[-1]<0.1:
                print(i)
                fault = True

        if not fault:
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
            f_target[np.where(f_target<0.2)] = 0
            if k_est is not None:
                f_target = f_target/k_est
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
            f_target = (f_target)/k_est

        model.step(f_target)
        obs = model.get_obs()[0]

        k_est = pf1.loop(f_target, obs)[1]
        end_time = time.perf_counter()
        print("time: ", end_time-start_time)
        # k_est = pf2.loop(f_target, obs)[-4:]
    pf1.log_show()
    # pf2.log_show()
    model.log_show()