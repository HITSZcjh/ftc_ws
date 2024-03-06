from particle_filter_cpp import ParticleFilterCPP
from particle_filter import ParticleFilter
import sys
sys.path.append("/home/jiao/ftc_ws")
from src.INDI.scripts.rotors_model import RotorsUAVModel
from mpc_without_fault import UAV_MPC_Without_Fault
from src.traj.script.traj import CircleTrajectory
from src.INDI.scripts.mpc import UAV_MPC
import numpy as np
import matplotlib.pyplot as plt
import time
import rospy
if __name__ == "__main__":

    rospy.init_node("UAV_MPC_node", anonymous=True)
    ts = 0.005
    rate = rospy.Rate(1/ts)
    model = RotorsUAVModel(ts, delay_time=0.03, log=True)
    mpc1 = UAV_MPC_Without_Fault(dt=0.05)
    pf1 = ParticleFilterCPP(350, 25, ts, log=True)

    obs, R, acc_B, f_real = model.get_obs()
    pf1.set_init_state(obs)
    traj = CircleTrajectory([0,0,3], 3, 1.5)

    mpc2 = UAV_MPC(dt=0.05)
    model.k = np.array([1, 1, 1, 1])

    t = 0
    fault = False
    k_est = None
    f_target = np.zeros(4)
    i = 0
    for i in range(2300):
        # i+=1
        start_time = time.perf_counter()
        print("**** step: ", i)
        obs, R, acc_B, f_real = model.get_obs()
        k_est = pf1.loop(f_target, obs)[1]
        t += ts
        if i==2000:
            model.k = np.array([1, 1, 1, 0.01])

        if k_est[-1]<0.2:
            print(i)
            fault = True

        if not fault:
            for j in range(mpc1.N):
                pos = traj.step(t+j*mpc1.dt, 100000)
                mpc1.yref[0:3] = pos
                mpc1.solver.set(j, "yref", mpc1.yref)
            pos = traj.step(t+mpc1.N*mpc1.dt, 100000)
            mpc1.yref_e[0:3] = pos
            mpc1.solver.set(mpc1.N, "yref", mpc1.yref_e)

            u = mpc1.solver.solve_for_x0(np.hstack((obs, f_real)))
            f_target = f_real+u*0.03
            f_target[np.where(f_target<0.2)] = 0
            # if k_est is not None:
            #     f_target = f_target/k_est
        else:
            for j in range(mpc2.N):
                pos = traj.step(t+j*mpc2.dt, 100000)
                mpc2.yref[0:3] = pos
                mpc2.solver.set(j, "yref", mpc2.yref)
            pos = traj.step(t+mpc2.N*mpc2.dt, 100000)
            mpc2.yref_e[0:3] = pos
            mpc2.solver.set(mpc2.N, "yref", mpc2.yref_e)
            mpc2.solver.solve_for_x0(np.hstack((obs, f_real)))
            u = mpc2.solver.get(0, "u")
            
            f_target = f_real+u*0.03
            f_target[np.where(f_target<0.2)] = 0
            # f_target = (f_target)/k_est

        model.step(f_target,traj.step(t,100000))
        rate.sleep()
        end_time = time.perf_counter()
        print("time: ", end_time-start_time)
        # k_est = pf2.loop(f_target, obs)[-4:]
    pf1.log_show()
    model.log_show()
    plt.show()
    # pf2.log_show()
    # model.log_show()