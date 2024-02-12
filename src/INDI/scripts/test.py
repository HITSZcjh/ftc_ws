from uav_model import UAVModel

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
BW = 50

class LPF(object):
    def __init__(self, ts, cutoff_freq, data):
        self.ts = ts
        self.cutoff_freq = cutoff_freq
        if isinstance(data, np.ndarray):
            self.last_output = np.zeros_like(data)
        else:
            self.last_output = 0.0
    def calc(self, input):
        output = (self.cutoff_freq * self.ts * input + self.last_output) / (self.cutoff_freq * self.ts + 1)
        self.last_output = output
        return output
    def calc_with_derivative(self, input):
        output = (self.cutoff_freq * self.ts * input + self.last_output) / (self.cutoff_freq * self.ts + 1)
        derivative = (output - self.last_output) / self.ts
        self.last_output = output
        return output, derivative

class PositionController(object):
    def __init__(self, ts) -> None:
        self.ts = ts

        self.kp_pos = np.array([[1],[1],[10]])
        self.kp_vel = np.array([[2],[2],[6]])
        self.ki_vel = np.array([[0],[0],[0]])

        self.int_lim = 5.0
        self.max_vel = 10.0
        self.max_angle = 10./57.3
        self.max_lateral = abs(9.81*np.tan(self.max_angle))

        self.integrals = np.zeros((3,1))
        self.acc_I_des = np.zeros((3,1))

        self.g = np.array([[0],[0],[-9.81]])
        self.n_des_I = np.zeros((3,1))
        self.n_des_I_dot = np.zeros((3,1))

        self.n_des_I_lpf = LPF(self.ts, BW, self.n_des_I)

    def calc(self, pos_target, pos_real, vel_real):
        pos_err = pos_target - pos_real
        vel_target = self.kp_pos * pos_err
        vel_target = np.clip(vel_target, -self.max_vel, self.max_vel)
        vel_err = vel_target - vel_real
        self.integrals += vel_err * self.ts
        self.integrals = np.clip(self.integrals, -self.int_lim, self.int_lim)
        self.acc_I_des = self.kp_vel * vel_err + self.ki_vel * self.integrals
        
        lat_ratio = np.linalg.norm(self.acc_I_des[:2,0])/self.max_lateral
        if lat_ratio > 1:
            self.acc_I_des[:2,0] /= lat_ratio
        
        self.acc_I_des[2,0] = np.clip(self.acc_I_des[2,0], -9.81, 9.81)
        self.n_des_I = (self.acc_I_des-self.g)/np.linalg.norm(self.acc_I_des-self.g)
        _, self.n_des_I_dot = self.n_des_I_lpf.calc_with_derivative(self.n_des_I)
        return self.acc_I_des, self.n_des_I, self.n_des_I_dot
    
class PrimaryAxisAttitudeController(object):
    def __init__(self, ts) -> None:
        self.ts = ts
        self.g = np.array([[0],[0],[-9.81]])

        self.n_B = np.array([[0.0],[0.0],[1.00]])
        self.kx = 0.02
        self.ky = 0.02

        self.p_lpf = LPF(self.ts, BW, 0.0)
        self.q_lpf = LPF(self.ts, BW, 0.0)

        self.n_des_B = None
    def calc(self, R, r, acc_I_des, n_des_I, n_des_I_dot):
        self.n_des_B = R@n_des_I
        h1 = self.n_des_B[0,0]
        h2 = self.n_des_B[1,0]
        h3 = self.n_des_B[2,0]
        n_B_x = self.n_B[0,0]
        n_B_y = self.n_B[1,0]
        n_B_z = self.n_B[2,0]
        vout = np.array([[self.kx*(n_B_x-h1)],[self.ky*(n_B_y-h2)]])
        temp = np.array([[0,1/h3],[-1/h3,0]])
        n_des_I_hat_dot = (R@n_des_I_dot)[:2,0]
        temp1 = temp@(vout-r*np.array([[h2],[-h1]])-n_des_I_hat_dot)
        p_des = temp1[0,0]
        q_des = temp1[1,0]
        _, p_des_dot = self.p_lpf.calc_with_derivative(p_des)
        _, q_des_dot = self.q_lpf.calc_with_derivative(q_des)
        # acc_z_des = (R@acc_I_des)[2,0]
        acc_z_des = np.linalg.norm(acc_I_des-self.g)/n_B_z
        return p_des, q_des, acc_z_des, p_des_dot, q_des_dot

class INDIController(object):
    def __init__(self, ts) -> None:
        self.ts = ts
        self.tau_x = 0.0
        self.tau_y = 0.0
        self.f_z = 0.0

        self.Ix = 0.007
        self.Iy = 0.007
        self.m = 0.716

        self.tau_x_lpf = LPF(self.ts, 50, 0.0)
        self.tau_y_lpf = LPF(self.ts, 50, 0.0)
        self.f_z_lpf = LPF(self.ts, 50, 0.0)
        self.p_lpf = LPF(self.ts, 50, 0.0)
        self.q_lpf = LPF(self.ts, 50, 0.0)
        self.acc_z_lpf = LPF(self.ts, 50, 0.0)
        self.acc_z_des_lpf = LPF(self.ts, 50, 0.0)

        self.k1 = 30
        self.k2 = 30
        self.k3 = 30

        self.integrals = 0.0

    def calc(self, p, p_des, p_des_dot, tau_x, q, q_des, q_des_dot, tau_y, acc_z, acc_z_des, f_z):
        self.integrals += (acc_z_des - acc_z) * self.ts
        self.integrals = np.clip(self.integrals, -0.5, 0.5)
        _, p_dot = self.p_lpf.calc_with_derivative(p)
        _, q_dot = self.q_lpf.calc_with_derivative(q)
        self.tau_x = self.tau_x_lpf.calc(tau_x)+self.Ix*(-p_dot+p_des_dot+self.k1*(p_des-p))
        self.tau_y = self.tau_y_lpf.calc(tau_y)+self.Iy*(-q_dot+q_des_dot+self.k2*(q_des-q))
        self.f_z = self.f_z_lpf.calc(f_z)+self.m*(-self.acc_z_lpf.calc(acc_z)+self.acc_z_des_lpf.calc(acc_z_des)
                                                  +self.k3*self.integrals)
        return self.tau_x, self.tau_y, self.f_z


if __name__ == "__main__":
    ts = 0.002
    model = UAVModel(ts)
    pos_controller = PositionController(ts)
    pat_controller = PrimaryAxisAttitudeController(ts)
    indi_controller = INDIController(ts)

    pos_target = np.array([[0.1],[0],[3]])
    tau_x = 0
    tau_y = 0
    f_z = 0

    state_list = []
    obs_list = []
    acc_list = []
    acc_I_des_list = []
    n_des_B_list = []
    show_list1 = []
    show_list2 = []
    show_list3 = []
    R_list = []
    for i in range(1000):
        if i==87:
            print(1)
        print("****** ",i," ******")
        state_list.append(model.state.copy())
        obs, R, acc_B = model.get_obs()
        obs_list.append(obs.copy())
        acc_list.append(acc_B[2,0])
        R_list.append(R.copy())
        pos_real = obs[:3].reshape(-1,1)
        vel_real = obs[3:6].reshape(-1,1)
        acc_I_des, n_des_I, n_des_I_dot = pos_controller.calc(pos_target, pos_real, vel_real)
        print("acc_I_des: ", acc_I_des)
        print("n_des_I: ", n_des_I)
        print("n_des_I_dot: ", n_des_I_dot)
        acc_I_des_list.append(acc_I_des)

        p_des, q_des, acc_z_des, p_des_dot, q_des_dot = pat_controller.calc(R, obs[12], acc_I_des, n_des_I, n_des_I_dot)
        print("p_des: ", p_des)
        print("q_des: ", q_des)
        print("acc_z_des: ", acc_z_des)
        print("p_des_dot: ", p_des_dot)
        print("q_des_dot: ", q_des_dot)
        show_list2.append([p_des, q_des, acc_z_des, p_des_dot, q_des_dot])
        
        tau_x, tau_y, f_z = indi_controller.calc(obs[10], p_des, p_des_dot, tau_x, 
                                                 obs[11], q_des, q_des_dot, tau_y, 
                                                 acc_B[2,0], acc_z_des, f_z)
        print("tau_x: ", tau_x)
        print("tau_y: ", tau_y)
        print("f_z: ", f_z)
        show_list3.append([tau_x, tau_y, f_z])
        
        n_des_B_list.append(pat_controller.n_des_B)

        temp = np.linalg.inv(model.AllocationMatrix_failed)@np.array([[tau_x],[tau_y],[f_z]])
        action = np.array([temp[0,0], temp[1,0], temp[2,0], 0])

        model.step(action)
        temp = model.AllocationMatrix@model.action.reshape(-1,1)
        tau_x = temp[0,0]
        tau_y = temp[1,0]
        f_z = temp[3,0]

    t = np.linspace(0,ts*len(obs_list),len(obs_list))

    obs_list = np.array(obs_list)
    fig,axs = plt.subplots(obs_list.shape[1],1)
    fig.suptitle('obs')
    for i in range(obs_list.shape[1]):
        axs[i].plot(t, obs_list[:,i])


    show_list2 = np.array(show_list2)
    fig2,axs2 = plt.subplots(show_list2.shape[1],1)
    for i in range(show_list2.shape[1]):
        axs2[i].plot(t, show_list2[:,i])
    axs2[0].plot(t, obs_list[:,10])
    axs2[1].plot(t, obs_list[:,11])
    axs2[2].plot(t, acc_list)

    n_des_B_list = np.array(n_des_B_list)
    fig3,axs3 = plt.subplots(n_des_B_list.shape[1],1)
    fig3.suptitle('h')
    for i in range(n_des_B_list.shape[1]):
        axs3[i].plot(t, n_des_B_list[:,i])


    state_list = np.array(state_list)
    fig4,axs4 = plt.subplots(state_list.shape[1],1)
    fig4.suptitle('state')
    for i in range(state_list.shape[1]):
        axs4[i].plot(t, state_list[:,i])

    acc_I_des_list = np.array(acc_I_des_list)
    fig5,axs5 = plt.subplots(acc_I_des_list.shape[1],1)
    fig5.suptitle('acc_I_des')
    for i in range(acc_I_des_list.shape[1]):
        axs5[i].plot(t, acc_I_des_list[:,i])

    R_list = np.array(R_list)
    fig6 = plt.figure()
    ax6 = fig6.add_subplot(111, projection='3d')
    ax6.set_xlim(-1, 1)
    ax6.set_ylim(-1, 1)
    ax6.set_zlim(2.5, 3.5)
    arrow_length = 0.5
    x_arrow = ax6.quiver(state_list[0,0],state_list[0,1],state_list[0,2],
                          arrow_length * R_list[0, 0, 0], arrow_length * R_list[0, 1, 0], arrow_length * R_list[0, 2, 0], 
                          color='r', label='X')
    y_arrow = ax6.quiver(state_list[0,0],state_list[0,1],state_list[0,2],
                            arrow_length * R_list[0, 0, 1], arrow_length * R_list[0, 1, 1], arrow_length * R_list[0, 2, 1], 
                            color='g', label='Y')
    z_arrow = ax6.quiver(state_list[0,0],state_list[0,1],state_list[0,2],
                            arrow_length * R_list[0, 0, 2], arrow_length * R_list[0, 1, 2], arrow_length * R_list[0, 2, 2], 
                            color='b', label='Z')

    def update(frame):
        global x_arrow, y_arrow, z_arrow, ax6
        # 旋转坐标轴
        x_arrow.remove()
        y_arrow.remove()
        z_arrow.remove()
        
        x_arrow = ax6.quiver(state_list[frame,0],state_list[frame,1],state_list[frame,2],
                             arrow_length * R_list[frame, 0, 0], arrow_length * R_list[frame, 1, 0], arrow_length * R_list[frame, 2, 0], 
                            color='r', label='X')
        y_arrow = ax6.quiver(state_list[frame,0],state_list[frame,1],state_list[frame,2],
                                arrow_length * R_list[frame, 0, 1], arrow_length * R_list[frame, 1, 1], arrow_length * R_list[frame, 2, 1],
                                color='g', label='Y')
        z_arrow = ax6.quiver(state_list[frame,0],state_list[frame,1],state_list[frame,2],
                                arrow_length * R_list[frame, 0, 2], arrow_length * R_list[frame, 1, 2], arrow_length * R_list[frame, 2, 2], 
                                color='b', label='Z')

    ani = FuncAnimation(fig6, update, frames=R_list.shape[0], interval=5, repeat=True)
    plt.show()