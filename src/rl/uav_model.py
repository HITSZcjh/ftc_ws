from acados_template import AcadosModel, AcadosSimSolver, AcadosSim, AcadosOcp, AcadosOcpSolver
import casadi as ca
import numpy as np
import os
import timeit


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

class SimpleUAVModel(object):
    def __init__(self, ts:float=0.01, delay_time:float=None, log:bool=False, name="UAVModel", BW=5) -> None:
        # 系统状态
        p = ca.SX.sym("p", 3, 1)
        v = ca.SX.sym("v", 3, 1)
        q = ca.SX.sym("q", 4, 1)
        w = ca.SX.sym("w", 3, 1)

        f_real = ca.SX.sym("f_real", 4, 1)
        # 系统状态集合
        state = ca.vertcat(p, v, q, w, f_real)

        # 控制输入
        f_target = ca.SX.sym("f_target", 4, 1)

        # 参数
        rotor_time_constant_up = 0.0125
        rotor_time_constant_down = 0.025
        Kf = 8.54858e-06  # rotot_motor_constant
        rotor_drag_coeff = 0.016  # rotor_moment_constant
        body_length = 0.17
        self.mass = 0.73
        g = 9.81
        inertia = np.array([[0.007, 0, 0], [0, 0.007, 0], [0, 0, 0.012]])
        R = ca.vertcat(
            ca.horzcat(
                1 - 2 * (q[2] ** 2 + q[3] ** 2),
                2 * (q[1] * q[2] - q[0] * q[3]),
                2 * (q[1] * q[3] + q[0] * q[2]),
            ),
            ca.horzcat(
                2 * (q[1] * q[2] + q[0] * q[3]),
                1 - 2 * (q[1] ** 2 + q[3] ** 2),
                2 * (q[2] * q[3] - q[0] * q[1]),
            ),
            ca.horzcat(
                2 * (q[1] * q[3] - q[0] * q[2]),
                2 * (q[2] * q[3] + q[0] * q[1]),
                1 - 2 * (q[1] ** 2 + q[2] ** 2),
            ),
        )
        self.R = ca.Function("R", [state], [R])
        self.AllocationMatrix = np.array([[0, body_length, 0, -body_length],
                                     [-body_length, 0, body_length, 0],
                                     [rotor_drag_coeff, -rotor_drag_coeff, rotor_drag_coeff, -rotor_drag_coeff],
                                     [1, 1, 1, 1]])
        self.AllocationMatrix_failed = np.array([[0, body_length, 0],
                                     [-body_length, 0, body_length],
                                     [1, 1, 1]])
        temp = self.AllocationMatrix@f_real
        F = ca.vertcat(np.zeros([2, 1]), temp[3])
        tau = temp[0:3]
        G = np.array([[0], [0], [-g]])
        q_dot = 1/2*ca.vertcat(-q[1]*w[0]-q[2]*w[1]-q[3]*w[2],
                               q[0]*w[0]+q[2]*w[2]-q[3]*w[1],
                               q[0]*w[1]-q[1]*w[2]+q[3]*w[0],
                               q[0]*w[2]+q[1]*w[1]-q[2]*w[0])
        # 空气阻力和空气阻力矩
        f_drag = -0.1*v
        tau_drag = -4e-4*ca.sign(w)*w**2
        # 系统动力学
        # 添加噪声与失效系数
        noise = ca.SX.sym("noise", state.size()[0], 1)
        k = ca.SX.sym("k", f_target.size()[0], 1)
        f_expl = ca.vertcat(
            v,
            G+1/self.mass*(R@F+f_drag),
            q_dot,
            np.linalg.inv(inertia)@(tau+tau_drag-ca.cross(w, inertia@w)),
            1/rotor_time_constant_down*(k*f_target-f_real)
        )+noise

        self.state_dot = ca.Function("state_dot", [state, f_target, noise, k], [f_expl])

        model = AcadosModel()
        x_dot = ca.SX.sym('x_dot', state.size()[0], 1)
        model.x = state
        model.f_expl_expr = f_expl
        model.xdot = x_dot
        model.u = f_target
        model.p = ca.vertcat(noise, k)
        model.name = name

        # 测试RK45积分和acados积分的一致性
        ode = ca.Function('ode',[state, f_target, noise, k], [f_expl])
        x = model.x
        u = model.u
        k1 = ode(x,       u, noise, k)
        k2 = ode(x+ts/2*k1, u, noise, k)
        k3 = ode(x+ts/2*k2, u, noise, k)
        k4 = ode(x+ts*k3,  u, noise, k)
        xf = x + ts/6 * (k1 + 2*k2 + 2*k3 + k4)
        self.test_fun = ca.Function('test_fun', [state, f_target, noise, k],[xf])

        self.ts = ts
        sim = AcadosSim()
        sim.model = model
        sim.parameter_values = np.ones([model.p.size()[0], 1])
        sim.solver_options.T = self.ts
        sim.solver_options.integrator_type = "ERK"
        sim.solver_options.num_stages = 3
        sim.solver_options.num_steps = 3
        sim.solver_options.newton_iter = 3  # for implicit integrator
        sim.solver_options.collocation_type = "GAUSS_RADAU_IIA"
        sim.code_export_directory = os.path.dirname(os.path.realpath(__file__))+"/c_generated_code/sim_"+model.name
        json_files_path = os.path.dirname(os.path.realpath(__file__))+"/json_files"
        if not (os.path.exists(json_files_path)):
            os.makedirs(json_files_path)
        json_file = json_files_path + "/sim_"+model.name+'_acados.json'
        self.integrator = AcadosSimSolver(sim, json_file=json_file)

        self.action_range = [0.0, 6.0]
        self.state = np.array([0,0,0,
                               0,0,0,
                               1,0,0,0,
                               0,0,0,
                               0,0,0,0])
        self.action = np.zeros(4)
        self.state_noise = np.zeros(17)
        self.k = np.ones(4)
        self.obs_noise = np.zeros(13)
        self.integrator.set("x", self.state)

        self.omega_lpf = LPF(self.ts, BW, np.zeros(3))

        self.delay_time = delay_time
        if self.delay_time is not None:
            num = int(delay_time/self.ts)
            self.delay_state_list = [self.state.copy() for _ in range(num)]
            
            obs = self.state[:13]
            R = np.array(self.R(self.state))
            state_dot = np.array(self.state_dot(self.state, self.action, self.state_noise, self.k))
            acc_I = state_dot[3:6].reshape(-1, 1) + np.array([0, 0, 9.81]).reshape(-1, 1)
            acc_B = R.T @ acc_I
            f_real = self.state[-4:]

            self.delay_obs_list = [obs.copy() for _ in range(num)]
            self.delay_R_list = [R.copy() for _ in range(num)]
            self.delay_acc_B_list = [acc_B.copy() for _ in range(num)]
            self.delay_f_real_list = [f_real.copy() for _ in range(num)]

            # RL
            acc = np.sum(self.state[-4:])/self.mass
            omega_dot_f = self.omega_lpf.calc_with_derivative(self.state[10:13])[1]
            self.delay_acc_list = [acc for _ in range(num)]
            self.delay_omega_dot_f_list = [omega_dot_f for _ in range(num)]

        self.log = log
        if self.log:
            self.log_state_list = []
            self.log_action_list = []



    def step(self, action:np.ndarray, state_noise:np.ndarray=None, k:np.ndarray=None, state:np.ndarray=None):
        if state_noise is not None:
            self.state_noise = state_noise
        if k is not None:
            self.k = k
        if state is not None:
            self.state = state

        self.action = np.clip(action, self.action_range[0], self.action_range[1]) 

        if self.log:
            self.log_state_list.append(self.state.copy())
            self.log_action_list.append(self.action.copy())

        self.integrator.set("x", self.state)
        self.integrator.set("u", self.action)
        self.integrator.set("p", np.concatenate([self.state_noise, self.k]))
        self.integrator.solve()
        self.state = self.integrator.get("x")

        # self.state[3:6] = np.clip(self.state[3:6], -5, 5)
        # self.state[10:13] = np.clip(self.state[10:13], -5, 5)

        self.state[6:10] = self.state[6:10]/np.linalg.norm(self.state[6:10])

        if self.delay_time is not None:
            self.delay_state_list.append(self.state.copy())
            self.delay_state_list.pop(0)
            return self.delay_state_list[0]
        else:
            return self.state
    
    def predict(self, state, action_list, ts, k):
        self.integrator.set("T", ts)
        for action in action_list:
            state = self.step(action, state=state, k=k)
        obs = state[:13]
        obs[6:10] = obs[6:10]/np.linalg.norm(obs[6:10])
        R = np.array(self.R(state))
        state_dot = np.array(self.state_dot(state, action_list[-1], self.state_noise, self.k))
        
        r_mi = np.array([0.005,0.005,0.005])

        acc_with_bias = R.T @ (state_dot[3:6].reshape(-1, 1) + np.array([0, 0, 9.81]).reshape(-1, 1)).reshape(-1) + np.cross(state_dot[10:13].reshape(-1),r_mi) + np.cross(state[10:13].reshape(-1),np.cross(state[10:13].reshape(-1),r_mi))

        acc_I = state_dot[3:6].reshape(-1, 1) + np.array([0, 0, 9.81]).reshape(-1, 1)
        acc_B = R.T @ acc_I
        f_real = state[-4:]
        return obs, R, acc_B, f_real, acc_with_bias

    def get_obs(self, obs_noise:np.ndarray=None):
        if obs_noise is not None:
            self.obs_noise = obs_noise
        obs = self.state[:13] + self.obs_noise
        obs[6:10] = obs[6:10]/np.linalg.norm(obs[6:10])

        R = np.array(self.R(self.state))
        state_dot = np.array(self.state_dot(self.state, self.action, self.state_noise, self.k))
        acc_I = state_dot[3:6].reshape(-1, 1) + np.array([0, 0, 9.81]).reshape(-1, 1)
        acc_B = R.T @ acc_I
        f_real = self.state[-4:]
        # acc_B 为模拟加速度计测量值

        if self.delay_time is not None:
            self.delay_obs_list.append(obs.copy())
            self.delay_obs_list.pop(0)
            self.delay_R_list.append(R.copy())
            self.delay_R_list.pop(0)
            self.delay_acc_B_list.append(acc_B.copy())
            self.delay_acc_B_list.pop(0)
            self.delay_f_real_list.append(f_real.copy())
            self.delay_f_real_list.pop(0)
            return self.delay_obs_list[0], self.delay_R_list[0], self.delay_acc_B_list[0], self.delay_f_real_list[0]
        else:
            return obs, R, acc_B, f_real
    
    # without noise
    def get_obs_rl(self):
        obs = self.state[:13]
        acc = np.sum(self.state[-4:])/self.mass
        omega_dot_f = self.omega_lpf.calc_with_derivative(self.state[10:13])[1]

        if self.delay_time is not None:
            self.delay_obs_list.append(obs.copy())
            self.delay_obs_list.pop(0)
            self.delay_acc_list.append(acc)
            self.delay_acc_list.pop(0)
            self.delay_omega_dot_f_list.append(omega_dot_f)
            self.delay_omega_dot_f_list.pop(0)
            return self.delay_obs_list[0], self.delay_acc_list[0], self.delay_omega_dot_f_list[0]
        else:
            return obs, acc, omega_dot_f
    
    def update(self, frame):
        # 旋转坐标轴
        self.x_arrow.remove()
        self.y_arrow.remove()
        self.z_arrow.remove()
        
        self.x_arrow = self.axs.quiver(self.log_state_list[frame,0],self.log_state_list[frame,1],self.log_state_list[frame,2],
                            self.arrow_length * self.log_R_list[frame, 0, 0], self.arrow_length * self.log_R_list[frame, 1, 0], self.arrow_length * self.log_R_list[frame, 2, 0], 
                            color='r', label='X')
        self.y_arrow = self.axs.quiver(self.log_state_list[frame,0],self.log_state_list[frame,1],self.log_state_list[frame,2],
                                self.arrow_length * self.log_R_list[frame, 0, 1], self.arrow_length * self.log_R_list[frame, 1, 1], self.arrow_length * self.log_R_list[frame, 2, 1],
                                color='g', label='Y')
        self.z_arrow = self.axs.quiver(self.log_state_list[frame,0],self.log_state_list[frame,1],self.log_state_list[frame,2],
                                self.arrow_length * self.log_R_list[frame, 0, 2], self.arrow_length * self.log_R_list[frame, 1, 2], self.arrow_length * self.log_R_list[frame, 2, 2], 
                                color='b', label='Z')

    def log_show(self):
        if self.log:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation, FFMpegWriter
            t = np.arange(0, len(self.log_state_list)*self.ts, self.ts)
            self.log_state_list = np.array(self.log_state_list)
            self.log_action_list = np.array(self.log_action_list)

            self.fig, self.axs = plt.subplots(2, 2)
            self.axs[0,0].plot(t, self.log_state_list[:, 0], label="px")
            self.axs[0,0].plot(t, self.log_state_list[:, 1], label="py")
            self.axs[0,0].plot(t, self.log_state_list[:, 2], label="pz")
            self.axs[0,0].legend()
            self.axs[0,1].plot(t, self.log_state_list[:, 3], label="vx")
            self.axs[0,1].plot(t, self.log_state_list[:, 4], label="vy")
            self.axs[0,1].plot(t, self.log_state_list[:, 5], label="vz")
            self.axs[0,1].legend()
            self.axs[1,0].plot(t, self.log_state_list[:, 6], label="w")
            self.axs[1,0].plot(t, self.log_state_list[:, 7], label="x")
            self.axs[1,0].plot(t, self.log_state_list[:, 8], label="y")
            self.axs[1,0].plot(t, self.log_state_list[:, 9], label="z")
            self.axs[1,0].legend()
            self.axs[1,1].plot(t, self.log_state_list[:, 10], label="wx")
            self.axs[1,1].plot(t, self.log_state_list[:, 11], label="wy")
            self.axs[1,1].plot(t, self.log_state_list[:, 12], label="wz")
            self.axs[1,1].legend()


            self.fig, self.axs = plt.subplots(2,2)
            self.axs[0,0].plot(t, self.log_action_list[:, 0], label="f1_target")
            self.axs[0,0].plot(t, self.log_state_list[:,13], label="f1_real")
            self.axs[0,0].set_ylim(0, 6)
            self.axs[0,0].legend()
            self.axs[0,1].plot(t, self.log_action_list[:, 1], label="f2_target")
            self.axs[0,1].plot(t, self.log_state_list[:,14], label="f2_real")
            self.axs[0,1].set_ylim(0, 6)
            self.axs[0,1].legend()
            self.axs[1,0].plot(t, self.log_action_list[:, 2], label="f3_target")
            self.axs[1,0].plot(t, self.log_state_list[:,15], label="f3_real")
            self.axs[1,0].set_ylim(0, 6)
            self.axs[1,0].legend()
            self.axs[1,1].plot(t, self.log_action_list[:, 3], label="f4_target")
            self.axs[1,1].plot(t, self.log_state_list[:,16], label="f4_real")
            self.axs[1,1].set_ylim(0, 6)
            self.axs[1,1].legend()


            self.fig = plt.figure()
            self.axs = self.fig.add_subplot(111, projection='3d')
            self.axs.set_xlim(-5,5)
            self.axs.set_ylim(-5,5)
            self.axs.set_zlim(0,6)
            self.arrow_length = 1.0
            self.log_R_list = []
            for i in range(self.log_state_list.shape[0]):
                w, x, y, z = self.log_state_list[i, 6:10]
                self.log_R_list.append(np.array([[1-2*(y**2+z**2), 2*(x*y-z*w), 2*(x*z+y*w)],
                                        [2*(x*y+z*w), 1-2*(x**2+z**2), 2*(y*z-x*w)],
                                        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x**2+y**2)]])
                            )
            self.log_R_list = np.array(self.log_R_list)
            self.x_arrow = self.axs.quiver(self.log_state_list[0,0],self.log_state_list[0,1],self.log_state_list[0,2],
                                self.arrow_length * self.log_R_list[0, 0, 0], self.arrow_length * self.log_R_list[0, 1, 0], self.arrow_length * self.log_R_list[0, 2, 0], 
                                color='r', label='X')
            self.y_arrow = self.axs.quiver(self.log_state_list[0,0],self.log_state_list[0,1],self.log_state_list[0,2],
                                    self.arrow_length * self.log_R_list[0, 0, 1], self.arrow_length * self.log_R_list[0, 1, 1], self.arrow_length * self.log_R_list[0, 2, 1], 
                                    color='g', label='Y')
            self.z_arrow = self.axs.quiver(self.log_state_list[0,0],self.log_state_list[0,1],self.log_state_list[0,2],
                                    self.arrow_length * self.log_R_list[0, 0, 2], self.arrow_length * self.log_R_list[0, 1, 2], self.arrow_length * self.log_R_list[0, 2, 2], 
                                    color='b', label='Z')
            
            ani = FuncAnimation(self.fig, self.update, 
                                frames=self.log_R_list.shape[0], interval=10, repeat=True)
            # writer = FFMpegWriter(fps=100, metadata=dict(artist='Me'), bitrate=1800)
            # path_prefix = os.path.dirname(os.path.realpath(__file__))+"/data/"
            # ani.save(path_prefix+"output.mp4", writer=writer)
            plt.show()


if __name__ == "__main__":
    model = SimpleUAVModel()
    action = np.ones(4)
    model.step(action)
    print(model.state)

    action = np.array([3.5,2.5,3.5,0]).reshape(-1, 1)
    print(model.AllocationMatrix@action)

    tau = np.array([0,0,0.112,7]).reshape(-1,1)
    print(np.linalg.inv(model.AllocationMatrix)@tau)

    # print(model1.state)
    # model12 = SimpleUAVModel(0.00005)
    # for i in range(200):
    #     model12.state = model12.test_fun(model12.state, action, model12.state_noise, model12.k)
    # print(model12.state)
