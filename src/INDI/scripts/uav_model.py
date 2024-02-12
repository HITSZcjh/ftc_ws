from acados_template import AcadosModel, AcadosSimSolver, AcadosSim, AcadosOcp, AcadosOcpSolver
import casadi as ca
import numpy as np
import os
import timeit
class UAVModel(object):
    def __init__(self, dt:float=0.01) -> None:
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
        km = 0.016  # rotor_moment_constant
        body_length = 0.17
        mass = 0.716
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
                                     [km, -km, km, -km],
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
        f_drag = -0.3*v
        tau_drag = -0.005*w
        # 系统动力学
        # 添加噪声与失效系数
        noise = ca.SX.sym("noise", state.size()[0], 1)
        k = ca.SX.sym("k", f_target.size()[0], 1)
        f_expl = ca.vertcat(
            v,
            G+1/mass*(R@F+f_drag),
            q_dot,
            np.linalg.inv(inertia)@(tau+tau_drag-ca.cross(w, inertia@w)),
            1/rotor_time_constant_down*(k*f_target-f_real)
        )*noise

        self.state_dot = ca.Function("state_dot", [state, f_target, noise, k], [f_expl])

        model = AcadosModel()
        x_dot = ca.SX.sym('x_dot', state.size()[0], 1)
        model.x = state
        model.f_expl_expr = f_expl
        model.xdot = x_dot
        model.u = f_target
        model.p = ca.vertcat(noise, k)
        model.name = "UAVModel"

        self.dt = dt
        sim = AcadosSim()
        sim.model = model
        sim.parameter_values = np.ones([model.p.size()[0], 1])
        sim.solver_options.T = self.dt
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
        self.state = np.array([0,0,3,
                               0,0,0,
                               1,0,0,0,
                               0,0,0,
                               1.75,1.75,1.75,1.75])
        self.action = np.zeros(4)
        self.state_noise = np.ones(17)
        self.k = np.ones(4)
        self.obs_noise = np.ones(13)
        self.integrator.set("x", self.state)

    def step(self, action:np.ndarray, state_noise:np.ndarray=None, k:np.ndarray=None, state:np.ndarray=None):
        if state_noise is not None:
            self.state_noise = state_noise
        if k is not None:
            self.k = k
        if state is not None:
            self.state = state

        self.action = np.clip(action, self.action_range[0], self.action_range[1]) 
        self.integrator.set("x", self.state)
        self.integrator.set("u", self.action)
        self.integrator.set("p", np.concatenate([self.state_noise, self.k]))
        self.integrator.solve()
        self.state = self.integrator.get("x")

        # self.state[3:6] = np.clip(self.state[3:6], -5, 5)
        # self.state[10:13] = np.clip(self.state[10:13], -5, 5)

        self.state[6:10] = self.state[6:10]/np.linalg.norm(self.state[6:10])
        return self.state
    

    def get_obs(self, obs_noise:np.ndarray=None):
        if obs_noise is not None:
            self.obs_noise = obs_noise
        obs = self.state[:13] * self.obs_noise
        obs[6:10] = obs[6:10]/np.linalg.norm(obs[6:10])

        R = np.array(self.R(self.state))
        state_dot = np.array(self.state_dot(self.state, self.action, self.state_noise, self.k))
        acc_I = state_dot[3:6].reshape(-1, 1) + np.array([0, 0, 9.81]).reshape(-1, 1)
        acc_B = R @ acc_I
        # acc_B 为模拟加速度计测量值
        return obs, R, acc_B

if __name__ == "__main__":
    model = UAVModel()
    action = np.ones(4)
    time_now = timeit.default_timer()
    for i in range(1):
        model.step(model.state, action)
    print(model.state)
    print(np.array(model.state_dot(model.state,action,model.noise,model.k)))
    end = timeit.default_timer()
    print(end-time_now)