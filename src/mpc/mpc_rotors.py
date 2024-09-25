#!~/python_env/acados_env/bin/python
# coding=UTF-8
import rospy
import numpy as np
import casadi as ca
from acados_template import AcadosModel

import os

import sys
sys.path.append("/home/jiao/rl_quad_ws/ftc_ws")
import scipy.linalg

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

from casadi import *

from geometry_msgs.msg import PoseStamped
import time
from matplotlib import pyplot as plt

import sys

from src.rl.uav_model import SimpleUAVModel
from src.rl.rotors_model import RotorsUAVModel
# python强制打印整个数组
np.set_printoptions(threshold=sys.maxsize, precision=4, suppress=True)


class UAV_MPC(object):
    def __init__(self, dt=0.05):
        # 控制输入
        df = ca.SX.sym('df', 4, 1)
        controls = ca.vertcat(df)

        # 系统状态
        p = ca.SX.sym('p', 3, 1)
        v = ca.SX.sym('v', 3, 1)
        q = ca.SX.sym('q', 4, 1)
        w = ca.SX.sym('w', 3, 1)
        f = ca.SX.sym('f', 4, 1)

        # 系统状态集合
        states = ca.vertcat(p, v, q, w, f)

        self.nx = states.size()[0]
        self.nu = controls.size()[0]
        self.ny = self.nx + self.nu


        self.rotor_drag_coeff = 0.016  
        self.body_length = 0.17
        self.mass = 0.73
        self.g = 9.81
        self.inertia = np.array([[0.007, 0, 0], [0, 0.007, 0], [0, 0, 0.012]])

        R1 = ca.horzcat(1-2*(q[2]**2+q[3]**2), 2 *
                        (q[1]*q[2]-q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2]))
        R2 = ca.horzcat(2*(q[1]*q[2]+q[0]*q[3]), 1-2 *
                        (q[1]**2+q[3]**2), 2*(q[2]*q[3]-q[0]*q[1]))
        R3 = ca.horzcat(2*(q[1]*q[3]-q[0]*q[2]), 2 *
                        (q[2]*q[3]+q[0]*q[1]), 1-2*(q[1]**2+q[2]**2))
        # R = ca.vertcat(1-2*(q[2]**2+q[3]**2),2*(q[1]*q[2]-q[0]*q[3]),2*(q[1]*q[3]+q[0]*q[2]),
        #               2*(q[1]*q[2]+q[0]*q[3]),1-2*(q[1]**2+q[3]**2),2*(q[2]*q[3]-q[0]*q[1]),
        #               2*(q[1]*q[3]-q[0]*q[2]),2*(q[2]*q[3]+q[0]*q[1]),1-2*(q[1]**2+q[2]**2))
        R = ca.vertcat(R1, R2, R3)
        self.AllocationMatrix = np.array([[0, self.body_length, 0, -self.body_length],
                                     [-self.body_length, 0, self.body_length, 0],
                                     [self.rotor_drag_coeff, -self.rotor_drag_coeff, self.rotor_drag_coeff, -self.rotor_drag_coeff],
                                     [1, 1, 1, 1]])
        temp = self.AllocationMatrix@f
        total_thrust = ca.vertcat(np.zeros([2, 1]), temp[3])
        tau = temp[0:3]
        G = np.array([[0], [0], [-self.g]])
        q_dot = 1/2*ca.vertcat(-q[1]*w[0]-q[2]*w[1]-q[3]*w[2],
                               q[0]*w[0]+q[2]*w[2]-q[3]*w[1],
                               q[0]*w[1]-q[1]*w[2]+q[3]*w[0],
                               q[0]*w[2]+q[1]*w[1]-q[2]*w[0])

        # 系统微分方程
        f_expl = ca.vertcat(
            v,
            G+1/self.mass*R@total_thrust,
            q_dot,
            np.linalg.inv(self.inertia)@(tau-ca.cross(w, self.inertia@w)),
            df
        )
        x_dot = ca.SX.sym('x_dot', self.nx, 1)

        # ACADOS 模型类
        model = AcadosModel()
        model.u = controls
        model.x = states
        model.xdot = x_dot
        # 模型各个部分定义，从CasADi的表达式映射到ACADOS的模型中
        model.f_expl_expr = f_expl
        model.f_impl_expr = x_dot-f_expl
        model.p = []
        model.name = 'UAV'

        # set up RK4
        self.dt = dt
        self.N = 20
        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.N
        ocp.solver_options.tf = self.N*self.dt

        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        Q = np.zeros((self.nx, 1))
        Q[0:3] = 5
        Q[10:12] = 1
        Q = np.diagflat(Q)
        R = 1*0.05*np.eye(self.nu)

        ocp.cost.W = scipy.linalg.block_diag(Q, R)
        ocp.cost.W_e = Q
        ocp.cost.Vx = np.zeros((self.ny, self.nx))
        ocp.cost.Vx[:self.nx, :self.nx] = np.eye(self.nx)
        ocp.cost.Vx_e = np.eye(self.nx)
        ocp.cost.Vu = np.zeros((self.ny, self.nu))
        ocp.cost.Vu[-self.nu:, -self.nu:] = np.eye(self.nu)

        self.yref = np.zeros(self.ny)
        self.yref_e = np.zeros(self.nx)
        ocp.cost.yref = self.yref
        ocp.cost.yref_e = self.yref_e

        self.x0 = np.zeros((self.nx))
        self.x0[9] = 1
        ocp.constraints.x0 = self.x0

        
        ocp.constraints.idxbx = np.array([10, 11,  13, 14, 15, 16])
        ocp.constraints.lbx = np.array([-5, -5, 0, 0, 0, 0])
        ocp.constraints.ubx = np.array([5, 5,
                                        6, 
                                        6, 
                                        6, 
                                        0.1])

        ocp.constraints.idxbu = np.array([0, 1, 2, 3])
        ocp.constraints.lbu = np.array([-15*7, -15*7, -15*7, -15*7])
        ocp.constraints.ubu = np.array([25*7, 25*7, 25*7, 25*7])

        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'  # 'EXACT''GAUSS_NEWTON'
        ocp.solver_options.nlp_solver_ext_qp_res = 0
        ocp.solver_options.nlp_solver_max_iter = 100
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.qp_solver_cond_N = 5
        ocp.solver_options.integrator_type = 'ERK'  # 'DISCRETE' 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # SQP_RTI, SQP

        ocp.code_export_directory = os.path.dirname(
            os.path.realpath(__file__))+"/c_generated_code/mpc_"+model.name
        json_files_path = os.path.dirname(os.path.realpath(__file__))+"/json_files"
        if not (os.path.exists(json_files_path)):
            os.makedirs(json_files_path)
        json_file = json_files_path + "/mpc_"+model.name+'_acados.json'
        self.solver = AcadosOcpSolver(ocp, json_file=json_file)


if __name__ == '__main__':  
    rospy.init_node("UAV_RL_node", anonymous=None)
  
    ts = 0.0025
    rate = rospy.Rate(1/ts)

    model = RotorsUAVModel(ts=ts, delay_time=0.03 ,log=True, BW=1/0.12)
    controller = UAV_MPC(dt=0.05)
    controller.yref[0:3] = np.array([0, 0, 3])
    controller.yref_e[0:3] = np.array([0, 0, 3])

    du = np.zeros(4)
    u = 0*np.ones(4)
    last_f_target = np.zeros(4)
    last_f_target_list = []
    # while not rospy.is_shutdown():
    for j in range(5000):
        start_time = time.perf_counter()
        for i in range(controller.N+1):
            if(i<controller.N):
                controller.solver.set(i, 'yref', controller.yref)
            else:
                controller.solver.set(i, 'yref', controller.yref_e)
        
        obs, R, acc_B, f_real = model.get_obs()
        u = u+du*ts
        # print(u)
        # print(f_real)
        controller.x0 = np.hstack((obs, u))
        du = controller.solver.solve_for_x0(controller.x0)

        # 此处3倍是考虑电机一阶模型，使得电机在ts时能够达到f_target
        f_target = u+du*ts*(1/(1-np.exp(-ts/0.025)))
        model.step(f_target)
        last_f_target_list.append(du)
        # last_f_target_list.append((f_target-last_f_target).copy()/ts)
        last_f_target = f_target
        end_time = time.perf_counter()
        print("time: ", end_time-start_time)

        rate.sleep()
    plt.plot(last_f_target_list)
    model.log_show()
    plt.show()