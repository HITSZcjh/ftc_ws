#!~/python_env/acados_env/bin/python
# coding=UTF-8

import numpy as np
import casadi as ca
from acados_template import AcadosModel

import os

import sys
sys.path.append("/home/jiao/ftc_ws")
import scipy.linalg
from src.INDI.scripts.rotors_model import RotorsUAVModel
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from src.traj.script.traj import CircleTrajectory

from casadi import *

from geometry_msgs.msg import PoseStamped
import rospy
from matplotlib import pyplot as plt

import sys

np.set_printoptions(threshold=sys.maxsize, precision=4, suppress=True)


class UAV_MPC_Without_Fault(object):
    def __init__(self, dt=0.04):
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
        self.rotor_thrust_coeff = 8.54858e-06
        self.body_length = 0.17
        self.mass = 0.73
        self.g = 9.81
        self.inertia = np.array([[0.007, 0, 0], [0, 0.007, 0], [0, 0, 0.012]])
        self.max_rotors_speed = 838

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
        model.name = 'UAV_Without_Fault'

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
        Q[10:13] = 1
        Q = np.diagflat(Q)
        R = 1*0.01*np.eye(self.nu)

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

        
        ocp.constraints.idxbx = np.array([10, 11, 12, 13, 14, 15, 16])
        ocp.constraints.lbx = np.array([-10, -10, -10, 0, 0, 0, 0])
        ocp.constraints.ubx = np.array([10, 10, 10, 
                                        self.rotor_thrust_coeff*self.max_rotors_speed**2, 
                                        self.rotor_thrust_coeff*self.max_rotors_speed**2, 
                                        self.rotor_thrust_coeff*self.max_rotors_speed**2, 
                                        self.rotor_thrust_coeff*self.max_rotors_speed**2])

        ocp.constraints.idxbu = np.array([0, 1, 2, 3])
        ocp.constraints.lbu = np.array([-15*7, -15*7, -15*7, -15*7])
        ocp.constraints.ubu = np.array([25*7, 25*7, 25*7, 25*7])

        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'  # 'EXACT''GAUSS_NEWTON'
        ocp.solver_options.nlp_solver_ext_qp_res = 0
        ocp.solver_options.nlp_solver_max_iter = 15
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.qp_solver_cond_N = 5
        ocp.solver_options.integrator_type = 'ERK'  # 'DISCRETE' 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # SQP_RTI, SQP

        ocp.code_export_directory = os.path.dirname(
            os.path.realpath(__file__))+"/c_generated_code/mpc_"+model.name
        json_file = os.path.dirname(os.path.realpath(
            __file__))+"/json_files/mpc_"+model.name+'_acados.json'
        self.solver = AcadosOcpSolver(ocp, json_file=json_file)


if __name__ == '__main__':

    rospy.init_node("UAV_MPC_node", anonymous=True)
    
    ts = 0.005
    model = RotorsUAVModel(ts,delay_time=0.03)
    controller = UAV_MPC_Without_Fault(dt=0.05)
    controller.yref[0:3] = np.array([0, 0, 1])
    controller.yref_e[0:3] = np.array([0, 0, 1])
    rate = rospy.Rate(1/ts)
    traj = CircleTrajectory([0,0,3], 3, 1)

    f_real_list = []
    f_target_list = []
    # while not rospy.is_shutdown():
    for j in range(1000):
        time_now = rospy.Time.now().to_sec()
        for i in range(controller.N+1):
            pos = traj.step(i*ts+j*controller.dt,i)
            if(i<controller.N):
                controller.yref[0:3] = pos
                controller.solver.set(i, 'yref', controller.yref)
            else:
                controller.yref_e[0:3] = pos
                controller.solver.set(i, 'yref', controller.yref_e)
        
        obs, R, acc_B, f_real = model.get_obs()
        controller.x0 = np.hstack((obs, f_real))
        u = controller.solver.solve_for_x0(controller.x0)

        # 此处3倍是考虑电机一阶模型，使得电机在ts时能够达到f_target
        f_target = f_real+u*ts*6
        model.step(f_target)

        rate.sleep()
        time_record = rospy.Time.now().to_sec() - time_now
        print("estimation time is {}".format(time_record))
        f_real_list.append(f_real.copy())
        f_target_list.append(f_target.copy())

    t = np.linspace(0, 0.01*len(f_real_list), len(f_real_list))
    fig7,axs7 = plt.subplots(4,1)
    fig7.suptitle('f')
    f_real_list = np.array(f_real_list)
    f_target_list = np.array(f_target_list)
    for i in range(4):
        axs7[i].plot(t, f_real_list[:,i], label='f_real')
        axs7[i].plot(t, f_target_list[:,i], label='f_target')

    plt.show()