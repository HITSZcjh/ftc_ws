import sys
from rotors_model import RotorsUAVModel,LPF
from PPO import PPO
import numpy as np
import rospy
import time
import matplotlib.pyplot as plt
from rl_model import obs_with_k

class PositionController:
    def __init__(self, ts) -> None:
        self.ts = ts
        self.kp_pos = np.array([1,1,0.01], dtype=np.float32)
        self.ki_pos = np.array([0.0,0.0,0.1], dtype=np.float32)
        self.int_pos = np.zeros_like(self.ki_pos)
        self.kp_vel = np.array([1,1,1], dtype=np.float32)
        self.ki_vel = np.array([0.2,0.2,0], dtype=np.float32)
        self.int_vel = np.zeros_like(self.ki_vel)

        self.vel_max = 0.5
        # self.int_max_pos = 2*self.vel_max / (self.ki_pos+1e-8)
        self.acc_max = 4.0
        self.g = np.array([0,0,-9.81])
        self.nw_xy_max = np.sin(np.pi/18)

        # self.kp = np.array([1,1,0.3], dtype=np.float32)
        # self.kv = np.array([1,1,1], dtype=np.float32)
        
        # self.vel_max = 1.0
        # self.acc_max = 1.0
        # self.g = np.array([0,0,-9.81])
        # self.nw_xy_max = np.sin(np.pi/6)

    def calc(self, ref_pos, fdb_pos, fdb_vel):
        err_pos = ref_pos - fdb_pos
        self.int_pos += err_pos * self.ts
        ref_vel = np.clip(err_pos*self.kp_pos+self.int_pos*self.ki_pos, -self.vel_max, self.vel_max)
        err_vel = ref_vel-fdb_vel
        self.int_vel += err_vel * self.ts
        ref_acc = np.clip(err_vel*self.kp_vel+self.int_vel*self.ki_vel, -self.acc_max, self.acc_max)
        nw = (ref_acc-self.g)/np.linalg.norm(ref_acc-self.g)

        if(np.linalg.norm(nw[:2])>self.nw_xy_max):
            nw[:2] = nw[:2]/np.linalg.norm(nw[:2])*self.nw_xy_max
        
        print(ref_vel[2])
        # print(np.linalg.norm(nw[:2]),self.nw_xy_max)

        return nw[0], nw[1], ref_vel[2]

load_actor_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/actor_model26-09-2024_08-57-08"
load_critic_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/critic_model26-09-2024_08-57-08"
num = 500

# load_actor_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/actor_model26-09-2024_09-56-27"
# load_critic_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/critic_model26-09-2024_09-56-27"
# num = 1500

load_actor_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/actor_model26-09-2024_11-10-58"
load_critic_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/critic_model26-09-2024_11-10-58"
num = 800

load_actor_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/actor_model26-09-2024_12-15-41"
load_critic_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/critic_model26-09-2024_12-15-41"
num = 600

load_actor_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/actor_model26-09-2024_16-34-59"
load_critic_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/critic_model26-09-2024_16-34-59"
num = 1900

load_actor_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/actor_model26-09-2024_17-11-27"
load_critic_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/critic_model26-09-2024_17-11-27"
num = 1170

load_actor_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/actor_model26-09-2024_21-55-21"
load_critic_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/critic_model26-09-2024_21-55-21"
num = 3000

load_actor_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/actor_model27-09-2024_11-19-55"
load_critic_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/critic_model27-09-2024_11-19-55"
num = 1400

load_actor_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/actor_model27-09-2024_21-55-21"
load_critic_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/critic_model27-09-2024_21-55-21"
num = 2400

load_actor_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/actor_model02-10-2024_17-29-30"
load_critic_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/critic_model02-10-2024_17-29-30"
num = 1200

load_actor_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/actor_model02-10-2024_21-40-27"
load_critic_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/critic_model02-10-2024_21-40-27"
num = 1800

if __name__=="__main__":
    rospy.init_node("UAV_RL_node", anonymous=None)
    ts = 0.0025
    rate = rospy.Rate(1/ts)
    model = RotorsUAVModel(ts=ts, delay_time=None ,log=True, BW=1/0.12)
    ppo = PPO(None, None, None, True)
    ppo.load_model(load_actor_model_path, load_critic_model_path, num)
    state = np.zeros(ppo.env.state_dim)
    u = 0*np.ones(4)
    u_lpf = LPF(ts,1/0.02,np.zeros(4))
    acc_list = []
    last_f_target = np.zeros(4)
    last_f_target_list = []

    pos_controller = PositionController(ts)
    ref_pos = np.array([0,0,3])

    for i in range(10000):
        start_time = time.perf_counter()
        obs, acc, omega_dot_f, acc_with_bias = model.get_obs(rl=True)
        acc_list.append(acc_with_bias.copy())


        state[:8] = obs[5:13]
        state[8:12] = u
        nwx, nwy, velz = pos_controller.calc(ref_pos, obs[0:3], obs[3:6])

        state[12:15] = np.array([nwx, nwy, velz])
        state /= np.array((5, 1, 1, 1, 1, 10, 10, 10, 3, 3, 3, 3, 1, 1, 5))
        
        model.k = np.array([1,1,1,1])
        if not obs_with_k:
            u = ppo.actor.get_action_without_sample(state)
        else:
            u = ppo.actor.get_action_without_sample(np.hstack((state,model.k)))
        u = u[0]

        u = (u+1)*1.5
        u = np.clip(u, 0, 3)
        u = u_lpf.calc(u)
        # if(i>1000 and i < 3000):
        #     model.k = np.array([1,1,(3000-i)/2000,1])
        # if(i>3000):
            # model.k = np.array([1,1,(3000-i)/2000,1])

        # if(i<1000):
        #     model.k = np.array([1,1,1,1])
        # elif(i<2000):
        #     model.k = np.array([1,1,0.5,1])
        # elif(i<3000):
        #     model.k = np.array([1,1,0,1])
        model.step(u)
        last_f_target_list.append((u-last_f_target).copy()/ts)
        last_f_target = u
        end_time = time.perf_counter()
        # print("time: ", end_time-start_time)
        rate.sleep()

    # state = model.state
    # for i in range(500):
    #     u = ppo.actor.get_action(state)
    #     u = u[0]*6
    #     state = model.step(u)
    # acc_list = np.array(acc_list)
    # t = np.arange(0, len(acc_list)*ts, ts)
    # plt.plot(t, acc_list[:, 0], label="acc_x")
    # plt.plot(t, acc_list[:, 1], label="acc_y")
    # plt.plot(t, acc_list[:, 2], label="acc_z")
    # plt.legend()
    plt.plot(last_f_target_list)

    model.log_show()
    plt.tight_layout()  # 调整子图布局
    plt.show()