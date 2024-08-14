import sys
from rotors_model import RotorsUAVModel,LPF
from PPO import PPO
import numpy as np
import rospy
import time
import matplotlib.pyplot as plt
from rl_model import obs_with_k

# no k add noise
load_actor_model_path = "/home/jiao/test_ws/src/rl/model/actor_model01-08-2024_16-18-16"
load_critic_model_path = "/home/jiao/test_ws/src/rl/model/critic_model01-08-2024_16-18-16"
num = 700

# with k add noise
load_actor_model_path = "/home/jiao/test_ws/src/rl/model/actor_model31-07-2024_10-21-15"
load_critic_model_path = "/home/jiao/test_ws/src/rl/model/critic_model31-07-2024_10-21-15"
num = 200

# with k
load_actor_model_path = "/home/jiao/test_ws/src/rl/model/actor_model31-07-2024_19-39-14"
load_critic_model_path = "/home/jiao/test_ws/src/rl/model/critic_model31-07-2024_19-39-14"
num = 280

load_actor_model_path = "/home/jiao/test_ws/src/rl/model/actor_model07-08-2024_15-11-13"
load_critic_model_path = "/home/jiao/test_ws/src/rl/model/critic_model07-08-2024_15-11-13"
num = 300

# no k
load_actor_model_path = "/home/jiao/test_ws/src/rl/model/actor_model06-08-2024_18-47-29"
load_critic_model_path = "/home/jiao/test_ws/src/rl/model/critic_model06-08-2024_18-47-29"
num = 440

# load_actor_model_path = "/home/jiao/test_ws/src/rl/model/actor_model07-08-2024_16-34-42"
# load_critic_model_path = "/home/jiao/test_ws/src/rl/model/critic_model07-08-2024_16-34-42"
# num = 610

load_actor_model_path = "/home/jiao/test_ws/src/rl/model/actor_model10-08-2024_16-31-03"
load_critic_model_path = "/home/jiao/test_ws/src/rl/model/critic_model10-08-2024_16-31-03"
num = 610

if __name__=="__main__":
    rospy.init_node("UAV_RL_node", anonymous=None)
    ts = 0.0025
    rate = rospy.Rate(1/ts)
    model = RotorsUAVModel(ts=ts, delay_time=0.03 ,log=True, BW=1/0.12)
    ppo = PPO(None, None, None, True)
    ppo.load_model(load_actor_model_path, load_critic_model_path, num)

    state = np.zeros(23)
    u = 2*np.ones(4)
    u_lpf = LPF(ts,1/0.08,np.zeros(4))
    acc_list = []
    for i in range(5000):
        start_time = time.perf_counter()
        obs, acc, omega_dot_f, acc_with_bias = model.get_obs(rl=True)
        acc_list.append(acc_with_bias.copy())


        state[:13] = obs
        # state[13:17] = u_lpf.calc(u)
        state[13:17] = u
        
        state[17:20] = acc
        state[20:23] = omega_dot_f
        state /= np.array((5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 5, 5, 30, 6, 6, 6, 6, 5, 5, 30, 30, 30, 30))
        
        model.k = np.array([1,1,1,1])
        if not obs_with_k:
            u = ppo.actor.get_action_without_sample(state)
        else:
            u = ppo.actor.get_action_without_sample(np.hstack((state,model.k)))
        u = u[0]

        u = (u+1)*3
        u = np.clip(u, 0, 6)
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
        end_time = time.perf_counter()
        print("time: ", end_time-start_time)
        rate.sleep()

    # state = model.state
    # for i in range(500):
    #     u = ppo.actor.get_action(state)
    #     u = u[0]*6
    #     state = model.step(u)
    acc_list = np.array(acc_list)
    t = np.arange(0, len(acc_list)*ts, ts)
    plt.plot(t, acc_list[:, 0], label="acc_x")
    plt.plot(t, acc_list[:, 1], label="acc_y")
    plt.plot(t, acc_list[:, 2], label="acc_z")
    plt.legend()

    model.log_show()
    plt.show()