import sys
sys.path.append("/home/jiao/ftc_ws")
from src.INDI.scripts.rotors_model import RotorsUAVModel
from PPO import PPO
import numpy as np
import rospy
import time
import matplotlib.pyplot as plt
# load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model21-02-2024_22-02-02"
# load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model21-02-2024_22-02-02"
# num = 100

# load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model22-02-2024_22-47-55"
# load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model22-02-2024_22-47-55"
# num = 250

load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model02-03-2024_20-58-31"
load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model02-03-2024_20-58-31"
num = 360

load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model03-03-2024_12-30-19"
load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model03-03-2024_12-30-19"
num = 180

load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model04-03-2024_22-16-05"
load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model04-03-2024_22-16-05"
num = 280

# load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model11-03-2024_13-00-32"
# load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model11-03-2024_13-00-32"
# num = 300

load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model11-03-2024_16-07-23"
load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model11-03-2024_16-07-23"
num = 40
if __name__=="__main__":
    rospy.init_node("UAV_RL_node", anonymous=True)
    ts = 0.01
    rate = rospy.Rate(1/ts)
    model = RotorsUAVModel(ts=ts, delay_time=0.03 ,log=True, BW=4)
    ppo = PPO(None, None, None, True)
    ppo.load_model(load_actor_model_path, load_critic_model_path, num)

    state = np.zeros(21)
    u = 2*np.ones(4)
    for i in range(1000):
        start_time = time.perf_counter()
        obs, acc, omega_dot_f = model.get_obs(rl=True)

        state[:13] = obs
        state[13:17] = u
        state[17] = acc
        state[18:21] = omega_dot_f
        state /= np.array((5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 5, 5, 30, 6, 6, 6, 6, 35, 30, 30, 30))

        u = ppo.actor.get_action_without_sample(state)
        u = u[0]

        u = (u+1)*3
        u = np.clip(u, 0, 6)

        model.k = np.array([1,1,1,1])
        # u[2] = 0
        model.step(u)
        rate.sleep()
        end_time = time.perf_counter()
        print("time: ", end_time-start_time)
    # state = model.state
    # for i in range(500):
    #     u = ppo.actor.get_action(state)
    #     u = u[0]*6
    #     state = model.step(u)

    model.log_show()
    plt.show()