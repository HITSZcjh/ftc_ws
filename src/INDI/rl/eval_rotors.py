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


if __name__=="__main__":
    rospy.init_node("UAV_RL_node", anonymous=True)
    ts = 0.01
    rate = rospy.Rate(1/ts)
    model = RotorsUAVModel(ts=ts,delay_time=0.03, log=True)
    ppo = PPO(None, None, None, False)
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

        du = ppo.actor.get_action_without_sample(state)
        du = du[0]

        du = np.clip(du,-1,1)
        u += 50*du*model.ts
        model.k = np.array([1,1,1,0])
        model.step(u)
        u = np.clip(u, 0, 6)
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