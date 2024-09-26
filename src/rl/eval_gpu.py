import sys
from rl_model import RLModel
from PPO import PPO
import numpy as np
import matplotlib.pyplot as plt

load_actor_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/actor_model26-09-2024_11-10-58"
load_critic_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/critic_model26-09-2024_11-10-58"
num = 800
if __name__=="__main__":
    model = RLModel(log=True)
    obs = model.reset()
    model.print_quad_param()

    ppo = PPO(None, None, None, False)
    ppo.load_model(load_actor_model_path, load_critic_model_path, num)
    
    # obs = model.set_state()
        
    # model.k = np.ones([model.num_envs, 4], dtype=np.float32)
    # num = int(model.num_envs / 4)
    # k = np.linspace(0, 1, num)
    # model.k[:num,0] = k
    # model.k[num:num*2,1] = k
    # model.k[num*2:num*3,2] = k
    # model.k[num*3:num*4,3] = k
    # model.set_k()
    # print(model.k)

    dones = []
    for i in range(501):
        # print(obs[1])
        action = ppo.actor.get_action_without_sample(obs)
        # print(obs[:2,5:8])
        state = model.get_state()
        # print(state[:2])
        obs, reward, done = model.step(action)
        dones.append(done)

    # dones = np.array(dones)
    # live_list = []
    # for i in range(model.num_envs):
    #     first_true_index = np.where(dones[:,i] == True)[0][0]
    #     live_list.append(first_true_index)
    
    # live_list = np.array(live_list)

    # fig, axs = plt.subplots(2, 2)
    # k = np.linspace(0, 1, int(model.num_envs/4))
    # axs[0,0].plot(k, live_list[:int(model.num_envs/4)], label="k1")
    # axs[0,0].set_ylim(0, 1200)
    # axs[0,0].legend()
    # axs[0,1].plot(k, live_list[int(model.num_envs/4):int(model.num_envs/2)], label="k2")
    # axs[0,1].set_ylim(0, 1200)
    # axs[0,1].legend()
    # axs[1,0].plot(k, live_list[int(model.num_envs/2):int(model.num_envs*3/4)], label="k3")
    # axs[1,0].set_ylim(0, 1200)
    # axs[1,0].legend()
    # axs[1,1].plot(k, live_list[int(model.num_envs*3/4):], label="k4")
    # axs[1,1].set_ylim(0, 1200)
    # axs[1,1].legend()
    # live_acc = np.sum(live_list > 999) / model.num_envs
    # print(live_acc)
    # self.axs[0,0].set_ylim(0, 2)
    # self.axs[0,0].legend()    

    # plt.plot(live_list)

    # obs = model.obs
    # for i in range(500):
    #     u = ppo.actor.get_action(obs)
    #     u = u[0]*6
    #     obs = model.step(u)

    model.log_show()