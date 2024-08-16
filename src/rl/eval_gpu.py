import sys
from rl_model import RLModel
from PPO import PPO
import numpy as np
import matplotlib.pyplot as plt

# load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model21-02-2024_22-02-02"
# load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model21-02-2024_22-02-02"
# num = 100

# load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model22-02-2024_22-47-55"
# load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model22-02-2024_22-47-55"
# num = 250

# 1.添加噪声可用
load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model02-03-2024_20-58-31"
load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model02-03-2024_20-58-31"
num = 360

# 2.在1基础上使用exp型reward设计，可用
# load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model03-03-2024_12-30-19"
# load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model03-03-2024_12-30-19"
# num = 180

# 3.缩小动作幅度，不可用
# load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model03-03-2024_22-36-24"
# load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model03-03-2024_22-36-24"
# num = 290

# 4.添加噪声和0.02s延时，不可用
# load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model04-03-2024_13-34-20"
# load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model04-03-2024_13-34-20"
# num = 400

# 5.增大噪声幅度
load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model04-03-2024_22-16-05"
load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model04-03-2024_22-16-05"
num = 280

# # 6.5基础上使用exp型reward设计，且增大训练步数2000
# load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model05-03-2024_10-40-07"
# load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model05-03-2024_10-40-07"
# num = 0

# 7.5基础上调整样本的概率，且增大训练步数2000
load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model05-03-2024_13-03-19"
load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model05-03-2024_13-03-19"
num = 60

# # 8.7基础上使用exp型reward设计，可用
load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model05-03-2024_13-45-33"
load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model05-03-2024_13-45-33"
num = 30

# 9.
load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model11-03-2024_00-21-05"
load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model11-03-2024_00-21-05"
num = 300

load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model23-03-2024_17-11-09"
load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model23-03-2024_17-11-09"
num = 200
load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model23-03-2024_18-28-54"
load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model23-03-2024_18-28-54"
num = 190

load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model26-03-2024_22-31-19"
load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model26-03-2024_22-31-19"
num = 110

load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model27-03-2024_13-39-37"
load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model27-03-2024_13-39-37"
num = 800

# 修改了空气力矩系数
load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model27-03-2024_19-00-56"
load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model27-03-2024_19-00-56"
num = 150

# 控制不自旋，继续训练
load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model27-03-2024_19-57-48"
load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model27-03-2024_19-57-48"
num = 150

load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model28-03-2024_11-42-06"
load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model28-03-2024_11-42-06"
num = 250

load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model29-03-2024_14-47-06"
load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model29-03-2024_14-47-06"
num = 600

# 控制不自旋，0开始训练
# load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model28-03-2024_10-25-00"
# load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model28-03-2024_10-25-00"
# num = 170

# 毕设 无自旋，无done 
load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model05-05-2024_21-08-32"
load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model05-05-2024_21-08-32"
num = 200

# 毕设 无自旋，有done 
# load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model11-03-2024_00-21-05"
# load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model11-03-2024_00-21-05"
# num = 150

# 毕设 有自旋，无done 
# load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model06-05-2024_10-29-19"
# load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model06-05-2024_10-29-19"
# num = 200

# with k
load_actor_model_path = "/home/jiao/test_ws/src/rl/model/actor_model31-07-2024_19-39-14"
load_critic_model_path = "/home/jiao/test_ws/src/rl/model/critic_model31-07-2024_19-39-14"
num = 280

# no k
# load_actor_model_path = "/home/jiao/test_ws/src/rl/model/actor_model31-07-2024_10-21-15"
# load_critic_model_path = "/home/jiao/test_ws/src/rl/model/critic_model31-07-2024_10-21-15"
# num = 200

# with k add noise
# load_actor_model_path = "/home/jiao/test_ws/src/rl/model/actor_model01-08-2024_12-06-12"
# load_critic_model_path = "/home/jiao/test_ws/src/rl/model/critic_model01-08-2024_12-06-12"
# num = 710

# # no k add noise
# load_actor_model_path = "/home/jiao/test_ws/src/rl/model/actor_model01-08-2024_16-18-16"
# load_critic_model_path = "/home/jiao/test_ws/src/rl/model/critic_model01-08-2024_16-18-16"
# num = 700

# no k
# load_actor_model_path = "/home/jiao/test_ws/src/rl/model/actor_model10-08-2024_10-59-11"
# load_critic_model_path = "/home/jiao/test_ws/src/rl/model/critic_model10-08-2024_10-59-11"
# num = 280


#ADD LPF with K
load_actor_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/actor_model15-08-2024_14-15-19"
load_critic_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/critic_model15-08-2024_14-15-19"
num = 410

load_actor_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/actor_model15-08-2024_11-59-43"
load_critic_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/critic_model15-08-2024_11-59-43"
num = 1200

#ADD LPF no K
load_actor_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/actor_model15-08-2024_17-51-31"
load_critic_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/critic_model15-08-2024_17-51-31"
num = 460

load_actor_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/actor_model15-08-2024_20-51-36"
load_critic_model_path = "/home/jiao/rl_quad_ws/ftc_ws/src/rl/model/critic_model15-08-2024_20-51-36"
num = 660

if __name__=="__main__":
    model = RLModel(log=True)
    model.print_quad_param()
    model.reset()
    model.reset()

    ppo = PPO(None, None, None, False)
    ppo.load_model(load_actor_model_path, load_critic_model_path, num)

    state = model.set_state()
        
    model.k = np.ones([model.num_envs, 4], dtype=np.float32)
    num = int(model.num_envs / 4)
    k = np.linspace(0, 1, num)
    model.k[:num,0] = k
    model.k[num:num*2,1] = k
    model.k[num*2:num*3,2] = k
    model.k[num*3:num*4,3] = k
    model.set_k()
    # print(model.k)

    dones = []
    for i in range(501):
        # print(state[1])
        action = ppo.actor.get_action_without_sample(state)
        state, reward, done = model.step(action)
        dones.append(done)

    dones = np.array(dones)
    live_list = []
    for i in range(model.num_envs):
        first_true_index = np.where(dones[:,i] == True)[0][0]
        live_list.append(first_true_index)
    
    live_list = np.array(live_list)

    fig, axs = plt.subplots(2, 2)
    k = np.linspace(0, 1, int(model.num_envs/4))
    axs[0,0].plot(k, live_list[:int(model.num_envs/4)], label="k1")
    axs[0,0].set_ylim(0, 1200)
    axs[0,0].legend()
    axs[0,1].plot(k, live_list[int(model.num_envs/4):int(model.num_envs/2)], label="k2")
    axs[0,1].set_ylim(0, 1200)
    axs[0,1].legend()
    axs[1,0].plot(k, live_list[int(model.num_envs/2):int(model.num_envs*3/4)], label="k3")
    axs[1,0].set_ylim(0, 1200)
    axs[1,0].legend()
    axs[1,1].plot(k, live_list[int(model.num_envs*3/4):], label="k4")
    axs[1,1].set_ylim(0, 1200)
    axs[1,1].legend()
    live_acc = np.sum(live_list > 999) / model.num_envs
    print(live_acc)
    # self.axs[0,0].set_ylim(0, 2)
    # self.axs[0,0].legend()    

    # plt.plot(live_list)

    # state = model.state
    # for i in range(500):
    #     u = ppo.actor.get_action(state)
    #     u = u[0]*6
    #     state = model.step(u)

    model.log_show()