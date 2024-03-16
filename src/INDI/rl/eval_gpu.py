import sys
sys.path.append("/home/jiao/ftc_ws")
from src.INDI.rl.rl_model import RLModel
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

# 9.修改网络层数，不可用
load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model11-03-2024_00-21-05"
load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model11-03-2024_00-21-05"
num = 300

load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model11-03-2024_13-00-32"
load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model11-03-2024_13-00-32"
num = 300

load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model16-03-2024_13-36-33"
load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model16-03-2024_13-36-33"
num = 350

load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model11-03-2024_00-21-05"
load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model11-03-2024_00-21-05"
num = 300

if __name__=="__main__":
    model = RLModel(log=True)
    ppo = PPO(None, None, None, True)
    ppo.load_model(load_actor_model_path, load_critic_model_path, num)

    state = model.set_state()
    
    num = int(model.num_envs / 4)
    k = np.linspace(0, 1, num)
    model.k[:num,0] = k
    model.k[num:num*2,1] = k
    model.k[num*2:num*3,2] = k
    model.k[num*3:num*4,3] = k
    model.set_k()
    dones = []
    for i in range(1001):
        action = ppo.actor.get_action_without_sample(state)
        state, reward, done = model.step(action)
        dones.append(done)

    dones = np.array(dones)
    live_list = []
    for i in range(model.num_envs):
        first_true_index = np.where(dones[:,i] == True)[0][0]
        live_list.append(first_true_index)
    
    live_list = np.array(live_list)

    
    plt.plot(live_list)

    # state = model.state
    # for i in range(500):
    #     u = ppo.actor.get_action(state)
    #     u = u[0]*6
    #     state = model.step(u)

    model.log_show()