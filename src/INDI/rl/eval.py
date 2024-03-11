import sys
sys.path.append("/home/jiao/ftc_ws")
from src.INDI.scripts.uav_model import SimpleUAVModel
from PPO import PPO
import numpy as np

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

# 6.5基础上使用exp型reward设计，且增大训练步数2000
load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model05-03-2024_10-40-07"
load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model05-03-2024_10-40-07"
num = 0

# 7.5基础上调整样本的概率，且增大训练步数2000
load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model05-03-2024_13-03-19"
load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model05-03-2024_13-03-19"
num = 40

# 8.7基础上使用exp型reward设计，可用
load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model05-03-2024_13-45-33"
load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model05-03-2024_13-45-33"
num = 40

# 9.修改随机初始化姿态设定
# load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model08-03-2024_09-22-55"
# load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model08-03-2024_09-22-55"
# num = 320

# 10.8基础上修改reward和maxeplen
load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model08-03-2024_11-31-57"
load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model08-03-2024_11-31-57"
num = 80

# 11.修改随机初始化姿态设定，修改角加速度滤波器系数
load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model08-03-2024_13-05-09"
load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model08-03-2024_13-05-09"
num = 140

# *.测试
load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model09-03-2024_10-21-44"
load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model09-03-2024_10-21-44"
num = 140

if __name__=="__main__":
    model = SimpleUAVModel(ts=0.01, log=True, BW=5)
    ppo = PPO(None, None, None, False)
    ppo.load_model(load_actor_model_path, load_critic_model_path, num)

    state = np.zeros(21)
    u = 2*np.ones(4)
    # u_list = [u for _ in range(int(model.delay_time/model.ts))]
    for i in range(1000):
        obs, acc, omega_dot_f = model.get_obs_rl()

        # u_list.append(u.copy())
        # u = u_list.pop(0)

        state[:13] = obs
        state[13:17] = u # u_list.pop(0)
        state[17] = acc
        state[18:21] = omega_dot_f

        du = ppo.actor.get_action_without_sample(state)
        du = du[0]

        du = np.clip(du,-1,1)
        u += 50*du*model.ts
        model.k = np.array([1,1,1,0])
        model.step(u)
        u = np.clip(u, 0, 6)

    # state = model.state
    # for i in range(500):
    #     u = ppo.actor.get_action(state)
    #     u = u[0]*6
    #     state = model.step(u)

    model.log_show()