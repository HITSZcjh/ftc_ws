import sys
sys.path.append("/home/jiao/ftc_ws")
from src.INDI.scripts.uav_model import SimpleUAVModel
from PPO import PPO
import numpy as np

load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model21-02-"+"2024_22-02-02"
load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model21-02-"+"2024_22-02-02"
num = 100

if __name__=="__main__":
    model = SimpleUAVModel(ts=0.01,log=True)
    ppo = PPO(None, None, None)
    ppo.load_model(load_actor_model_path, load_critic_model_path, num)

    state = model.state
    u = np.zeros(4)
    state[-4:] = u
    for i in range(500):
        du = ppo.actor.get_action_without_sample(state)
        du = du[0]
        du[-1] = 0
        du = np.clip(du,-1,1)
        u += 50*du*model.ts
        state = model.step(u)
        u = np.clip(u, 0, 6)
        state[-4:] = u

    # state = model.state
    # for i in range(500):
    #     u = ppo.actor.get_action(state)
    #     u = u[0]*6
    #     state = model.step(u)

    model.log_show()