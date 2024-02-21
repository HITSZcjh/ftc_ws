import sys
sys.path.append("/home/jiao/ftc_ws")
from src.INDI.scripts.uav_model import SimpleUAVModel
from PPO import PPO

load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model21-02-"+"2024_14-02-14"
load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model21-02-"+"2024_14-02-14"
if __name__=="__main__":
    model = SimpleUAVModel(ts=0.01,log=True)
    ppo = PPO(None, None, None)
    ppo.load_model(load_actor_model_path, load_critic_model_path, 150)
    state = model.state
    for i in range(1000):
        action = ppo.actor.get_action(state)
        action = action[0]
        state = model.step(action*6)
    model.log_show()