from QuadrotorEnv import VecEnv
import numpy as np

class RLModel:
    def __init__(self):
        self.env = VecEnv()
        self.num_envs = self.env.get_num_envs()
        self.state_dim = self.env.get_obs_dim()
        # self.state_dim = 17
        self.action_dim = self.env.get_action_dim()
        self._observation = np.zeros([self.num_envs, self.state_dim],
                                    dtype=np.float64)
        self._reward = np.zeros(self.num_envs, dtype=np.float64)
        self._done = np.zeros((self.num_envs), dtype=bool)

    def reset(self):
        self.env.reset(self._observation)
        return self._observation.copy().astype(np.float32)
    
    def step(self, action):
        action = action*50
        action = action.astype(np.float64)
        self.env.step(action, self._observation, self._reward, self._done)
        return self._observation.copy().astype(np.float32), self._reward.copy().astype(np.float32), self._done.copy().astype(np.float32)
    
if __name__ == '__main__':
    model = RLModel()
    action = np.ones((model.num_envs, model.action_dim))
    for i in range(100):
        obs = model.step(action)[0]
        print(obs[0])