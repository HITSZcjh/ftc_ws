from QuadrotorEnv import VecEnv
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize, precision=3, suppress=True)
class RLModel:
    def __init__(self, log=False, ts=0.01):
        self.env = VecEnv()
        self.num_envs = self.env.get_num_envs()
        self.state_dim = self.env.get_obs_dim()
        # self.state_dim = 17
        self.action_dim = self.env.get_action_dim()
        self._observation = np.zeros([self.num_envs, self.state_dim],
                                    dtype=np.float64)
        self._reward = np.zeros(self.num_envs, dtype=np.float64)
        self._done = np.zeros((self.num_envs), dtype=bool)
        self.k = np.ones([self.num_envs, 4], dtype=np.float64)
        self.state = np.zeros([self.num_envs, self.env.get_state_dim()], dtype=np.float64)
        self.state[:,2] = 3
        self.state[:,6] = 1

        self.log = log
        if log:
            self.obs_list = [[],[],[],[],[]]
        self.ts = ts

    def reset(self):
        self.env.reset(self._observation)
        return self._observation.copy().astype(np.float32)
    
    def step(self, action):
        action = action.copy()
        action = (action+1)*3
        action = action.astype(np.float64)
        self.env.step(action, self._observation, self._reward, self._done)
        if self.log:
            log_temp = np.hstack((self._observation[:,:13], np.clip(action, 0, 6))).astype(np.float32)
            self.obs_list[0].append(log_temp[0,:].copy())
            self.obs_list[1].append(log_temp[int(self.num_envs/4),:].copy())
            self.obs_list[2].append(log_temp[int(self.num_envs/2),:].copy())
            self.obs_list[3].append(log_temp[int(self.num_envs*3/4),:].copy())
            self.obs_list[4].append(log_temp[self.num_envs-1,:].copy())
        return self._observation.copy().astype(np.float32), self._reward.copy().astype(np.float32), self._done.copy().astype(np.float32)
    
    def set_k(self):
        self.env.set_k(self.k)

    def set_state(self):
        self.env.set_state(self.state, self._observation)
        return self._observation.copy().astype(np.float32)

    def get_state(self):
        self.env.get_state(self.state)
        return self.state.copy().astype(np.float32)

    def log_show(self):
        if self.log:
            import matplotlib.pyplot as plt
            self.obs_list = np.array(self.obs_list)
            self.obs_list[:,:,:13] = self.obs_list[:,:,:13]*np.array((5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 5, 5, 30))
            self.obs_list = self.obs_list[:,:-1,:]
            t = np.linspace(0, self.obs_list.shape[1]*self.ts, self.obs_list.shape[1])

            for i in range(5):
                self.fig, self.axs = plt.subplots(2, 3)
                self.axs[0,0].plot(t, self.obs_list[i, :, 0], label="px")
                self.axs[0,0].plot(t, self.obs_list[i, :, 1], label="py")
                self.axs[0,0].plot(t, self.obs_list[i, :, 2], label="pz")
                # self.axs[0,0].set_ylim(-5, 5)   
                self.axs[0,0].legend()
                self.axs[0,1].plot(t, self.obs_list[i, :, 3], label="vx")
                self.axs[0,1].plot(t, self.obs_list[i, :, 4], label="vy")
                self.axs[0,1].plot(t, self.obs_list[i, :, 5], label="vz")
                # self.axs[0,1].set_ylim(-5, 5)
                self.axs[0,1].legend()
                self.axs[1,0].plot(t, self.obs_list[i, :, 6], label="w")
                self.axs[1,0].plot(t, self.obs_list[i, :, 7], label="x")
                self.axs[1,0].plot(t, self.obs_list[i, :, 8], label="y")
                self.axs[1,0].plot(t, self.obs_list[i, :, 9], label="z")
                self.axs[1,0].legend()
                self.axs[1,1].plot(t, self.obs_list[i, :, 10], label="wx")
                self.axs[1,1].plot(t, self.obs_list[i, :, 11], label="wy")
                self.axs[1,1].plot(t, self.obs_list[i, :, 12], label="wz")
                self.axs[1,1].legend()
                self.axs[0,2].plot(t, self.obs_list[i, :, 13], label="f1")
                self.axs[0,2].plot(t, self.obs_list[i, :, 14], label="f2")
                self.axs[0,2].plot(t, self.obs_list[i, :, 15], label="f3")
                self.axs[0,2].plot(t, self.obs_list[i, :, 16], label="f4")
                self.axs[0,2].legend()
            plt.show()



if __name__ == '__main__':
    model = RLModel()
    obs = model.reset()
    print(obs[0])
    obs = model.set_state()
    print(obs[0])
    state = model.get_state()
    # model.k[:,0] = 0
    model.set_k()
    print(state[0])
    action = 0.5*np.ones((model.num_envs, model.action_dim))
    for i in range(10):
        action = -action
        obs = model.step(action)[0]
        state = model.get_state()
        print(obs[0])
        # print(state[0])


    # print(obs[0])


    # q_list = obs[:,6:10]
    # for i in range(10):
    #     q = q_list[i]
    #     R = np.array([[1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] - q[0] * q[3]), 2 * (q[1] * q[3] + q[0] * q[2])],
    #                   [2 * (q[1] * q[2] + q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] - q[0] * q[1])],
    #                   [2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)]])
    #     print(R)
    exit()

    obs = model.set_state()
    print(obs[0])
    model.set_k()
    action = 1*np.ones((model.num_envs, model.action_dim))
    for i in range(100):
        # action = -action
        obs = model.step(action)[0]
        print(obs[0])