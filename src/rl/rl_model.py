from MyQuadrotorEnv import VecEnv
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize, precision=3, suppress=True)

obs_with_k = True

class RLModel:
    def __init__(self, log=False, ts=0.02):
        self.env = VecEnv()
        self.num_envs = self.env.get_num_envs()
        
        if(obs_with_k):
            self.state_dim = self.env.get_obs_dim() + 4
            self._observation = np.zeros([self.num_envs, self.state_dim - 4],
                                        dtype=np.float32)
        else:
            self.state_dim = self.env.get_obs_dim()
            self._observation = np.zeros([self.num_envs, self.state_dim],
                                        dtype=np.float32)

        self.action_dim = self.env.get_action_dim()

        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros((self.num_envs), dtype=bool)
        self._extraInfoNames = self.env.getExtraInfoNames()
        self._extraInfo = np.zeros([self.num_envs, len(self._extraInfoNames)], dtype=np.float32)
        self._cmd_vels = np.zeros([self.num_envs, 3], dtype=np.float32)
        self.k = np.ones([self.num_envs, 4], dtype=np.float32)
        self.state = np.zeros([self.num_envs, self.env.get_state_dim()], dtype=np.float32)
        self.state[:,2] = 3
        self.state[:,6] = 1
        # self.state[:,12] = 10


        self.rewards = [[] for _ in range(self.num_envs)]
        self.epreturn = []
        self.eplen = []
        self.extra_infos = []
        self.curriculum_level = 0

        self.log = log
        if log:
            self.obs_list = [[],[],[],[],[]]
            self.pos_err = np.zeros([self.num_envs])
            self.pos_goal = np.array([0,0,3])
        self.ts = ts

        self.reset()
    
    def reset(self):
        self.rewards = [[] for _ in range(self.num_envs)]
        self.epreturn = []
        self.eplen = []
        self.extra_infos = []
        self.env.reset(self._observation)
        return self.get_obs()
    
    def step(self, action):
        action = action.copy()
        action = (action+1)*3
        self.env.step(action, self._observation, self._reward, self._done, self._extraInfo)

        for i in range(self.num_envs):
            self.rewards[i].append(self._reward[i])
            if self._done[i]:
                self.epreturn.append(sum(self.rewards[i]))
                self.eplen.append(len(self.rewards[i]))
                self.rewards[i] = []


        info = {self._extraInfoNames[j]: np.mean(self._extraInfo[:, j]) for j in range(0, len(self._extraInfoNames))}
        self.extra_infos.append(info.copy())

        if self.log:
            log_temp = np.hstack((self._observation[:,:13], np.clip(action, 0, 6))).astype(np.float32)
            self.obs_list[0].append(log_temp[0,:].copy())
            self.obs_list[1].append(log_temp[int(self.num_envs/4),:].copy())
            self.obs_list[2].append(log_temp[int(self.num_envs/2)+int(self.num_envs/4*0.0),:].copy())
            self.obs_list[3].append(log_temp[int(self.num_envs*3/4)+int(self.num_envs/4*0.0),:].copy())
            self.obs_list[4].append(log_temp[self.num_envs-1,:].copy())

            pos = self._observation[:,0:3]*np.array((5, 5, 5))
            self.pos_err += np.linalg.norm(pos - self.pos_goal, axis=1) 

        
        return self.get_obs(), self._reward.copy(), self._done.copy()
    
    def set_k(self):
        self.env.set_k(self.k)

    def set_curriculum_level(self):
        self.env.set_curriculum_level(self.curriculum_level)

    def get_k(self):
        self.env.get_k(self.k)
        return self.k.copy()
    
    def set_state(self):
        self.env.set_state(self.state, self._observation)
        return self.get_obs()

    def get_state(self):
        self.env.get_state(self.state)
        return self.state.copy()

    def set_baseline(self, averageReturn, stdReturn):
        self.env.set_baseline(averageReturn, stdReturn)

    def get_obs(self):
        if(obs_with_k):
            obs = self._observation.copy().astype(np.float32)
            k = self.get_k()
            return np.hstack((obs, k))
        else:
            return self._observation.copy()
    
    def print_quad_param(self):
        self.env.print_quad_param()

    def set_cmd_vels(self):
        self.env.set_cmd_vels(self._cmd_vels)

    def log_show(self):
        if self.log:
            import matplotlib.pyplot as plt
            self.obs_list = np.array(self.obs_list)
            self.obs_list[:,:,:13] = self.obs_list[:,:,:13]*np.array((5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 5, 5, 30))
            self.obs_list = self.obs_list[:,:-1,:]
            t = np.linspace(0, self.obs_list.shape[1]*self.ts, self.obs_list.shape[1])

            self.pos_err /= self.obs_list.shape[1]
            self.fig, self.axs = plt.subplots(2, 2)
            k = np.linspace(0, 1, int(self.num_envs/4))
            self.axs[0,0].plot(k, self.pos_err[:int(self.num_envs/4)], label="k1")
            self.axs[0,0].set_ylim(0, 2)
            self.axs[0,0].legend()
            # self.axs[0,0].set_ylabel("pos_err")
            self.axs[0,1].plot(k, self.pos_err[int(self.num_envs/4):int(self.num_envs/2)], label="k2")
            self.axs[0,1].set_ylim(0, 2)
            # self.axs[0,1].set_ylabel("pos_err")
            self.axs[0,1].legend()
            self.axs[1,0].plot(k, self.pos_err[int(self.num_envs/2):int(self.num_envs*3/4)], label="k3")
            self.axs[1,0].set_ylim(0, 2)
            # self.axs[1,0].set_ylabel("pos_err")
            self.axs[1,0].legend()
            self.axs[1,1].plot(k, self.pos_err[int(self.num_envs*3/4):], label="k4")
            self.axs[1,1].set_ylim(0, 2)
            # self.axs[1,1].set_ylabel("pos_err")
            self.axs[1,1].legend()

            for i in range(5):

                # self.fig, self.axs = plt.subplots(2, 2)
                # self.axs[0,0].plot(t, self.obs_list[i, :, 0], label="px")
                # self.axs[0,0].plot(t, self.obs_list[i, :, 1], label="py")
                # self.axs[0,0].plot(t, self.obs_list[i, :, 2], label="pz")
                # self.axs[0,0].set_ylim(-1, 4)   
                # self.axs[0,0].legend()
                # self.axs[0,1].plot(t, self.obs_list[i, :, 3], label="vx")
                # self.axs[0,1].plot(t, self.obs_list[i, :, 4], label="vy")
                # self.axs[0,1].plot(t, self.obs_list[i, :, 5], label="vz")
                # self.axs[0,1].set_ylim(-3, 3)
                # self.axs[0,1].legend()
                # self.axs[1,0].plot(t, self.obs_list[i, :, 6], label="w")
                # self.axs[1,0].plot(t, self.obs_list[i, :, 7], label="x")
                # self.axs[1,0].plot(t, self.obs_list[i, :, 8], label="y")
                # self.axs[1,0].plot(t, self.obs_list[i, :, 9], label="z")
                # self.axs[1,0].legend()
                # # self.axs[1,0].set_ylim(0, 1)

                # self.axs[1,1].plot(t, self.obs_list[i, :, 10], label="wx")
                # self.axs[1,1].plot(t, self.obs_list[i, :, 11], label="wy")
                # self.axs[1,1].plot(t, self.obs_list[i, :, 12], label="wz")
                # self.axs[1,1].legend()
                # self.axs[1,1].set_ylim(-18, 18)


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
    model.print_quad_param()


    # num = int(model.num_envs / 4)
    # k = np.linspace(0, 1, num)
    # model.k[:num,0] = k
    # model.k[num:num*2,1] = k
    # model.k[num*2:num*3,2] = k
    # model.k[num*3:num*4,3] = k
    # model.set_k()

    # print(model.get_k())

    # print(model.get_obs())



    # model = RLModel()
    # print(model.get_obs())

    # action = 0.5*np.ones((model.num_envs, model.action_dim))
    # model.step(action)
    # model.step(action)
    # sum_dict = {key: 0.0 for key in model.infos[0].keys()}
    # for d in model.infos:
    #     for key, value in d.items():
    #         sum_dict[key] += value

    # # 计算每个键的均值
    # num_data = len(model.infos)
    # mean_dict = {key: value / num_data for key, value in sum_dict.items()}

    # print(mean_dict)
    # print(model._extraInfo)

    # obs = model.reset()
    # print(obs.shape)
    # print(model.state_dim)
    # print(obs[0])
    # obs = model.set_state()
    # print(obs[0])
    # state = model.get_state()
    # # model.k[:,0] = 0
    # model.set_k()
    # print(state[0])
    # action = 0.5*np.ones((model.num_envs, model.action_dim),dtype=np.float32)
    # for i in range(10):
    #     action = -action
    #     obs = model.step(action)[0][0]

    #     state = model.get_state()[0]
    #     print("obs:", obs)
    #     print("x:", state)


    # print(obs[0])


    # q_list = obs[:,6:10]
    # for i in range(10):
    #     q = q_list[i]
    #     R = np.array([[1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] - q[0] * q[3]), 2 * (q[1] * q[3] + q[0] * q[2])],
    #                   [2 * (q[1] * q[2] + q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] - q[0] * q[1])],
    #                   [2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)]])
    #     print(R)

    # obs = model.set_state()
    # print(obs[0])
    # model.set_k()
    # action = 1*np.ones((model.num_envs, model.action_dim))
    # for i in range(100):
    #     # action = -action
    #     obs = model.step(action)[0]
    #     print(obs[0])