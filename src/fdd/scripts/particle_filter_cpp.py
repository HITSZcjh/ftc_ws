from FDDParticleFilter import ParticleFilter
import numpy as np

class ParticleFilterCPP(object):
    def __init__(self, num_particles, num_threads, ts, log=False) -> None:
        self.pf = ParticleFilter(num_particles, num_threads, ts)
        self.state_est = np.zeros(17,dtype=np.float64)
        self.k_est = np.ones(4,dtype=np.float64)

        self.ts = ts
        self.log = log
        if self.log:
            self.log_state_est_list = []
            self.log_k_est_list = []

    def set_init_state(self, obs):
        obs = obs.astype(np.float64)
        self.pf.SetInitState(obs)
        
    def loop(self, action, obs):
        action = action.astype(np.float64)
        obs = obs.astype(np.float64)
        self.pf.Loop(obs, action, self.state_est, self.k_est)

        if self.log:
            self.log_state_est_list.append(self.state_est.copy())
            self.log_k_est_list.append(self.k_est.copy())
        return self.state_est, self.k_est
    
    def log_show(self):
        if self.log:
            import matplotlib.pyplot as plt
            t = np.arange(0, len(self.log_state_est_list)*self.ts, self.ts)
            self.log_state_est_list = np.array(self.log_state_est_list)
            self.log_k_est_list = np.array(self.log_k_est_list)

            self.fig, self.axs = plt.subplots(2, 2)
            self.axs[0,0].plot(t, self.log_state_est_list[:, 0], label="px")
            self.axs[0,0].plot(t, self.log_state_est_list[:, 1], label="py")
            self.axs[0,0].plot(t, self.log_state_est_list[:, 2], label="pz")
            self.axs[0,0].legend()
            self.axs[0,1].plot(t, self.log_state_est_list[:, 3], label="vx")
            self.axs[0,1].plot(t, self.log_state_est_list[:, 4], label="vy")
            self.axs[0,1].plot(t, self.log_state_est_list[:, 5], label="vz")
            self.axs[0,1].legend()
            self.axs[1,0].plot(t, self.log_state_est_list[:, 6], label="w")
            self.axs[1,0].plot(t, self.log_state_est_list[:, 7], label="x")
            self.axs[1,0].plot(t, self.log_state_est_list[:, 8], label="y")
            self.axs[1,0].plot(t, self.log_state_est_list[:, 9], label="z")
            self.axs[1,0].legend()
            self.axs[1,1].plot(t, self.log_state_est_list[:, 10], label="wx")
            self.axs[1,1].plot(t, self.log_state_est_list[:, 11], label="wy")
            self.axs[1,1].plot(t, self.log_state_est_list[:, 12], label="wz")
            self.axs[1,1].legend()


            self.fig, self.axs = plt.subplots(2,2)
            self.axs[0,0].plot(t, self.log_state_est_list[:,13], label="f1")
            self.axs[0,0].legend()
            self.axs[0,1].plot(t, self.log_state_est_list[:,14], label="f2")
            self.axs[0,1].legend()
            self.axs[1,0].plot(t, self.log_state_est_list[:,15], label="f3")
            self.axs[1,0].legend()
            self.axs[1,1].plot(t, self.log_state_est_list[:,16], label="f4")
            self.axs[1,1].legend()

            self.fig, self.axs = plt.subplots(2,2)
            self.axs[0,0].plot(t, self.log_k_est_list[:,0], label="k1")
            self.axs[0,0].set_ylim(0,1)
            self.axs[0,0].legend()
            self.axs[0,1].plot(t, self.log_k_est_list[:,1], label="k2")
            self.axs[0,1].set_ylim(0,1)
            self.axs[0,1].legend()
            self.axs[1,0].plot(t, self.log_k_est_list[:,2], label="k3")
            self.axs[1,0].set_ylim(0,1)
            self.axs[1,0].legend()
            self.axs[1,1].plot(t, self.log_k_est_list[:,3], label="k4")
            self.axs[1,1].set_ylim(0,1)
            self.axs[1,1].legend()

            self.fig, self.axs = plt.subplots()
            self.axs.plot(t, self.log_k_est_list[:,0], label="k1")
            self.axs.plot(t, self.log_k_est_list[:,1], label="k2")
            self.axs.plot(t, self.log_k_est_list[:,2], label="k3")
            self.axs.plot(t, self.log_k_est_list[:,3], label="k4")
            self.axs.legend()
            self.axs.set_ylim(0,1)

if __name__ == "__main__":
    pf1 = ParticleFilterCPP(250, 25, 0.005, log=True)
    obs = np.array([0,0,2,
                    0,0,0,
                    1,0,0,0,
                    0,0,0])
    state = np.zeros((17,250),dtype=np.float64)
    pf1.set_init_state(obs)
    pf1.pf.GetStateMatrix(state)
    print(state[:,:2].T)
