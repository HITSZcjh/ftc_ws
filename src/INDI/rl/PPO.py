from MLP_policy import MLPPolicy
from bootstrapped_continuous_critic import BootstrappedContinuousCritic
from replay_buffer import ReplayBuffer
from utils import *
from rl_model import RLModel
import torch
import pytorch_util as ptu
from logger import Logger
from collections import OrderedDict
import time
import os
import matplotlib.pyplot as plt
np.set_printoptions(precision=2,suppress=True,threshold=np.inf)

class PPO:
    def __init__(self, logdir, save_actor_model_path, save_critic_model_path, use_gpu) -> None:
        ptu.init_gpu(use_gpu=use_gpu)
        self.env = RLModel()
        self.learning_rate = 1e-4
        self.actor = MLPPolicy(self.env.action_dim, self.env.state_dim, 2, 128, self.learning_rate)
        self.critic = BootstrappedContinuousCritic(self.env.action_dim, self.env.state_dim, 2, 128, self.learning_rate, gamma=0.99, gae_lambda=0.95)
        self.n_step = 100
        self.batchsize = 1600000
        self.eval_bathsize = 100000
        self.replay_buffer = ReplayBuffer(max_size=self.batchsize)
        self.epochs = 5
        self.eps = 0.2
        self.n_iter = 10000
        self.logger = Logger(logdir)

        self.schedule = "deadaptive"
        self.desired_kl = 0.01
        self.max_grad_norm = 1.0
        self.entropy_coef = 0.01

        self.save_actor_model_path = save_actor_model_path
        self.save_critic_model_path = save_critic_model_path

    def load_model(self, load_actor_model_path, load_critic_model_path, num):
        self.actor.load_state_dict(torch.load(load_actor_model_path+'/%d.pt' %num))
        self.critic.load_state_dict(torch.load(load_critic_model_path+'/%d.pt' %num))

    def actor_update(self, states:np.ndarray, actions:np.ndarray, rewards:np.ndarray, dones:np.ndarray, advantages):
        old_action_distr = self.actor.forward(ptu.from_numpy(states))
        old_log_probs = old_action_distr.log_prob(ptu.from_numpy(actions)).detach()

        old_sigma_batch = old_action_distr.stddev
        old_mu_batch = old_action_distr.mean
        for _ in range(self.epochs):

            action_distr = self.actor.forward(ptu.from_numpy(states))
            log_probs = action_distr.log_prob(ptu.from_numpy(actions))

            if self.schedule == "adaptive":
                sigma_batch = action_distr.stddev
                mu_batch = action_distr.mean

                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-6, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-3, self.learning_rate * 1.5)

                    for param_group in self.actor.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

                    for param_group in self.critic.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate
                    print(self.learning_rate)

            ratio = torch.exp(log_probs-old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps)*advantages
            actor_loss = torch.mean(-torch.min(surr1,surr2))

            # entropy_loss = -self.entropy_coef*torch.mean(action_distr.entropy().sum(dim=-1))
            # actor_loss += entropy_loss

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor.optimizer.step()        

    def perform_logging(self, itr, epreturn, eplen):
        logs = OrderedDict()
        clear_flag = False
        if len(epreturn) > 100:
            epreturn = np.array(epreturn)
            eplen = np.array(eplen)
            logs["Train_AverageReturn"] = np.mean(epreturn)
            logs["Train_StdReturn"] = np.std(epreturn)
            logs["Train_MaxReturn"] = np.max(epreturn)
            logs["Train_MinReturn"] = np.min(epreturn)
            logs["Train_MinEpLen"] = np.min(eplen)
            logs["Train_MaxEpLen"] = np.max(eplen)
            logs["Train_AverageEpLen"] = np.mean(eplen)
            logs["TimeSinceStart"] = time.time() - self.start_time
            clear_flag = True
        else:
            logs["TimeSinceStart"] = time.time() - self.start_time
        # perform the logging
        for key, value in logs.items():
            # print('{} : {}'.format(key, value))
            self.logger.log_scalar(value, key, itr)

        self.logger.flush()
        return clear_flag
    
    def train_loop(self):
        self.start_time = time.time()
        for itr in range(self.n_iter):
            print("collecting train data...\n")
            states, actions, rewards, next_states, dones = run_steps(self.env, self.actor, self.n_step)
            print("Training model...\n")
            
            # print(path[0]["states"])
            # print(path[0]["actions"])
            advantages = self.critic.estimate_advantage(states, next_states, rewards, dones)
            advantages = advantages.reshape(-1)

            advantages = ptu.from_numpy(advantages)

            states = states.reshape(-1, self.env.state_dim)

            next_states = next_states.reshape(-1, self.env.state_dim)
            actions = actions.reshape(-1, self.env.action_dim)
            rewards = rewards.reshape(-1)
            dones = dones.reshape(-1)
            self.actor_update(states, actions, rewards, dones, advantages)
            self.critic.update(states, next_states, rewards, dones)

            # print("collecting eval data...\n")
            # eval_paths = sample_trajectories(self.env, self.actor, self.eval_bathsize)
            print("Perform the logging...\n")
            if(self.perform_logging(itr, self.env.epreturn, self.env.eplen)):
                self.env.epreturn = []
                self.env.eplen = []

            if(itr%10 == 0):
                print("Saving model...\n")
                # print(eval_paths[0]["rewards"])
                self.actor.save(self.save_actor_model_path+'/%d.pt' %itr)
                self.critic.save(self.save_critic_model_path+'/%d.pt' %itr)
    
    def eval_plot(self):
        paths = sample_n_trajectories(self.env, self.actor, 2)
        for i in range(len(paths)):
            print(paths[i])
            states = paths[i]["states"]
            fig, axes = plt.subplots(states.shape[1],1)
            for j in range(states.shape[1]):
                axes[j].plot(states[:,j])
        plt.show()
    
    def get_path(self):
        return sample_trajectory(self.env, self.actor)
    
test = False

load_model = False
load_actor_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/actor_model11-03-2024_00-21-05"
load_critic_model_path = "/home/jiao/ftc_ws/src/INDI/rl/model/critic_model11-03-2024_00-21-05"
num = 300

if __name__ == "__main__":
    if(test):
        ppo = PPO(None, None, None, True)
        ppo.load_model(load_actor_model_path, load_critic_model_path, 360)
        ppo.eval_plot()
    else:
        log_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), './log_data1')
        if not (os.path.exists(log_data_path)):
            os.makedirs(log_data_path)
        logdir = "rl_model" + time.strftime("%d-%m-%Y_%H-%M-%S")
        logdir = os.path.join(log_data_path, logdir)
        
        save_actor_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), './model/actor_model') + time.strftime("%d-%m-%Y_%H-%M-%S")
        save_critic_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), './model/critic_model') + time.strftime("%d-%m-%Y_%H-%M-%S")
        if not (os.path.exists(save_actor_model_path)):
            os.makedirs(save_actor_model_path)
        if not (os.path.exists(save_critic_model_path)):
            os.makedirs(save_critic_model_path)
        ppo = PPO(logdir, save_actor_model_path, save_critic_model_path, True)

        if load_model:
            ppo.load_model(load_actor_model_path, load_critic_model_path, num)

        ppo.train_loop()

