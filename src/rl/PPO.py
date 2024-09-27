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
        self.learning_rate = 5e-3
        self.n_step = 125
        self.batchsize = self.n_step*self.env.num_envs
        self.eval_bathsize = 100000
        self.replay_buffer = ReplayBuffer(max_size=self.batchsize)
        self.epochs = 50
        self.num_mini_batches = 10
        self.eps = 0.2
        self.n_iter = 10000
        self.logger = Logger(logdir)
        self.actor = MLPPolicy(self.env.action_dim, self.env.state_dim, 2, 64, self.learning_rate)
        self.critic = BootstrappedContinuousCritic(self.env.action_dim, self.env.state_dim, 2, 64, self.learning_rate, gamma=0.99, 
                                                   gae_lambda=0.99, num_target_updates=10, num_grad_steps_per_target_update=10)

        self.schedule = "adaptive"
        self.desired_kl = 0.01
        self.max_grad_norm = 1.0
        self.entropy_coef = 1e-10

        self.save_actor_model_path = save_actor_model_path
        self.save_critic_model_path = save_critic_model_path

        self.curriculum_cnt = 0
        self.curriculum_length = 0
    def load_model(self, load_actor_model_path, load_critic_model_path, num):
        self.actor.load_state_dict(torch.load(load_actor_model_path+'/%d.pt' %num))
        self.critic.load_state_dict(torch.load(load_critic_model_path+'/%d.pt' %num))

    def actor_update(self, states:np.ndarray, actions:np.ndarray, advantages, old_log_probs, old_sigma, old_mu):
        old_log_probs = old_log_probs.reshape(-1)
        old_sigma = old_sigma.reshape(-1, self.env.action_dim)
        old_mu = old_mu.reshape(-1, self.env.action_dim)
        old_log_probs = ptu.from_numpy(old_log_probs)
        old_sigma = ptu.from_numpy(old_sigma)
        old_mu = ptu.from_numpy(old_mu)

        for _ in range(self.epochs):

            action_distr = self.actor.forward(ptu.from_numpy(states))
            log_probs = action_distr.log_prob(ptu.from_numpy(actions))

            if self.schedule == "adaptive":
                sigma_batch = action_distr.stddev
                mu_batch = action_distr.mean

                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma + 1.0e-5)
                        + (torch.square(old_sigma) + torch.square(old_mu - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(5e-3, self.learning_rate * 1.5)

                    for param_group in self.actor.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

                    for param_group in self.critic.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate/2


            ratio = torch.exp(log_probs-old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps)*advantages
            ppo_loss = torch.mean(-torch.min(surr1,surr2))

            # entropy_loss = -self.entropy_coef*torch.mean(action_distr.entropy().sum(dim=-1))
            actor_loss = ppo_loss

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor.optimizer.step()        

        return ppo_loss

    def update(self, states, actions, rewards, next_states, dones, old_log_probs, old_sigma, old_mu):
        # 948
        advantages = self.critic.estimate_advantage(states, next_states, rewards, dones)
        # 2522
        advantages = advantages.reshape(-1)

        states = states.reshape(-1, self.env.state_dim)
        next_states = next_states.reshape(-1, self.env.state_dim)
        actions = actions.reshape(-1, self.env.action_dim)
        rewards = rewards.reshape(-1)
        dones = dones.reshape(-1)
        old_log_probs = old_log_probs.reshape(-1)
        old_sigma = old_sigma.reshape(-1, self.env.action_dim)
        old_mu = old_mu.reshape(-1, self.env.action_dim)


        states = ptu.from_numpy(states)
        next_states = ptu.from_numpy(next_states)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)
        dones = ptu.from_numpy(dones)
        rewards = ptu.from_numpy(rewards)
        old_log_probs = ptu.from_numpy(old_log_probs)
        old_sigma = ptu.from_numpy(old_sigma)
        old_mu = ptu.from_numpy(old_mu)


        mini_batch_size = self.batchsize // self.num_mini_batches
        indices = torch.randperm(self.num_mini_batches * mini_batch_size, requires_grad=False, device=ptu.device)
        
        for _ in range(self.epochs):
            for i in range(self.num_mini_batches):
                ind = indices[i * mini_batch_size : (i + 1) * mini_batch_size]
                states_batch = states[ind]
                action_batch = actions[ind]
                old_log_probs_batch = old_log_probs[ind]
                old_sigma_batch = old_sigma[ind]
                old_mu_batch = old_mu[ind]
                next_states_batch = next_states[ind]
                dones_batch = dones[ind]
                rewards_batch = rewards[ind]

                advantage_batch = advantages[ind]

                action_distr = self.actor.forward(states_batch)
                log_probs = action_distr.log_prob(action_batch)

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
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(5e-3, self.learning_rate * 1.5)

                        for param_group in self.actor.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate

                        for param_group in self.critic.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate/2

                ratio = torch.exp(log_probs-old_log_probs_batch)
                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps)*advantage_batch
                ppo_loss = torch.mean(-torch.min(surr1,surr2))

                # entropy_loss = -self.entropy_coef*torch.mean(action_distr.entropy().sum(dim=-1))
                actor_loss = ppo_loss

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor.optimizer.step()

                value_s_prime_batch = self.critic.forward(next_states_batch)
                targets_batch = rewards_batch + self.critic.gamma * value_s_prime_batch * (1-dones_batch)
            
                predictions_batch = self.critic.forward(states_batch)

                critic_loss = self.critic.loss(predictions_batch, targets_batch)
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic.optimizer.step()

        return actor_loss, critic_loss


    def perform_logging(self, itr, epreturn, eplen, extra_infos, loss):
        logs = OrderedDict()
        logs["Train/fps"] = (self.n_step*self.env.num_envs)/(time.time() - self.start_time)
        logs["Train/curriculum_level"] = self.env.curriculum_level

        # extra_infos
        if extra_infos is not None:
            for key in extra_infos[0].keys():
                logs["Extra_infos/"+key] = np.mean([info[key] for info in extra_infos])

        # Loss
        for key, value in loss.items():
            logs["Loss/"+key] = value

        clear_flag = False
        self.curriculum_length += 1
        if len(epreturn) > int(self.env.num_envs/2):
            epreturn = np.array(epreturn)
            eplen = np.array(eplen)
            logs["Train/averageReturn"] = np.mean(epreturn)
            logs["Train/stdReturn"] = np.std(epreturn)
            logs["Train/maxReturn"] = np.max(epreturn)
            logs["Train/minReturn"] = np.min(epreturn)
            logs["Train/minEpLen"] = np.min(eplen)
            logs["Train/maxEpLen"] = np.max(eplen)
            logs["Train/averageEpLen"] = np.mean(eplen)
            logs["Train/learning_rate"] = self.learning_rate
            clear_flag = True
            # self.env.set_baseline(logs["Train/averageReturn"], logs["Train/stdReturn"])

            # if(logs["Train/averageEpLen"]>800 and self.env.curriculum_level<5 and self.curriculum_length > 50):
            #     if self.curriculum_cnt>5:
            #         self.curriculum_cnt = 0
            #         self.curriculum_length = 0
            #         self.env.curriculum_level += 1
            #         self.env.set_curriculum_level()
            #         self.env.rewards = [[] for _ in range(self.env.num_envs)]
            #         self.env.epreturn = []
            #         self.env.eplen = []
            #         self.env.extra_infos = []
            #     else:
            #         self.curriculum_cnt += 1
            # else:
            #     self.curriculum_cnt = 0

        # perform the logging
        for key, value in logs.items():
            # print('{} : {}'.format(key, value))
            self.logger.log_scalar(value, key, itr)

        self.logger.flush()
        return clear_flag
    
    def train_loop(self):
        for itr in range(self.n_iter):
            self.start_time = time.time()
            print("collecting train data...\n")

            # path = sample_trajectories(self.env, self.actor, self.batchsize)
            # self.replay_buffer.add_rollouts(path)
            # states, actions, rewards, next_states, dones = self.replay_buffer.sample_recent_data(self.batchsize)
            # 578
            states, actions, rewards, next_states, dones, log_probs, stds, means = run_steps(self.env, self.actor, self.n_step)
            
            # 948
            print("Training model...\n")
            
            # actor_loss, critic_loss = self.update(states, actions, rewards, next_states, dones, log_probs, stds, means)
            # loss = {"critic_loss":critic_loss, "actor_loss":actor_loss}

            advantages = self.critic.estimate_advantage(states, next_states, rewards, dones) 
            # 2522
            advantages = advantages.reshape(-1)
            advantages = ptu.from_numpy(advantages)

            states = states.reshape(-1, self.env.state_dim)
            next_states = next_states.reshape(-1, self.env.state_dim)
            actions = actions.reshape(-1, self.env.action_dim)
            rewards = rewards.reshape(-1)
            dones = dones.reshape(-1)
            critic_loss = self.critic.update(states, next_states, rewards, dones)
            # 3020
            actor_loss = self.actor_update(states, actions, advantages, log_probs, stds, means)
            # 4494
            loss = {"critic_loss":critic_loss, "actor_loss":actor_loss}


            print("Perform the logging...\n")
            if(self.perform_logging(itr, self.env.epreturn, self.env.eplen, self.env.extra_infos, loss)):
                self.env.epreturn = []
                self.env.eplen = []
            self.env.extra_infos = []

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
load_actor_model_path = "/home/jiao/test_ws/src/rl/model/actor_model01-08-2024_12-06-12"
load_critic_model_path = "/home/jiao/test_ws/src/rl/model/critic_model01-08-2024_12-06-12"
num = 710

if __name__ == "__main__":
    if(test):
        ppo = PPO(None, None, None, True)
        ppo.load_model(load_actor_model_path, load_critic_model_path, 360)
        ppo.eval_plot()
    else:
        log_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), './without_fault/log_data2')
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

