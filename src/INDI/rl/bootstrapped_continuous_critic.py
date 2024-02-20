from torch import nn
from torch import optim
import torch
import pytorch_util as ptu
import numpy as np
import utils
class BootstrappedContinuousCritic(nn.Module):
    """
        Notes on notation:

        Prefixes and suffixes:
        ob - observation
        ac - action
        _no - this tensor should have shape (batch self.size /n/, observation dim)
        _na - this tensor should have shape (batch self.size /n/, action dim)
        _n  - this tensor should have shape (batch self.size /n/)

        Note: batch self.size /n/ is defined at runtime.
        is None
    """
    def __init__(self, action_dim, state_dim, n_layers, layer_size, learning_rate=1e-4, gamma=0.99, gae_lambda = 0.99,num_target_updates=10, num_grad_steps_per_target_update=10) -> None:
        super().__init__()
        self.state_dim = state_dim

        # critic parameters
        self.num_target_updates = num_target_updates
        self.num_grad_steps_per_target_update = num_grad_steps_per_target_update
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.critic_network = ptu.build_mlp(
            self.state_dim,
            1,
            n_layers=n_layers,
            size=layer_size,
        )
        self.critic_network.to(ptu.device)
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.critic_network.parameters(),
            learning_rate,
        )

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def forward(self, states):
        return self.critic_network(states).squeeze(1)

    def forward_np(self, states):
        states = ptu.from_numpy(states)
        predictions = self(states)
        return ptu.to_numpy(predictions)

    def update(self, states, next_states, rewards, dones):
        """
            Update the parameters of the critic.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                states: shape: (sum_of_path_lengths, ob_dim)
                next_states: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                rewards: length: sum_of_path_lengths. Each element in rewards is a scalar containing
                    the reward for each timestep
                dones: length: sum_of_path_lengths. Each element in dones is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                training loss
        """
        # TODO: Implement the pseudocode below: do the following (
        # self.num_grad_steps_per_target_update * self.num_target_updates)
        # times:
        # every self.num_grad_steps_per_target_update steps (which includes the
        # first step), recompute the target values by
        #     a) calculating V(s') by querying the critic with next_states
        #     b) and computing the target values as r(s, a) + gamma * V(s')
        # every time, update this critic using the observations and targets
        #
        # HINT: don't forget to use dones to cut off the V(s') (ie set it
        #       to 0) when a terminal state is reached
        # HINT: make sure to squeeze the output of the critic_network to ensure
        #       that its dimensions match the reward

        targets = None

        for i in range(self.num_grad_steps_per_target_update * self.num_target_updates):
            if i % self.num_grad_steps_per_target_update == 0:
                value_s_prime = self.forward_np(next_states)
                targets = rewards + self.gamma * value_s_prime * (1-dones)
                targets = ptu.from_numpy(targets)
            
            predictions = self.forward(ptu.from_numpy(states))

            assert predictions.shape == targets.shape
            loss = self.loss(predictions, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def estimate_advantage(self, states:np.ndarray, rewards:np.ndarray, dones:np.ndarray):
        values = self.forward_np(states)
        values = np.append(values, [0])
        batch_size = states.shape[0]
        advantages = np.zeros(batch_size + 1)

        for i in reversed(range(batch_size)):
            ## TODO: recursively compute advantage estimates starting from
                ## timestep T.
            ## HINT: use terminals to handle edge cases. terminals[i]
                ## is 1 if the state is the last in its trajectory, and
                ## 0 otherwise.
            if dones[i]:
                delta = rewards[i]
                advantages[i] = delta
            else:
                delta = rewards[i] + self.gamma * values[i+1] - values[i]
                advantages[i] = delta + advantages[i+1] * self.gae_lambda * self.gamma
        # remove dummy advantage
        advantages = advantages[:-1]
        advantages = utils.normalize(advantages,np.mean(advantages),np.std(advantages))
        return advantages