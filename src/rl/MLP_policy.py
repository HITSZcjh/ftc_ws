from torch import nn, optim, distributions
from torch.nn import functional as F
import pytorch_util as ptu
import torch
import itertools
import numpy as np


class MLPPolicy(nn.Module):
    def __init__(self, action_dim, state_dim, n_layers, layer_size, learning_rate) -> None:
        super().__init__()

        self.action_dim = action_dim
        self.state_dim = state_dim

        self.mean_net = ptu.build_mlp(input_size=self.state_dim,
                                    output_size=self.action_dim,
                                    n_layers=n_layers, size=layer_size,output_activation='tanh')
        self.logstd = nn.Parameter(
            torch.zeros(self.action_dim, dtype=torch.float32, device=ptu.device)
        )
        self.mean_net.to(ptu.device)
        self.logstd.to(ptu.device)
        # self.parameter = itertools.chain([self.logstd], self.mean_net.parameters())
        self.optimizer = optim.Adam(
            self.parameters(),
            learning_rate
        )

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def get_action(self, state: np.ndarray):
        # TODO: get this from HW1
        if len(state.shape) > 1:
            state = state
        else:
            state = state[None]

        state = ptu.from_numpy(state)
        distr = self.forward(state)
        act = distr.sample()
        log_prob = distr.log_prob(act)
        return ptu.to_numpy(act), ptu.to_numpy(log_prob), ptu.to_numpy(distr.stddev), ptu.to_numpy(distr.mean)

    def get_action_without_sample(self, state):
        if len(state.shape) > 1:
            state = state
        else:
            state = state[None]

        state = ptu.from_numpy(state)
        act = self.mean_net(state)
        return ptu.to_numpy(act)

    def forward(self, state: torch.FloatTensor):
        batch_mean = self.mean_net(state)
        scale_tril = torch.diag(torch.exp(self.logstd))
        batch_dim = batch_mean.shape[0]
        batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
        # print(self.logstd)
        action_distribution = distributions.MultivariateNormal(
            batch_mean,
            scale_tril=batch_scale_tril,
        )
        return action_distribution



        # batch_dim = batch_mean.shape[0]
        # batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)

