from utils import *


class ReplayBuffer(object):

    def __init__(self, max_size=1000000):

        self.max_size = max_size
        self.paths = []
        self.states = None
        self.actions = None
        self.concatenated_rewards = None
        self.unconcatenated_rewards = None
        self.next_states = None
        self.dones = None

    def add_rollouts(self, paths, noised=False):

        # add new rollouts into our list of rollouts
        for path in paths:
            self.paths.append(path)

        # convert new rollouts into their component arrays, and append them onto our arrays
        states, actions, next_states, dones, concatenated_rewards, unconcatenated_rewards = convert_listofrollouts(paths)

        if noised:
            states = add_noise(states)
            next_states = add_noise(next_states)

        if self.states is None:
            self.states = states[-self.max_size:]
            self.actions = actions[-self.max_size:]
            self.next_states = next_states[-self.max_size:]
            self.dones = dones[-self.max_size:]
            self.concatenated_rewards = concatenated_rewards[-self.max_size:]
            self.unconcatenated_rewards = unconcatenated_rewards[-self.max_size:]
        else:
            self.states = np.concatenate([self.states, states])[-self.max_size:]
            self.actions = np.concatenate([self.actions, actions])[-self.max_size:]
            self.next_states = np.concatenate(
                [self.next_states, next_states]
            )[-self.max_size:]
            self.dones = np.concatenate(
                [self.dones, dones]
            )[-self.max_size:]
            self.concatenated_rewards = np.concatenate(
                [self.concatenated_rewards, concatenated_rewards]
            )[-self.max_size:]
            if isinstance(unconcatenated_rewards, list):
                self.unconcatenated_rewards += unconcatenated_rewards  # TODO keep only latest max_size around
            else:
                self.unconcatenated_rewards.append(unconcatenated_rewards)  # TODO keep only latest max_size around

    ########################################
    ########################################

    def sample_random_rollouts(self, num_rollouts):
        rand_indices = np.random.permutation(len(self.paths))[:num_rollouts]
        return self.paths[rand_indices]

    def sample_recent_rollouts(self, num_rollouts=1):
        return self.paths[-num_rollouts:]

    ########################################
    ########################################

    def sample_random_data(self, batch_size):

        assert self.states.shape[0] == self.actions.shape[0] == self.concatenated_rewards.shape[0] == self.next_states.shape[0] == self.dones.shape[0]
        rand_indices = np.random.permutation(self.states.shape[0])[:batch_size]
        return self.states[rand_indices], self.actions[rand_indices], self.concatenated_rewards[rand_indices], self.next_states[rand_indices], self.dones[rand_indices]

    def sample_recent_data(self, batch_size=1, concat_rew=True):

        if concat_rew:
            return self.states[-batch_size:], self.actions[-batch_size:], self.concatenated_rewards[-batch_size:], self.next_states[-batch_size:], self.dones[-batch_size:]
        else:
            num_recent_rollouts_to_return = 0
            num_datapoints_so_far = 0
            index = -1
            while num_datapoints_so_far < batch_size:
                recent_rollout = self.paths[index]
                index -=1
                num_recent_rollouts_to_return +=1
                num_datapoints_so_far += get_pathlength(recent_rollout)
            rollouts_to_return = self.paths[-num_recent_rollouts_to_return:]
            states, actions, next_states, dones, concatenated_rewards, unconcatenated_rewards = convert_listofrollouts(rollouts_to_return)
            return states, actions, unconcatenated_rewards, next_states, dones