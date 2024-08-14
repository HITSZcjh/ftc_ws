import numpy as np
import copy
import torch

def Path(states, actions, rewards, next_states, dones):
    return {"states" : np.array(states, dtype=np.float32),
            "rewards" : np.array(rewards, dtype=np.float32),
            "actions" : np.array(actions, dtype=np.float32),
            "next_states": np.array(next_states, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32)}

def get_pathlength(path):
    return len(path["rewards"])

def sample_trajectory(env, policy):
    # TODO: get this from hw1 or hw2
    # initialize env for the beginning of a new rollout
    state = env.reset() # HINT: should be the output of resetting the env

    # init vars
    states, actions, rewards, next_states, dones = [], [], [], [], []
    steps = 0
    while True:
        # use the most recent ob to decide what to do
        states.append(state)
        action = policy.get_action(state) # HINT: query the policy's get_action function
        action = action[0]
        actions.append(action)

        # take that action and record results
        state, reward, done = env.step(action)

        # record result of taking that action
        steps += 1
        next_states.append(state)
        rewards.append(reward)
        dones.append(done)

        if done:
            break

    return Path(states, actions, rewards, next_states, dones)

def sample_steps(env, policy, steps):
    state = env.reset()
    states, actions, rewards, next_states, dones = [], [], [], [], []
    for step in range(steps):
        states.append(state)
        action = policy.get_action(state)
        actions.append(action)
        state, reward, done = env.step(action)
        next_states.append(state)
        rewards.append(reward)
        dones.append(done)
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_states = np.array(next_states)
    dones = np.array(dones)

    states_list = []
    actions_list = []
    rewards_list = []
    next_states_list = []
    dones_list = []
    for i in range(env.num_envs):
        last_true_index = np.where(dones[:,i] == True)[0][-1]
        
        states_list.append(states[:last_true_index+1,i])
        actions_list.append(actions[:last_true_index+1,i])
        rewards_list.append(rewards[:last_true_index+1,i])
        next_states_list.append(next_states[:last_true_index+1,i])
        dones_list.append(dones[:last_true_index+1,i])
    
    states_list = np.concatenate(states_list)
    actions_list = np.concatenate(actions_list)
    rewards_list = np.concatenate(rewards_list)
    next_states_list = np.concatenate(next_states_list)
    dones_list = np.concatenate(dones_list)

    batch = states_list.shape[0]
    index = np.where(dones_list == True)[0]
    index = np.insert(index, 0, 0)
    paths = [{"states":states_list[index[i]:index[i+1]+1],
              "actions":actions_list[index[i]:index[i+1]+1],
              "rewards":rewards_list[index[i]:index[i+1]+1],
              "next_states":next_states_list[index[i]:index[i+1]+1],
              "dones":dones_list[index[i]:index[i+1]+1]} for i in range(index.shape[0]-1)]

    return paths, batch

def sample_trajectories(env, policy, min_timesteps_per_batch):
    """
        Collect rollouts using policy
        until we have collected min_timesteps_per_batch steps
    """
    # TODO: get this from hw1 or hw2


    num = int(1.2*min_timesteps_per_batch/env.num_envs)
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        path, batch = sample_steps(env, policy, num)
        timesteps_this_batch += batch
        paths += path
    return paths

def sample_n_trajectories(env, policy, ntraj):
    # TODO: get this from hw1
    paths = []

    for _ in range(ntraj):
        path = sample_trajectory(env=env, policy=policy)
        paths.append(path)

    return paths

def run_steps(env, policy, num_steps):
    state = env.get_obs()
    states, actions, rewards, next_states, dones, log_probs, stds, means = [], [], [], [], [], [], [], []
    for _ in range(num_steps):
        states.append(state)
        action, log_prob, std, mean = policy.get_action(state)
        actions.append(action)
        state, reward, done = env.step(action)
        next_states.append(state)
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob)
        stds.append(std)
        means.append(mean)
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_states = np.array(next_states)
    dones = np.array(dones)
    log_probs = np.array(log_probs)
    stds = np.array(stds)
    means = np.array(means)
    return states, actions, rewards, next_states, dones, log_probs, stds, means

def convert_listofrollouts(paths):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    states = np.concatenate([path["states"] for path in paths])
    actions = np.concatenate([path["actions"] for path in paths])
    next_states = np.concatenate([path["next_states"] for path in paths])
    dones = np.concatenate([path["dones"] for path in paths])
    concatenated_rewards = np.concatenate([path["rewards"] for path in paths])
    unconcatenated_rewards = [path["rewards"] for path in paths]
    return states, actions, next_states, dones, concatenated_rewards, unconcatenated_rewards

def normalize(data, mean, std, eps=1e-8):
    return (data-mean)/(std+eps)

def unnormalize(data, mean, std):
    return data*std+mean


def add_noise(data_inp, noiseToSignal=0.01):

    data = copy.deepcopy(data_inp) #(num data points, dim)

    #mean of data
    mean_data = np.mean(data, axis=0)

    #if mean is 0,
    #make it 0.001 to avoid 0 issues later for dividing by std
    mean_data[mean_data == 0] = 0.000001

    #width of normal distribution to sample noise from
    #larger magnitude number = could have larger magnitude noise
    std_of_noise = mean_data * noiseToSignal
    for j in range(mean_data.shape[0]):
        data[:, j] = np.copy(data[:, j] + np.random.normal(
            0, np.absolute(std_of_noise[j]), (data.shape[0],)))

    return data