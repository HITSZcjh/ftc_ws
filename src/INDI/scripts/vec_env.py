from QuadrotorEnv import VecEnv
import numpy as np
import time
if __name__ == "__main__":
    env = VecEnv()
    num_envs = env.get_num_envs()
    num_obs = env.get_obs_dim()
    num_acts = env.get_action_dim()
    _observation = np.zeros([num_envs, num_obs],
                                    dtype=np.float64)
    _action = np.ones([num_envs, num_acts], dtype=np.float64)
    _reward = np.zeros(num_envs, dtype=np.float64)
    _done = np.zeros((num_envs), dtype=bool)
    start_time = time.perf_counter()
    env.reset(_observation)
    
    action_list = []
    reward_list = []
    obs_list = []
    done_list = [] 
    for i in range(10000):
        env.step(_action, _observation, _reward, _done)
        action_list.append(_action.copy())
        reward_list.append(_reward.copy())
        obs_list.append(_observation.copy())
        done_list.append(_done.copy())

    action_list = np.array(action_list)
    reward_list = np.array(reward_list)
    obs_list = np.array(obs_list)
    done_list = np.array(done_list)
    acts = []
    rewards = []
    obs = []
    dones = []
    for i in range(num_envs):
        last_true_index = np.where(done_list[:,i] == True)[0][-1]
        acts.append(action_list[:last_true_index+1,i])
        rewards.append(reward_list[:last_true_index+1,i])
        obs.append(obs_list[:last_true_index+1,i])
        dones.append(done_list[:last_true_index+1,i])
    acts = np.concatenate(acts)
    rewards = np.concatenate(rewards)
    obs = np.concatenate(obs)
    dones = np.concatenate(dones)
    print(dones)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")