from flightgym import QuadrotorEnv_v1
from ruamel.yaml import YAML, dump, RoundTripDumper
import os
import numpy as np
import time

if __name__ == "__main__":
    cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] +
                           "/flightlib/configs/vec_env.yaml", 'r'))
    model = QuadrotorEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    num_obs = model.getObsDim()
    num_acts = model.getActDim()
    num_envs = model.getNumOfEnvs()
    _observation = np.zeros([num_envs, num_obs],
                                    dtype=np.float32)

    _reward = np.zeros(num_envs, dtype=np.float32)
    _done = np.zeros((num_envs), dtype=bool)
    _extraInfoNames = model.getExtraInfoNames()
    _extraInfo = np.zeros([num_envs,
                                len(_extraInfoNames)], dtype=np.float32)
    rewards = [[] for _ in range(num_envs)]
    start_time = time.perf_counter()
    for i in range(3000):
        model.step(np.zeros((num_envs,num_acts), dtype=np.float32), _observation,
                            _reward, _done, _extraInfo)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")
