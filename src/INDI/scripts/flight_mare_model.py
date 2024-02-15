from flightgym import QuadrotorEnv_v1
from ruamel.yaml import YAML, dump, RoundTripDumper
import os

if __name__ == "__main__":
    cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] +
                           "/flightlib/configs/vec_env.yaml", 'r'))
    model = QuadrotorEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    print(model.getObsDim())
    print(model.getActDim())