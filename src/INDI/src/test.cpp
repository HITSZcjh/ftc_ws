#include "INDI/uav_simulator.hpp"
using namespace QuadrotorEnv;
int main()
{
    Simulator sim(0.01, YAML::LoadFile("/home/jiao/ftc_ws/src/INDI/configs/vec_env.yaml"));
    for(int i=0;i<1;i++)
        sim.test();
    return 0;
}