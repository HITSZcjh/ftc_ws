#include <iostream>
#include "quadrotors_sim/simulator.hpp"

using namespace quadrotors;
int main()
{
    Simulator simulator;
    Vector<NU> u={1,1,1,1};
    for(int i = 0;i<100;i++)
    {
        simulator.step(u);
        std::cout<<"x:"<<simulator.x << std::endl;
    }
    
}