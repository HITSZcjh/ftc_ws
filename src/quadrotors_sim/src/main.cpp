#include <iostream>
#include "quadrotors_sim/vec_env.hpp"
#include <iomanip> // 包含 iomanip 库

using namespace quadrotors;
int main()
{

    Simulator simulator;
    std::cout << simulator.integrator.param;
    Vector<NU> u={1,1,1,0};
    std::cout << std::fixed << std::setprecision(2);

    for(int i = 0;i<100000;i++)
    {
        std::cout<<i<<std::endl;
        simulator.step(u);
        std::cout<<"x:"<<simulator.x.transpose() << std::endl;
    }
    
}