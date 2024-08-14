cmake_minimum_required(VERSION 3.0.2)
project(quadrotors_sim)

set(CMAKE_BUILD_TYPE "Release")

add_subdirectory(third_party/yaml-cpp)

find_package(catkin REQUIRED COMPONENTS
  roscpp
)
find_package(OpenMP)
find_package(Eigen3)

catkin_package()

add_library(MyQuadrotorEnv_lib src/simulator.cpp src/vec_env.cpp) 
target_include_directories(MyQuadrotorEnv_lib PUBLIC
  include
  ${PROJECT_SOURCE_DIR}/third_party/yaml-cpp/include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)
target_link_libraries(MyQuadrotorEnv_lib PUBLIC 
  ${catkin_LIBRARIES} 
  OpenMP::OpenMP_CXX 
  yaml-cpp
)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.6.2
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(pybind11)
pybind11_add_module(MyQuadrotorEnv src/pybind_wrapper.cpp)
target_link_libraries(MyQuadrotorEnv PUBLIC MyQuadrotorEnv_lib)
# target_include_directories(MyQuadrotorEnv PUBLIC
#   include
#   ${PROJECT_SOURCE_DIR}/third_party/yaml-cpp/include
#   ${catkin_INCLUDE_DIRS}
#   ${EIGEN3_INCLUDE_DIRS}
# )
set_target_properties(MyQuadrotorEnv PROPERTIES SUFFIX ".so")