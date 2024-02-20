cmake_minimum_required(VERSION 3.0.2)
project(INDI)

add_subdirectory(third_party/yaml-cpp)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
)
find_package(Eigen3)
find_package(OpenMP)
catkin_package()

# include_directories(

# )

add_library(QuadrotorEnv_lib src/uav_simulator.cpp src/vec_env.cpp)

target_include_directories(QuadrotorEnv_lib PUBLIC
  include
  ${PROJECT_SOURCE_DIR}/third_party/yaml-cpp/include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  $ENV{ACADOS_SOURCE_DIR}/include
  $ENV{ACADOS_SOURCE_DIR}/include/blasfeo/include
  $ENV{ACADOS_SOURCE_DIR}/include/hpipm/include
  ${PROJECT_SOURCE_DIR}/scripts/c_generated_code/sim_UAVModel
)

target_link_directories(QuadrotorEnv_lib PUBLIC
${catkin_LIB_DIRS}
$ENV{ACADOS_SOURCE_DIR}/lib
${PROJECT_SOURCE_DIR}/scripts/c_generated_code/sim_UAVModel)

target_link_libraries(QuadrotorEnv_lib PUBLIC 
  acados_sim_solver_UAVModel 
  acados 
  hpipm 
  blasfeo 
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
pybind11_add_module(QuadrotorEnv src/pybind_wrapper.cpp)
target_link_libraries(QuadrotorEnv PUBLIC QuadrotorEnv_lib)
set_target_properties(QuadrotorEnv PROPERTIES SUFFIX ".so")



add_executable(test1 src/test.cpp src/uav_simulator.cpp) 
target_include_directories(test1 PUBLIC
  include
  ${PROJECT_SOURCE_DIR}/third_party/yaml-cpp/include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  $ENV{ACADOS_SOURCE_DIR}/include
  $ENV{ACADOS_SOURCE_DIR}/include/blasfeo/include
  $ENV{ACADOS_SOURCE_DIR}/include/hpipm/include
  ${PROJECT_SOURCE_DIR}/scripts/c_generated_code/sim_UAVModel
)
target_link_directories(test1 PUBLIC
${catkin_LIB_DIRS}
$ENV{ACADOS_SOURCE_DIR}/lib
${PROJECT_SOURCE_DIR}/scripts/c_generated_code/sim_UAVModel)
target_link_libraries(test1 PUBLIC 
  acados_sim_solver_UAVModel 
  acados 
  hpipm 
  blasfeo 
  ${catkin_LIBRARIES} 
  OpenMP::OpenMP_CXX 
  yaml-cpp
)