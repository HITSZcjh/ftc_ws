cmake_minimum_required(VERSION 3.0.2)
project(fdd)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  INDI
)
find_package(Eigen3)
find_package(OpenMP)
catkin_package()

# include_directories(

# )

add_library(fdd_particle_filter_lib src/particle_filter.cpp src/uav_simulator.cpp)

target_include_directories(fdd_particle_filter_lib PUBLIC
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  $ENV{ACADOS_SOURCE_DIR}/include
  $ENV{ACADOS_SOURCE_DIR}/include/blasfeo/include
  $ENV{ACADOS_SOURCE_DIR}/include/hpipm/include
  ${PROJECT_SOURCE_DIR}/scripts/c_generated_code/sim_UAVModel
)

target_link_directories(fdd_particle_filter_lib PUBLIC
${catkin_LIB_DIRS}
$ENV{ACADOS_SOURCE_DIR}/lib
${PROJECT_SOURCE_DIR}/scripts/c_generated_code/sim_UAVModel)

target_link_libraries(fdd_particle_filter_lib acados_sim_solver_UAVModel acados hpipm blasfeo ${catkin_LIBRARIES} OpenMP::OpenMP_CXX)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.6.2
    GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(pybind11)
pybind11_add_module(FDDParticleFilter src/pybind_wrapper.cpp)
target_link_libraries(FDDParticleFilter PUBLIC fdd_particle_filter_lib)
set_target_properties(FDDParticleFilter PROPERTIES SUFFIX ".so")



add_executable(fdd_test src/test.cpp src/particle_filter.cpp src/uav_simulator.cpp)

target_include_directories(fdd_test PUBLIC
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  $ENV{ACADOS_SOURCE_DIR}/include
  $ENV{ACADOS_SOURCE_DIR}/include/blasfeo/include
  $ENV{ACADOS_SOURCE_DIR}/include/hpipm/include
  ${PROJECT_SOURCE_DIR}/scripts/c_generated_code/sim_UAVModel
)

target_link_directories(fdd_test PUBLIC
${catkin_LIB_DIRS}
$ENV{ACADOS_SOURCE_DIR}/lib
${PROJECT_SOURCE_DIR}/scripts/c_generated_code/sim_UAVModel)

target_link_libraries(fdd_test acados_sim_solver_UAVModel acados hpipm blasfeo ${catkin_LIBRARIES} OpenMP::OpenMP_CXX)
