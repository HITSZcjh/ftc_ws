cmake_minimum_required(VERSION 3.0.2)
project(fdd)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
)
find_package(Eigen3)
find_package(OpenMP)
catkin_package()

# include_directories(

# )

add_executable(test1 src/test.cpp src/uav_simulator.cpp src/particle_filter.cpp)

target_include_directories(test1 PUBLIC
  include
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

target_link_libraries(test1 acados_sim_solver_UAVModel acados hpipm blasfeo ${catkin_LIBRARIES} OpenMP::OpenMP_CXX)



add_executable(test2 src/test2.cpp src/uav_simulator.cpp)

target_include_directories(test2 PUBLIC
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  $ENV{ACADOS_SOURCE_DIR}/include
  $ENV{ACADOS_SOURCE_DIR}/include/blasfeo/include
  $ENV{ACADOS_SOURCE_DIR}/include/hpipm/include
  ${PROJECT_SOURCE_DIR}/scripts/c_generated_code/sim_UAVModel
)

target_link_directories(test2 PUBLIC
${catkin_LIB_DIRS}
$ENV{ACADOS_SOURCE_DIR}/lib
${PROJECT_SOURCE_DIR}/scripts/c_generated_code/sim_UAVModel)

target_link_libraries(test2 acados_sim_solver_UAVModel acados hpipm blasfeo ${catkin_LIBRARIES} OpenMP::OpenMP_CXX)