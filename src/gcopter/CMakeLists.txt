cmake_minimum_required(VERSION 2.8.3)

project(gcopter)

set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -fPIC")

find_package(Eigen3 REQUIRED)
find_package(ompl REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  sensor_msgs
  visualization_msgs
  nav_msgs
  tf
  quadrotor_msgs
  perception_utils
)

include_directories(
    ${catkin_INCLUDE_DIRS}
    ${OMPL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    include
)

catkin_package(
  CATKIN_DEPENDS quadrotor_msgs tf
)

add_executable(gcopter_node
  src/gcopter_node.cpp
  src/gcopter_fsm.cpp
  src/gcopter_planner.cpp
  src/gcopter_traj_server.cpp
  src/gcopter_optimizer.cpp
)

target_link_libraries(gcopter_node
  ${OMPL_LIBRARIES}
  ${catkin_LIBRARIES}
)

# ---- Upstream demo (global planning on a mock map) ----
add_executable(global_planning src/global_planning.cpp)

target_link_libraries(global_planning
  ${OMPL_LIBRARIES}
  ${catkin_LIBRARIES}
)

# ---- Install ----
install(TARGETS gcopter_node global_planning
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

