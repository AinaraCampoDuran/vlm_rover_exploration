cmake_minimum_required(VERSION 3.8)
project(vlm_rover_exploration)

# find dependencies
find_package(ament_cmake REQUIRED)

install(PROGRAMS
  vlm_rover_exploration/exploration_sm.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME exploration_sm
)

ament_python_install_package(${PROJECT_NAME})
ament_package()