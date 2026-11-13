cmake_minimum_required(VERSION 3.20)
project(gp7_sim)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch
  urdf
  meshes
  config
  worlds
  models
  rviz
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/move_test.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
