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
install(PROGRAMS
  scripts/pick_place.py
  scripts/plot_traj.py
  scripts/validate_ik.py
  DESTINATION lib/${PROJECT_NAME}
)
install(FILES
  scripts/gp7_ik_kdl.py
  scripts/gp7_ik_analytical.py
  scripts/ik_backend.py
  scripts/cartesian_planner.py
  DESTINATION lib/${PROJECT_NAME}
)
ament_package()
