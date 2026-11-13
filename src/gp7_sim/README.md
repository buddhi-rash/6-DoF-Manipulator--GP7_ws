# gp7_sim - Yaskawa GP7 pick-and-place (ROS 2 Lyrical + Gazebo Jetty)

## Install dependencies (Ubuntu 26.04 with ROS apt packages)
    sudo apt install ros-lyrical-ros-gz ros-lyrical-gz-ros2-control \
         ros-lyrical-ros2-control ros-lyrical-ros2-controllers \
         ros-lyrical-xacro ros-lyrical-robot-state-publisher \
         ros-lyrical-joint-state-publisher-gui ros-lyrical-rviz2

## Build
    mkdir -p ~/gp7_ws/src && cp -r gp7_sim ~/gp7_ws/src/
    cd ~/gp7_ws && rosdep install --from-paths src --ignore-src -r -y
    colcon build --symlink-install --packages-select gp7_sim
    source install/setup.bash

## Run
    ros2 launch gp7_sim view_gp7.launch.py                  # model only (RViz + sliders)
    ros2 launch gp7_sim sim.launch.py                       # Gazebo scene + controllers
    ros2 run    gp7_sim move_test.py                        # controller smoke test
    ros2 launch gp7_sim sim.launch.py seed:=3               # repeatable random Box B
    ros2 launch gp7_sim sim.launch.py box_a_x:=1.0 box_a_y:=1.0   # scene exactly as on the task sheet

## Controller interface
    /arm_controller/joint_trajectory      trajectory_msgs/JointTrajectory   (6 arm joints)
    /gripper_controller/joint_trajectory  trajectory_msgs/JointTrajectory   (2 finger joints, 0 = open, 0.015 = closed on 5 cm box)
    /joint_states                         sensor_msgs/JointState
    action: /arm_controller/follow_joint_trajectory

## Frames
    world -> base_link -> link_1_s ... link_6_t -> flange (-> tool0), gripper_base, finger_left/right
    Tool axis = +X of `flange`.  J5 = -90 deg (others 0) points the tool straight down.

## Reach (computed from this URDF)
    max wrist (J5) radius 0.927 m ; top-down grasp workable radius ~0.23-0.90 m
    -> Box A at (1,1) m (1.41 m away) is NOT reachable; default here is (0.55, 0.45).
    Box B is sampled in 0.35-0.75 m, |theta| <= 150 deg, clear of Box A.

## Credits
    Link meshes and joint layout come from the ROS-Industrial motoman_gp7 description
    (ros-industrial/motoman). Cite it in your report.
