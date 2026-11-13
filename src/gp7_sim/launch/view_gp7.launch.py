#!/usr/bin/env python3
"""Look at the model only (no Gazebo): RViz + joint sliders.

    ros2 launch gp7_sim view_gp7.launch.py

Use this FIRST. If the robot looks right here (correct size, joints move the right way)
the URDF/xacro is fine and any remaining problem is on the Gazebo side.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('gp7_sim')
    xacro_file = os.path.join(pkg, 'urdf', 'gp7.urdf.xacro')
    rviz_config = os.path.join(pkg, 'rviz', 'view_gp7.rviz')

    robot_description = ParameterValue(
        Command([FindExecutable(name='xacro'), ' ', xacro_file]), value_type=str)

    return LaunchDescription([
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot_description}], output='screen'),
        Node(package='joint_state_publisher_gui', executable='joint_state_publisher_gui',
             output='screen'),
        Node(package='rviz2', executable='rviz2', arguments=['-d', rviz_config],
             output='screen'),
    ])
