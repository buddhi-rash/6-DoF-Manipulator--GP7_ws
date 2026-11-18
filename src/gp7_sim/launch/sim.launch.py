#!/usr/bin/env python3
"""GP7 pick-and-place scene in Gazebo Jetty (ROS 2 Lyrical).

    ros2 launch gp7_sim sim.launch.py

Arguments (all optional):
    box_a_x, box_a_y   Box A centre (m).  Default 0.5 0.5  (reachable; see README)
                       
    box_b_x, box_b_y   Box B centre (m) or 'random' (default) -> random reachable spot
    box_b_yaw          Box B yaw in rad (default 'random')
    seed               integer for a repeatable random Box B, or 'none'
    gz_args            extra flags for gz sim, default '-r -v 2'  (add ' -s' for headless)
"""
import math
import os
import random

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            IncludeLaunchDescription, LogInfo, OpaqueFunction,
                            RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# Top-down grasp annulus around the base axis. A numerical IK sweep over the GP7 kinematics
# (joint limits respected, collisions ignored) gives ~0.23-0.90 m; we keep a safety margin.
# Top-down grasp annulus around the base axis...
R_MIN, R_MAX = 0.40, 0.70
MAX_THETA = math.radians(120)      # stay away from the J1 limit (+-170 deg)
BOX_A_HALF, BOX_B_HALF, CLEARANCE = 0.15, 0.025, 0.10


def sample_box_b(ax, ay, seed):
    rng = random.Random(None if seed in ('', 'none') else int(seed))
    for _ in range(10000):
        r = rng.uniform(R_MIN, R_MAX)
        th = rng.uniform(-MAX_THETA, MAX_THETA)
        x, y = r * math.cos(th), r * math.sin(th)
        # keep Box B (plus finger clearance) outside Box A's footprint
        if max(abs(x - ax), abs(y - ay)) > BOX_A_HALF + BOX_B_HALF + CLEARANCE:
            return x, y, rng.uniform(-math.pi, math.pi)
    raise RuntimeError('could not find a free spot for Box B')


def spawner(name, param_file=None):
    # ROS 2 Lyrical: the controller_manager reads the controller YAML, but the controller nodes
    # do NOT inherit those parameters, so a trajectory controller starts with an empty `joints`
    # list and fails with "Failed loading controller". --param-file hands the YAML to the
    # controller node before it is configured.
    args = [name, '--controller-manager', '/controller_manager',
            '--controller-manager-timeout', '60']
    if param_file:
        args += ['--param-file', param_file]
    return Node(package='controller_manager', executable='spawner', output='screen',
                arguments=args)


def launch_setup(context, *args, **kwargs):
    pkg = get_package_share_directory('gp7_sim')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')
    cfg = lambda n: LaunchConfiguration(n).perform(context)

    ax, ay = float(cfg('box_a_x')), float(cfg('box_a_y'))
    bx_s, by_s, byaw_s = cfg('box_b_x'), cfg('box_b_y'), cfg('box_b_yaw')
    rx, ry, ryaw = sample_box_b(ax, ay, cfg('seed'))
    bx = rx if bx_s == 'random' else float(bx_s)
    by = ry if by_s == 'random' else float(by_s)
    byaw = ryaw if byaw_s == 'random' else float(byaw_s)

    msgs = [LogInfo(msg=f'[gp7_sim] Box A at ({ax:.3f}, {ay:.3f}),  radius {math.hypot(ax, ay):.3f} m'),
            LogInfo(msg=f'[gp7_sim] Box B at ({bx:.3f}, {by:.3f}), yaw {byaw:.2f} rad, '
                        f'radius {math.hypot(bx, by):.3f} m')]
    for nm, x, y in (('Box A', ax, ay), ('Box B', bx, by)):
        if math.hypot(x, y) > 0.85:
            msgs.append(LogInfo(msg=f'[gp7_sim] WARNING: {nm} is {math.hypot(x, y):.2f} m from the '
                                    f'base; the GP7 reaches ~0.90 m top-down. It cannot be reached.'))

    # --- environment so Gazebo can find the meshes (package://) and the ros2_control plugin ----
    env = [AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.dirname(pkg))]
    try:
        env.append(AppendEnvironmentVariable(
            'GZ_SIM_SYSTEM_PLUGIN_PATH',
            os.path.join(get_package_prefix('gz_ros2_control'), 'lib')))
    except Exception:
        msgs.append(LogInfo(msg='[gp7_sim] WARNING: package gz_ros2_control not found '
                                '(sudo apt install ros-lyrical-gz-ros2-control)'))

    robot_description = ParameterValue(
        Command([FindExecutable(name='xacro'), ' ',
                 os.path.join(pkg, 'urdf', 'gp7.urdf.xacro')]), value_type=str)

    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher',
               output='both',
               parameters=[{'robot_description': robot_description, 'use_sim_time': True}])

    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': f"{cfg('gz_args')} {os.path.join(pkg, 'worlds', 'pick_place.sdf')}"
        }.items())

    clock_bridge = Node(package='ros_gz_bridge', executable='parameter_bridge', output='screen',
                        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'])

    entity_state_bridge = Node(
    package='ros_gz_bridge', executable='parameter_bridge', output='screen',
    arguments=['/world/pick_place/get_entity_state'
               '@ros_gz_interfaces/srv/GetEntityState'
               '[gz.msgs.Entity'])
    
    def create(name, extra):
        return Node(package='ros_gz_sim', executable='create', output='screen',
                    arguments=['-world', 'pick_place', '-name', name] + extra)

    spawn_robot = create('gp7', ['-topic', 'robot_description', '-x', '0', '-y', '0', '-z', '0'])
    spawn_a = create('box_a', ['-file', os.path.join(pkg, 'models', 'box_a.sdf'),
                               '-x', str(ax), '-y', str(ay), '-z', '0.10'])
    spawn_b = create('box_b', ['-file', os.path.join(pkg, 'models', 'box_b.sdf'),
                               '-x', str(bx), '-y', str(by), '-z', '0.026', '-Y', str(byaw)])

    controllers_yaml = os.path.join(pkg, 'config', 'gp7_controllers.yaml')
    jsb = spawner('joint_state_broadcaster')
    arm = spawner('arm_controller', controllers_yaml)
    grip = spawner('gripper_controller', controllers_yaml)

    chain = [
        RegisterEventHandler(OnProcessExit(target_action=spawn_robot,
                                           on_exit=[spawn_a, spawn_b, jsb])),
        RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[arm])),
        RegisterEventHandler(OnProcessExit(target_action=arm, on_exit=[grip])),
    ]
    return msgs + env + [rsp, gz, clock_bridge, entity_state_bridge, spawn_robot] + chain


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('box_a_x', default_value='0.5'),
        DeclareLaunchArgument('box_a_y', default_value='0.5'),
        DeclareLaunchArgument('box_b_x', default_value='random'),
        DeclareLaunchArgument('box_b_y', default_value='random'),
        DeclareLaunchArgument('box_b_yaw', default_value='random'),
        DeclareLaunchArgument('seed', default_value='none'),
        DeclareLaunchArgument('gz_args', default_value='-r -v 2'),
        OpaqueFunction(function=launch_setup),
    ])
