#!/usr/bin/env python3
"""Smoke test for the controllers: sends a few joint-space waypoints and opens/closes the gripper.

    ros2 run gp7_sim move_test.py

If the arm moves, joint_state_broadcaster / arm_controller / gripper_controller and the
Gazebo <-> ros2_control link are all working. Replace this later with your own node that
sends the dense, trapezoidal/S-curve trajectory computed from your IK.
"""
import math
import time

import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM = ['joint_1_s', 'joint_2_l', 'joint_3_u', 'joint_4_r', 'joint_5_b', 'joint_6_t']
FINGERS = ['finger_left_joint', 'finger_right_joint']

# (time_from_start [s], joint angles [rad])   J5 = -90 deg -> tool axis points straight down
ARM_WAYPOINTS = [
    (3.0,  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    (7.0,  [0.0, 0.0, 0.0, 0.0, -math.pi / 2, 0.0]),
    (11.0, [0.8, 0.0, 0.0, 0.0, -math.pi / 2, 0.0]),
    (15.0, [-0.8, 0.0, 0.0, 0.0, -math.pi / 2, 0.0]),
    (19.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
]
GRIPPER_WAYPOINTS = [(2.0, [0.0, 0.0]), (6.0, [0.015, 0.015]), (10.0, [0.0, 0.0])]


def make_traj(joints, waypoints):
    msg = JointTrajectory()
    msg.joint_names = joints
    for t, q in waypoints:
        p = JointTrajectoryPoint()
        p.positions = [float(v) for v in q]
        p.time_from_start = Duration(sec=int(t), nanosec=int((t % 1) * 1e9))
        msg.points.append(p)
    return msg


def main():
    rclpy.init()
    node = Node('gp7_move_test')
    arm_pub = node.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
    grip_pub = node.create_publisher(JointTrajectory, '/gripper_controller/joint_trajectory', 10)

    node.get_logger().info('waiting for the controllers to subscribe ...')
    t0 = time.time()
    while (arm_pub.get_subscription_count() == 0 or grip_pub.get_subscription_count() == 0):
        rclpy.spin_once(node, timeout_sec=0.2)
        if time.time() - t0 > 60:
            node.get_logger().error('controllers not found - is the simulation running?')
            return

    node.get_logger().info('sending trajectories')
    arm_pub.publish(make_traj(ARM, ARM_WAYPOINTS))
    grip_pub.publish(make_traj(FINGERS, GRIPPER_WAYPOINTS))
    time.sleep(21.0)
    node.get_logger().info('done')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
