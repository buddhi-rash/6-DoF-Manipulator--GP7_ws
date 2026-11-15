#!/usr/bin/env python3
"""
GP7 pick-and-place for Gazebo Jetty / ROS 2 Lyrical.

- Six straight-line Cartesian segments with proper trapezoidal velocity spacing
- Constant tool-down orientation — Box B stays upright throughout
- Box A and Box B poses read live from Gazebo via `gz model --pose` CLI
- Dynamic pre-pose to prevent IK branch folding
- Position-only trajectories with 5-point moving average smoothing
"""
import os
import sys
import time
import csv
import math
import re
import subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration as DurationMsg
from ament_index_python.packages import get_package_share_directory

from ik_backend import GP7IK, BACKEND
import cartesian_planner as cp

# ---------------- configuration ----------------
URDF_PATH = os.path.join(
    get_package_share_directory("gp7_sim"), "urdf", "gp7.urdf.xacro"
)
BASE_LINK   = "base_link"
TIP_LINK    = "flange"
TOOL_OFFSET = 0.14

GRIPPER_JOINTS = ["finger_left_joint", "finger_right_joint"]
GRIPPER_OPEN   = [0.0,   0.0]
GRIPPER_CLOSE  = [0.022, 0.022]   # Squeezes the box firmly

JOINT_VEL_LIMITS = np.array([3.14] * 6)

# Increased speeds for smoother, faster motion
V_MAX_PATH = 0.20
A_MAX_PATH = 0.40
DT         = 0.01

APPROACH_H = 0.12
LIFT_H     = 0.15
PLACE_DROP = 0.005

BOX_A_HEIGHT = 0.20
BOX_B_HALF   = 0.025
# ---------------------------------------------------


def _quat_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class PickPlace(Node):
    def __init__(self):
        super().__init__("gp7_pick_place")

        self.ik = GP7IK(URDF_PATH, base_link=BASE_LINK,
                        tip_link=TIP_LINK, tool_offset=TOOL_OFFSET)
        self.joint_names = self.ik.joint_names
        self.get_logger().info(
            f"IK backend: {BACKEND} | joints: {self.joint_names}"
        )

        self.arm_pub = self.create_publisher(
            JointTrajectory, "/arm_controller/joint_trajectory", 10)
        self.grip_pub = self.create_publisher(
            JointTrajectory, "/gripper_controller/joint_trajectory", 10)
        self.create_subscription(JointState, "/joint_states",
                                 self._joint_state_cb, 10)

        self.current_q = None
        self.log_rows = []
        self.plan_time = 0.0   # <--- ADD THIS LINE

    def _joint_state_cb(self, msg):
        q = np.zeros(len(self.joint_names))
        seen = 0
        for name, pos in zip(msg.name, msg.position):
            if name in self.joint_names:
                q[self.joint_names.index(name)] = pos
                seen += 1
        if seen == len(self.joint_names):
            self.current_q = q

    def _query_entity_pose(self, name, attempts=3):
        for attempt in range(attempts):
            try:
                out = subprocess.check_output(
                    ["gz", "model", "-m", name, "--pose"],
                    text=True, timeout=5.0, stderr=subprocess.STDOUT
                )
                arrays = re.findall(r"\[([-\d\.\seE]+)\]", out)
                if len(arrays) >= 2:
                    xyz_parts = arrays[-2].split()
                    rpy_parts = arrays[-1].split()
                    if len(xyz_parts) >= 3 and len(rpy_parts) >= 3:
                        xyz = np.array([float(xyz_parts[0]),
                                        float(xyz_parts[1]),
                                        float(xyz_parts[2])])
                        yaw = float(rpy_parts[2])
                        return xyz, yaw
            except Exception:
                pass
            time.sleep(0.5)
        raise RuntimeError(f"Could not query pose of '{name}' from Gazebo")

    def wait_for_sim(self, timeout=30.0):
        t0 = time.time()
        while (self.arm_pub.get_subscription_count() == 0
               or self.current_q is None):
            rclpy.spin_once(self, timeout_sec=0.2)
            if time.time() - t0 > timeout:
                missing = []
                if self.arm_pub.get_subscription_count() == 0:
                    missing.append("arm_controller")
                if self.current_q is None:
                    missing.append("joint_states")
                raise RuntimeError(f"Timed out waiting for: {missing}")
        self.get_logger().info("Arm controller + joint states ready.")

    def _joint_lerp(self, q_from, q_to, n_steps=100):
        q_list = np.array([q_from + (q_to - q_from) * i / (n_steps - 1)
                           for i in range(n_steps)])
        qd_list = np.gradient(q_list, DT, axis=0)
        qdd_list = np.gradient(qd_list, DT, axis=0)
        return q_list.tolist(), qd_list.tolist(), qdd_list.tolist()

    def _cartesian_segment(self, label, start_xyz, end_xyz, q_seed,
                           R_fixed, v_scale=1.0):
        start_xyz = np.asarray(start_xyz)
        end_xyz = np.asarray(end_xyz)
        length = np.linalg.norm(end_xyz - start_xyz)
        
        # Calculate trapezoidal profile: returns time, distance(s), velocity, accel
        t, s, s_dot, s_ddot = cp.trapezoidal_profile(
            length, V_MAX_PATH * v_scale, A_MAX_PATH * v_scale, DT)
        
        n = len(s)
        if n == 0:
            return [], [], [], q_seed

        # Generate positions using the 's' array for proper acceleration/deceleration
        positions = []
        for si in s:
            alpha = si / length if length > 1e-9 else 0.0
            pos = start_xyz + alpha * (end_xyz - start_xyz)
            positions.append(pos)

        q_list, qd_list, qdd_list = [], [], []
        q_prev = q_seed.copy()
        qd_prev = np.zeros(6)

        for i, pos in enumerate(positions):
            q_sol = self.ik.ik(pos, R_fixed, q_prev)
            if q_sol is None:
                raise RuntimeError(
                    f"IK failed on '{label}' sample {i}/{n} pos={pos}")
            q_list.append(q_sol)
            qd = (q_sol - q_prev) / DT if i > 0 else np.zeros(6)
            qdd = (qd - qd_prev) / DT if i > 0 else np.zeros(6)
            qd_list.append(qd)
            qdd_list.append(qdd)
            q_prev, qd_prev = q_sol, qd

        # Apply a 5-point moving average to smooth out IK micro-jitter
        if len(q_list) > 5:
            q_arr = np.array(q_list)
            q_smooth = np.copy(q_arr)
            for i in range(2, len(q_arr) - 2):
                q_smooth[i] = np.mean(q_arr[i-2:i+3], axis=0)
            q_smooth[0] = q_arr[0]
            q_smooth[-1] = q_arr[-1]
            q_list = q_smooth.tolist()

        self._log_segment(label, q_list, qd_list, qdd_list)
        return q_list, qd_list, qdd_list, q_list[-1]

    def _send_arm(self, q_list, qd_list, qdd_list):
        """Publish a position-only trajectory at full 100Hz resolution."""
        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        n = len(q_list)
        if n == 0:
            return

        for i in range(n):
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in q_list[i]]
            t = i * DT
            pt.time_from_start = DurationMsg(
                sec=int(t), nanosec=int((t - int(t)) * 1e9))
            msg.points.append(pt)

        self.arm_pub.publish(msg)
        time.sleep(n * DT + 1.0)

    def _gripper(self, positions, wait=1.2):
        msg = JointTrajectory()
        msg.joint_names = GRIPPER_JOINTS
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in positions]
        pt.time_from_start = DurationMsg(sec=1, nanosec=0)
        msg.points.append(pt)
        self.grip_pub.publish(msg)
        time.sleep(wait)

    def _log_segment(self, label, q_list, qd_list, qdd_list):
        for i in range(len(q_list)):
            # Use cumulative time
            row = {"segment": label, "t": self.plan_time + i * DT}
            for j in range(6):
                row[f"q{j}"] = float(q_list[i][j])
                row[f"qd{j}"] = float(qd_list[i][j])
                row[f"qdd{j}"] = float(qdd_list[i][j])
            self.log_rows.append(row)
        # Advance the global time by the duration of this segment
        self.plan_time += len(q_list) * DT

    def _write_csv(self):
        if not self.log_rows:
            return
        path = os.path.abspath("pick_place_log.csv")
        keys = list(self.log_rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.log_rows)
        self.get_logger().info(f"Wrote {path}")

    def run(self):
        self.wait_for_sim()

        # 1. Read Box poses
        box_a_xyz, _ = self._query_entity_pose("box_a")
        box_b_xyz, box_b_yaw = self._query_entity_pose("box_b")
        self.get_logger().info(
            f"Box A at {box_a_xyz.round(3)}, Box B at {box_b_xyz.round(3)}, "
            f"yaw {math.degrees(box_b_yaw):.1f} deg"
        )

        # 2. Derive targets
        box_a_top_z = box_a_xyz[2] + BOX_A_HEIGHT / 2.0
        # Grasp point lowered to 0.010 so fingers wrap around the box
        box_b_grasp = np.array([box_b_xyz[0], box_b_xyz[1], 0.010])
        approach = box_b_grasp + np.array([0, 0, APPROACH_H])
        lift     = box_b_grasp + np.array([0, 0, LIFT_H])

        place_target = np.array([
            box_a_xyz[0],
            box_a_xyz[1],
            box_a_top_z + BOX_B_HALF + PLACE_DROP,
        ])
        transfer = place_target + np.array([0, 0, APPROACH_H])
        retreat  = place_target + np.array([0, 0, LIFT_H])

        # 3. Tool orientation
        R_down = cp.tool_down_R(yaw=box_b_yaw)

        # 4. Dynamic Pre-pose
        theta = math.atan2(box_b_grasp[1], box_b_grasp[0])
        q_start_seed = np.array([theta, 0.0, 0.0, 0.0, -1.57, 0.0])
        p_start_seed, _ = self.ik.fk(q_start_seed)
        q_start = self.ik.ik(p_start_seed, R_down, q_start_seed)
        if q_start is None:
            q_start = q_start_seed

        self.get_logger().info("Moving to ready pose")
        qL, qdL, qddL = self._joint_lerp(self.current_q.copy(), q_start)
        self._send_arm(qL, qdL, qddL)
        
        q = q_start.copy()
        p_start, _ = self.ik.fk(q)

        # 5. Approach
        self.get_logger().info("Approaching Box B")
        qL, qdL, qddL, q = self._cartesian_segment(
            "approach", p_start, approach, q, R_down)
        self._send_arm(qL, qdL, qddL)

        # 6. Descend
        self.get_logger().info("Descending to grasp")
        qL, qdL, qddL, q = self._cartesian_segment(
            "grasp", approach, box_b_grasp, q, R_down)
        self._send_arm(qL, qdL, qddL)

        # 7. Close Gripper
        self.get_logger().info("Closing gripper")
        self._gripper(GRIPPER_CLOSE, wait=1.2)

        # 8. Lift
        self.get_logger().info("Lifting")
        qL, qdL, qddL, q = self._cartesian_segment(
            "lift", box_b_grasp, lift, q, R_down)
        self._send_arm(qL, qdL, qddL)

        # 9. Transfer
        self.get_logger().info("Transferring to Box A")
        qL, qdL, qddL, q = self._cartesian_segment(
            "transfer", lift, transfer, q, R_down)
        self._send_arm(qL, qdL, qddL)

        # 10. Place
        self.get_logger().info("Descending to place")
        qL, qdL, qddL, q = self._cartesian_segment(
            "place", transfer, place_target, q, R_down)
        self._send_arm(qL, qdL, qddL)

        # 11. Open Gripper
        self.get_logger().info("Opening gripper")
        self._gripper(GRIPPER_OPEN, wait=1.2)

        # 12. Retreat
        self.get_logger().info("Retreating")
        qL, qdL, qddL, q = self._cartesian_segment(
            "retreat", place_target, retreat, q, R_down)
        self._send_arm(qL, qdL, qddL)

        self._write_csv()
        self.get_logger().info("Pick-and-place complete.")


def main():
    rclpy.init()
    node = PickPlace()
    try:
        node.run()
    except Exception as e:
        node.get_logger().error(f"Aborted: {e}")
        raise
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()