#!/usr/bin/env python3
"""GP7 pick-and-place in Gazebo Jetty / ROS 2 Lyrical."""
import os
import sys
import time
import csv
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
GRIPPER_CLOSE  = [0.015, 0.015]

JOINT_VEL_LIMITS = np.array([3.14] * 6)

V_MAX_PATH = 0.10
A_MAX_PATH = 0.20
DT         = 0.01

Q_START = np.array([0.0, 0.5, -1.0, 0.0, -1.5, 0.0])

APPROACH_H = 0.12
LIFT_H     = 0.15
PLACE_H    = 0.015

BOX_B_PICK   = np.array([0.55, 0.35, 0.025])
BOX_A_CENTER = np.array([0.55, 0.45, 0.20])
# ---------------------------------------------------


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

    def _joint_state_cb(self, msg):
        q = np.zeros(len(self.joint_names))
        seen = 0
        for name, pos in zip(msg.name, msg.position):
            if name in self.joint_names:
                q[self.joint_names.index(name)] = pos
                seen += 1
        if seen == len(self.joint_names):
            self.current_q = q

    def wait_for(self, timeout=30.0):
        t0 = time.time()
        while (self.arm_pub.get_subscription_count() == 0
               or self.current_q is None):
            rclpy.spin_once(self, timeout_sec=0.2)
            if time.time() - t0 > timeout:
                raise RuntimeError(
                    "Timed out waiting for arm controller / joint states. "
                    "Is Gazebo + controllers running?")

    def _joint_lerp(self, q_from, q_to, n_steps=100):
        q_list = np.array([q_from + (q_to - q_from) * i / (n_steps - 1)
                           for i in range(n_steps)])
        qd_list = np.gradient(q_list, DT, axis=0)
        qdd_list = np.gradient(qd_list, DT, axis=0)
        return q_list.tolist(), qd_list.tolist(), qdd_list.tolist()

    def _cartesian_segment(self, label, start_xyz, end_xyz, q_seed,
                           R_fixed, v_scale=1.0):
        length = np.linalg.norm(np.asarray(end_xyz) - np.asarray(start_xyz))
        _, s, _, _ = cp.trapezoidal_profile(
            length, V_MAX_PATH * v_scale, A_MAX_PATH * v_scale, DT)
        n = len(s)
        positions = cp.straight_line(start_xyz, end_xyz, n)

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

        qd_arr = np.array(qd_list)
        peak = np.max(np.abs(qd_arr), axis=0)
        worst = np.max(peak / JOINT_VEL_LIMITS)
        if worst > 1.0:
            scale = 0.95 / worst
            self.get_logger().warn(
                f"'{label}': scaling by {scale:.3f} for joint velocity limits")
            return self._cartesian_segment(
                label, start_xyz, end_xyz, q_seed, R_fixed, scale)

        self._log_segment(label, q_list, qd_list, qdd_list)
        return q_list, qd_list, qdd_list, q_list[-1]

    def _send_arm(self, q_list, qd_list, qdd_list):
        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        for i, (q, qd, qdd) in enumerate(zip(q_list, qd_list, qdd_list)):
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in q]
            pt.velocities = [float(v) for v in qd]
            pt.accelerations = [float(v) for v in qdd]
            t = (i + 1) * DT
            pt.time_from_start = DurationMsg(
                sec=int(t), nanosec=int((t - int(t)) * 1e9))
            msg.points.append(pt)
        self.arm_pub.publish(msg)
        time.sleep(len(q_list) * DT + 0.8)

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
            row = {"segment": label, "t": i * DT}
            for j in range(6):
                row[f"q{j}"] = float(q_list[i][j])
                row[f"qd{j}"] = float(qd_list[i][j])
                row[f"qdd{j}"] = float(qdd_list[i][j])
            self.log_rows.append(row)

    def _write_csv(self):
        path = os.path.abspath("pick_place_log.csv")
        keys = list(self.log_rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.log_rows)
        self.get_logger().info(f"Wrote {path}")

    def run(self):
        self.wait_for()
        R_down = cp.tool_down_R()

        # Phase 0: joint-space to a known tool-down "ready" pose
        self.get_logger().info("Phase 0: moving to ready pose")
        q_now = self.current_q.copy()
        qL, qdL, qddL = self._joint_lerp(q_now, Q_START, n_steps=100)
        self._send_arm(qL, qdL, qddL)
        q = Q_START.copy()
        p_start, R_start = self.ik.fk(q)
        self.get_logger().info(
            f"Ready pose FK: pos={p_start.round(3)}, tool={R_start[:, 0].round(3)}"
        )

        # 1. Approach
        approach = BOX_B_PICK + np.array([0, 0, APPROACH_H])
        qL, qdL, qddL, q = self._cartesian_segment(
            "approach", p_start, approach, q, R_down)
        self._send_arm(qL, qdL, qddL)

        # 2. Descend to grasp
        qL, qdL, qddL, q = self._cartesian_segment(
            "grasp", approach, BOX_B_PICK, q, R_down)
        self._send_arm(qL, qdL, qddL)

        # 3. Close gripper
        self.get_logger().info("Closing gripper")
        self._gripper(GRIPPER_CLOSE, wait=1.2)

        # 4. Lift
        lift = BOX_B_PICK + np.array([0, 0, LIFT_H])
        qL, qdL, qddL, q = self._cartesian_segment(
            "lift", BOX_B_PICK, lift, q, R_down)
        self._send_arm(qL, qdL, qddL)

        # 5. Transfer
        transfer = BOX_A_CENTER + np.array([0, 0, APPROACH_H])
        qL, qdL, qddL, q = self._cartesian_segment(
            "transfer", lift, transfer, q, R_down)
        self._send_arm(qL, qdL, qddL)

        # 6. Place
        place = BOX_A_CENTER + np.array([0, 0, PLACE_H])
        qL, qdL, qddL, q = self._cartesian_segment(
            "place", transfer, place, q, R_down)
        self._send_arm(qL, qdL, qddL)

        # 7. Open gripper
        self.get_logger().info("Opening gripper")
        self._gripper(GRIPPER_OPEN, wait=1.2)

        # 8. Retreat
        retreat = place + np.array([0, 0, LIFT_H])
        qL, qdL, qddL, q = self._cartesian_segment(
            "retreat", place, retreat, q, R_down)
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