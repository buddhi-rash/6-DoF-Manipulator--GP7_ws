#!/usr/bin/env python3
"""
Analytical (D-H-derived) IK — fill in tonight with your teammates' derivation.
Must match the GP7IK interface in gp7_ik_kdl.py exactly.
"""
import numpy as np


class GP7IK:
    def __init__(self, urdf_path, base_link="base_link",
                 tip_link="flange", tool_offset=0.14, **_ignored):
        self.tool_offset = tool_offset
        self.joint_names = ["joint_1_s", "joint_2_l", "joint_3_u",
                            "joint_4_r", "joint_5_b", "joint_6_t"]
        # --- D-H parameters (replace with your derivation) ---
        self.d = [0.0] * 6
        self.a = [0.0] * 6
        self.alpha = [0.0] * 6

    def _dh(self, i, q):
        d, a, alpha = self.d[i], self.a[i], self.alpha[i]
        return np.array([
            [np.cos(q), -np.sin(q)*np.cos(alpha),  np.sin(q)*np.sin(alpha), a*np.cos(q)],
            [np.sin(q),  np.cos(q)*np.cos(alpha), -np.cos(q)*np.sin(alpha), a*np.sin(q)],
            [0.0,        np.sin(alpha),            np.cos(alpha),            d],
            [0.0, 0.0, 0.0, 1.0],
        ])

    def fk(self, q):
        """Returns (grasp_point_xyz, R) in base_link frame."""
        T = np.eye(4)
        for i in range(6):
            T = T @ self._dh(i, q[i])
        # Apply base_link -> base offset (0.33 m Z) and tool offset if needed
        p = T[:3, 3].copy() + np.array([0, 0, self.tool_offset])
        R = T[:3, :3].copy()
        return p, R

    def ik(self, pos_xyz, R, q_seed):
        raise NotImplementedError("Fill in the analytical IK from your D-H derivation.")
