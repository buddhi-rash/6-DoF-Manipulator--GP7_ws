#!/usr/bin/env python3

import numpy as np
import math


class GP7IK:
    def __init__(self, urdf_path=None, base_link="base_link",
                 tip_link="flange", tool_offset=0.14, **_ignored):
        self.tool_offset = tool_offset
        self.joint_names = ["joint_1_s", "joint_2_l", "joint_3_u",
                            "joint_4_r", "joint_5_b", "joint_6_t"]
        

        self.d = [0.330, 0.0, 0.0, -0.440, 0.0, -0.080]
        self.a = [0.040, 0.445, 0.040, 0.0, 0.0, 0.0]
        self.alpha = [np.radians(-90), np.radians(180), np.radians(-90), 
                      np.radians(90), np.radians(-90), np.radians(180)]
        

        self.q_min = np.radians([-170, -65, -70, -190, -135, -360])
        self.q_max = np.radians([170, 145, 190, 190, 135, 360])

    def _dh_transform(self, i, q_i):
        """Calculates the standard DH transformation matrix for joint i."""
        d, a, alpha = self.d[i], self.a[i], self.alpha[i]
        theta = q_i if i != 1 else q_i - np.pi/2  # Joint 2 has a -90 deg offset
        
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)

        return np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0,   sa,     ca,    d],
            [0,   0,      0,     1]
        ])

    def fk(self, q):
        """Forward kinematics using the DH table."""
        T = np.eye(4)
        for i in range(6):
            T = T @ self._dh_transform(i, q[i])
            
        p_flange = T[:3, 3]
        R_flange = T[:3, :3]
        

        p_grasp = p_flange + self.tool_offset * R_flange[:, 2]
        return p_grasp, R_flange

    def _get_jacobian(self, q):
        """Derives the 6x6 Jacobian matrix (Section 2.4.3)."""
        J = np.zeros((6, 6))
        T = np.eye(4)
        
        p_e = self.fk(q)[0]  # End-effector position
        
        for i in range(6):
            # The origin and z-axis BEFORE joint i rotates
            p_prev = T[:3, 3]
            z_prev = T[:3, 2]
            
            # Linear velocity component: Jv = z_{i-1} x (P_e - O_{i-1})
            J[:3, i] = np.cross(z_prev, (p_e - p_prev))
            # Angular velocity component: Jw = z_{i-1}
            J[3:, i] = z_prev
            
            # Move to the next frame
            T = T @ self._dh_transform(i, q[i])
            
        return J

    def ik(self, pos_xyz, R, q_seed):
        
        q = np.array(q_seed, dtype=float)
        p_target = np.asarray(pos_xyz)
        
        # DLS parameters
        max_iter = 100
        lambda_damp = 0.05
        tolerance = 1e-4

        for _ in range(max_iter):
            p_curr, R_curr = self.fk(q)
            
            # 1. Position error (e_p)
            e_p = p_target - p_curr
            
            # 2. Orientation error (e_o)
            # R_e = R_d * R(q)^T
            R_e = R @ R_curr.T
            # Extract orientation vector from rotation matrix (skew-symmetric)
            e_o = 0.5 * np.array([
                R_e[2, 1] - R_e[1, 2],
                R_e[0, 2] - R_e[2, 0],
                R_e[1, 0] - R_e[0, 1]
            ])
            
            e = np.concatenate([e_p, e_o])
            
            # Check for convergence
            if np.linalg.norm(e) < tolerance:
                return q
                
            # 3. Compute Jacobian
            J = self._get_jacobian(q)
            
            # 4. Damped Least Squares update: dq = J^T (JJ^T + lambda^2 I)^-1 e
            JJT = J @ J.T
            dq = J.T @ np.linalg.inv(JJT + (lambda_damp**2) * np.eye(6)) @ e
            
            # 5. Update joint angles
            q = q + dq
            
            # 6. Enforce joint limits
            q = np.clip(q, self.q_min, self.q_max)

        # If it didn't converge within max_iter, return None
        return None