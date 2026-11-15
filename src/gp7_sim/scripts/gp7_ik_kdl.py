#!/usr/bin/env python3
"""
KDL-based IK backend for GP7.
Interface:  fk(q) -> (pos, R);  ik(pos, R, q_seed) -> q or None
"""
import os
import subprocess
import tempfile
import numpy as np
import PyKDL as kdl
from kdl_parser_py.urdf import treeFromFile


def _ensure_urdf_file(path: str) -> str:
    if path.endswith(".xacro"):
        urdf_str = subprocess.check_output(["xacro", path]).decode()
        tmp = tempfile.NamedTemporaryFile(suffix=".urdf", delete=False, mode="w")
        tmp.write(urdf_str)
        tmp.close()
        return tmp.name
    return path


def _numpy_to_kdl(R, p):
    return kdl.Frame(
        kdl.Rotation(R[0, 0], R[0, 1], R[0, 2],
                     R[1, 0], R[1, 1], R[1, 2],
                     R[2, 0], R[2, 1], R[2, 2]),
        kdl.Vector(float(p[0]), float(p[1]), float(p[2]))
    )


def _kdl_to_numpy(frame):
    p = np.array([frame.p[0], frame.p[1], frame.p[2]])
    R = np.array([[frame.M[0, 0], frame.M[0, 1], frame.M[0, 2]],
                  [frame.M[1, 0], frame.M[1, 1], frame.M[1, 2]],
                  [frame.M[2, 0], frame.M[2, 1], frame.M[2, 2]]])
    return p, R


class GP7IK:
    def __init__(self, urdf_path, base_link="base_link",
                 tip_link="flange", tool_offset=0.14, **_ignored):
        urdf_file = _ensure_urdf_file(urdf_path)
        ok, tree = treeFromFile(urdf_file)
        if not ok:
            raise RuntimeError(f"Failed to build KDL tree from {urdf_file}")

        self.chain = tree.getChain(base_link, tip_link)
        if self.chain.getNrOfJoints() != 6:
            raise RuntimeError(
                f"Expected 6 joints, got {self.chain.getNrOfJoints()}"
            )
        self.n = self.chain.getNrOfJoints()
        self.joint_names = [
            self.chain.getSegment(i).getJoint().getName()
            for i in range(self.chain.getNrOfSegments())
            if self.chain.getSegment(i).getJoint().getType() != kdl.Joint.None
        ]

        self.tool_offset = tool_offset
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)
        self.ik_vel = kdl.ChainIkSolverVel_pinv(self.chain, 1e-6, 200)
        self.ik_pos = kdl.ChainIkSolverPos_NR(
            self.chain, self.fk_solver, self.ik_vel, 500, 1e-6
        )

    def _fk_flange(self, q):
        q_kdl = kdl.JntArray(self.n)
        for i in range(self.n):
            q_kdl[i] = float(q[i])
        frame = kdl.Frame()
        self.fk_solver.JntToCart(q_kdl, frame)
        return frame

    def fk(self, q):
        """Returns (grasp_point_xyz, R) — grasp point is flange + offset * +X."""
        T = self._fk_flange(q)
        p, R = _kdl_to_numpy(T)
        p_grasp = p + self.tool_offset * R[:, 0]
        return p_grasp, R

    def ik(self, pos_xyz, R, q_seed):
        """Solve for joints that put the grasp point at (pos_xyz, R)."""
        flange_pos = np.asarray(pos_xyz) - self.tool_offset * R[:, 0]
        T_target = _numpy_to_kdl(R, flange_pos)

        q_init = kdl.JntArray(self.n)
        for i in range(self.n):
            q_init[i] = float(q_seed[i])
        q_out = kdl.JntArray(self.n)
        ret = self.ik_pos.CartToJnt(q_init, T_target, q_out)
        if ret < 0:
            return None
        q = np.array([q_out[i] for i in range(self.n)])
        if np.any(np.abs(q) > 2 * np.pi):
            return None
        return q