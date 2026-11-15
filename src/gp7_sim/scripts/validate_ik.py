#!/usr/bin/env python3
"""Compare analytical IK against KDL reference on random poses."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import gp7_ik_kdl as kdl_backend
import gp7_ik_analytical as ana_backend
from ament_index_python.packages import get_package_share_directory

URDF = os.path.join(get_package_share_directory("gp7_sim"),
                    "urdf", "gp7.urdf.xacro")


def main():
    kdl_s = kdl_backend.GP7IK(URDF)
    ana_s = ana_backend.GP7IK(URDF)
    rng = np.random.default_rng(0)

    print(f"{'#':>3} {'FK_pos_err':>12} {'FK_rot_err':>12} {'q_err':>12}")
    worst_p = worst_r = worst_q = 0.0
    for i in range(20):
        q_test = rng.uniform(-1.2, 1.2, 6)
        p0, R0 = kdl_s.fk(q_test)
        q_kdl = kdl_s.ik(p0, R0, q_test)
        q_ana = ana_s.ik(p0, R0, q_test)
        if q_kdl is None or q_ana is None:
            print(f"{i:>3}  IK failed")
            continue
        pk, Rk = kdl_s.fk(q_kdl)
        pa, Ra = ana_s.fk(q_ana)
        pe = np.linalg.norm(pk - pa)
        re = np.linalg.norm(Rk - Ra, ord="fro")
        qe = np.max(np.abs(q_kdl - q_ana))
        worst_p = max(worst_p, pe)
        worst_r = max(worst_r, re)
        worst_q = max(worst_q, qe)
        print(f"{i:>3} {pe:12.3e} {re:12.3e} {qe:12.3e}")

    print(f"\nWorst pos error: {worst_p:.3e} m")
    print(f"Worst rot error: {worst_r:.3e}")
    print(f"Worst q error : {worst_q:.3e} rad")


if __name__ == "__main__":
    main()