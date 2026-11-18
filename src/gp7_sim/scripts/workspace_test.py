#!/usr/bin/env python3
import numpy as np
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ik_backend import GP7IK
import cartesian_planner as cp

URDF_PATH = os.path.join(os.path.dirname(__file__), "..", "urdf", "gp7.urdf.xacro")
TOOL_OFFSET = 0.14

def test_segment(ik, start_xyz, end_xyz, q_seed, R_fixed):
    """Simulates the _cartesian_segment function to check for IK failures."""
    length = np.linalg.norm(end_xyz - start_xyz)
    positions = cp.straight_line(start_xyz, end_xyz, 20) # Test 20 points
    
    q_prev = q_seed.copy()
    for i, pos in enumerate(positions):
        q_sol = ik.ik(pos, R_fixed, q_prev)
        if q_sol is None:
            return False, f"IK failed at {pos.round(2)}"
            
        if i > 0 and np.max(np.abs(q_sol - q_prev)) > 0.6:
            return False, f"Branch jump at {pos.round(2)}"
            
        q_prev = q_sol
    return True, ""

def main():
    ik = GP7IK(URDF_PATH, tip_link="flange", tool_offset=TOOL_OFFSET)
    
    # Box A (Target)
    box_a_xyz = np.array([0.55, 0.45, 0.10])
    box_a_top_z = box_a_xyz[2] + 0.10
    place_target = np.array([box_a_xyz[0], box_a_xyz[1], box_a_top_z + 0.025 + 0.005])
    transfer = place_target + np.array([0, 0, 0.12])
    
    # Workspace limits (from sim.launch.py)
    r_min, r_max = 0.40, 0.70
    theta_max = math.radians(120)
    
    print("Testing workspace grid (r: 0.40-0.70m, theta: -120 to 120 deg)...")
    print("Grid Size: 15 x 15 = 225 points\n")
    
    successes = 0
    failures = []
    
    # Generate a 15x15 grid
    for r in np.linspace(r_min, r_max, 15):
        for theta in np.linspace(-theta_max, theta_max, 15):
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            
            # Skip if too close to Box A (collision)
            if math.hypot(x - box_a_xyz[0], y - box_a_xyz[1]) < 0.25:
                continue
                
            # Define targets
            box_b_grasp = np.array([x, y, 0.010])
            approach = box_b_grasp + np.array([0, 0, 0.12])
            lift = box_b_grasp + np.array([0, 0, 0.25])
            
            R_down = cp.tool_down_R(yaw=0.0) # Assume 0 yaw for worst-case
            
            # Start seed (pointing at box)
            q_start = np.array([theta, 0.0, 0.0, 0.0, -1.57, 0.0])
            
            # Test segments
            valid, err = test_segment(ik, approach, box_b_grasp, q_start, R_down)
            if not valid:
                failures.append(f"Grasp ({x:.2f}, {y:.2f}): {err}")
                continue
                
            valid, err = test_segment(ik, lift, transfer, q_start, R_down)
            if not valid:
                failures.append(f"Transfer ({x:.2f}, {y:.2f}): {err}")
                continue
                
            successes += 1

    print(f"Successful Pick-and-Places: {successes}")
    print(f"Failed Pick-and-Places: {len(failures)}")
    
    if failures:
        print("\nFailures Details:")
        for f in failures[:10]: # Show first 10
            print(f" - {f}")
    else:
        print("\n✅ PERFECT! The robot can reach and plan trajectories for all valid random positions.")

if __name__ == "__main__":
    main()