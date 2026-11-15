#!/usr/bin/env python3
"""Single point of truth for IK backend selection."""
import os

BACKEND = os.environ.get("GP7_IK_BACKEND", "kdl").lower()

if BACKEND == "analytical":
    from gp7_ik_analytical import GP7IK
elif BACKEND == "kdl":
    from gp7_ik_kdl import GP7IK
else:
    raise RuntimeError(f"Unknown GP7_IK_BACKEND='{BACKEND}'. Use 'kdl' or 'analytical'.")

__all__ = ["GP7IK", "BACKEND"]
