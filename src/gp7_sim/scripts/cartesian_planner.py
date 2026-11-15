#!/usr/bin/env python3
"""Straight-line Cartesian path generation with trapezoidal velocity profile."""
import numpy as np


def trapezoidal_profile(s_total, v_max, a_max, dt=0.01):
    if s_total <= 1e-9:
        return (np.array([0.0]), np.array([0.0]),
                np.array([0.0]), np.array([0.0]))

    t_acc = v_max / a_max
    s_acc = 0.5 * a_max * t_acc**2

    if 2 * s_acc >= s_total:
        t_acc = np.sqrt(s_total / a_max)
        t_total = 2 * t_acc
        t = np.arange(0.0, t_total + dt, dt)
        s = np.where(t < t_acc, 0.5 * a_max * t**2,
                     s_total - 0.5 * a_max * (t_total - t)**2)
        s_dot = np.where(t < t_acc, a_max * t, a_max * (t_total - t))
        s_ddot = np.where(t < t_acc, a_max, -a_max)
    else:
        t_flat = (s_total - 2 * s_acc) / v_max
        t_total = 2 * t_acc + t_flat
        t = np.arange(0.0, t_total + dt, dt)
        s = np.zeros_like(t)
        s_dot = np.zeros_like(t)
        s_ddot = np.zeros_like(t)

        m = t < t_acc
        s[m] = 0.5 * a_max * t[m]**2
        s_dot[m] = a_max * t[m]
        s_ddot[m] = a_max

        m = (t >= t_acc) & (t < t_acc + t_flat)
        s[m] = s_acc + v_max * (t[m] - t_acc)
        s_dot[m] = v_max

        m = t >= t_acc + t_flat
        td = t[m] - (t_acc + t_flat)
        s[m] = s_total - s_acc + v_max * td - 0.5 * a_max * td**2
        s_dot[m] = v_max - a_max * td
        s_ddot[m] = -a_max

    return t, s, s_dot, s_ddot


def tool_down_R(yaw=0.0):
    """Tool axis (+X_flange) points to world -Z; fingers along world Y (yaw=0)."""
    R_base = np.array([[0.0, 0.0, 1.0],
                       [0.0, 1.0, 0.0],
                       [-1.0, 0.0, 0.0]])
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ R_base


def straight_line(start_xyz, end_xyz, n_samples):
    start = np.asarray(start_xyz, dtype=float)
    end = np.asarray(end_xyz, dtype=float)
    return np.linspace(start, end, n_samples)