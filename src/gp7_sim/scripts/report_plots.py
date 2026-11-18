#!/usr/bin/env python3
"""Report figures + diagnostics from the pick_place_log.csv written by pick_place.py.

    python3 report_plots.py pick_place_log.csv                 # figures go to ./report_figs
    python3 report_plots.py pick_place_log.csv --tool-offset 0.14 --outdir report_figs

Needs gp7_dh.py (the DH model) in the same folder. Produces
    tcp_profile.png            TCP position, speed and acceleration, segments labelled
                               (this is where the trapezoidal velocity profile is visible)
    joint_speed_vs_spec.png    joint speeds from the logged positions against the GP7 spec limits
    joint_position_vs_limits.png
and prints: segment durations, peak TCP speed, peak joint speed as % of spec, joint-limit check and
the jump in position / velocity at every segment boundary (a smooth plan has ~0 jump and ~0 speed there).
"""
import argparse
import csv
import math
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gp7_dh as dh

# GP7 spec / URDF values [deg]
Q_LO = np.array([-170, -65, -70, -190, -135, -360.0])
Q_HI = np.array([170, 145, 190, 190, 135, 360.0])
QD_MAX = np.array([375, 315, 410, 550, 550, 1000.0])      # deg/s


def smooth(x, w=5):
    if len(x) < w:
        return x
    k = np.ones(w) / w
    pad = w // 2
    xp = np.pad(x, ((pad, pad),) + ((0, 0),) * (x.ndim - 1), mode='edge')
    return np.column_stack([np.convolve(xp[:, j], k, mode='valid') for j in range(x.shape[1])]) \
        if x.ndim > 1 else np.convolve(xp, k, mode='valid')


def load(path):
    rows = list(csv.DictReader(open(path)))
    t = np.array([float(r['t']) for r in rows])
    seg = [r['segment'] for r in rows]
    q = np.array([[float(r[f'q{j}']) for j in range(6)] for r in rows])
    return t, seg, q


def shade(ax, t, seg):
    names, starts = [], []
    for i, s in enumerate(seg):
        if i == 0 or s != seg[i - 1]:
            names.append(s)
            starts.append(t[i])
    ends = starts[1:] + [t[-1]]
    for k, (a, b, n) in enumerate(zip(starts, ends, names)):
        if k % 2 == 0:
            ax.axvspan(a, b, color='0.92', zorder=0)
        ax.axvline(a, color='0.7', lw=0.7, zorder=0)
    return names, starts, ends


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv')
    ap.add_argument('--tool-offset', type=float, default=0.14, help='flange -> grasp point [m]')
    ap.add_argument('--outdir', default='report_figs')
    ap.add_argument('--vmax', type=float, default=0.20, help='planned Cartesian speed limit [m/s] (for the plot)')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    t, seg, q = load(a.csv)
    dt = float(np.median(np.diff(t)))
    P = np.array([dh.fk_tcp(qi, a.tool_offset)[:3, 3] for qi in q])      # TCP position from the DH model
    v = np.gradient(smooth(P), dt, axis=0)
    speed = np.linalg.norm(v, axis=1)
    acc = np.linalg.norm(np.gradient(smooth(v, 7), dt, axis=0), axis=1)
    qd = np.degrees(np.gradient(smooth(q), dt, axis=0))                 # deg/s from the logged positions
    qdeg = np.degrees(q)

    # ------------------------------------------------------------------ figure 1: TCP profile
    fig, axs = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    for k, c in enumerate('xyz'):
        axs[0].plot(t, P[:, k], label=f'TCP {c}')
    axs[0].set_ylabel('position [m]'); axs[0].legend(ncol=3, loc='upper right')
    axs[1].plot(t, speed, color='tab:orange')
    axs[1].axhline(a.vmax, color='r', ls='--', lw=0.8, label=f'v_max = {a.vmax} m/s')
    axs[1].set_ylabel('TCP speed [m/s]'); axs[1].legend(loc='upper right')
    axs[2].plot(t, acc, color='tab:green'); axs[2].set_ylabel('TCP acceleration [m/s²]')
    for ax in axs:
        names, starts, ends = shade(ax, t, seg); ax.grid(alpha=.3)
    for n, s, e in zip(names, starts, ends):
        axs[0].text((s + e) / 2, axs[0].get_ylim()[1], n, ha='center', va='bottom', fontsize=9)
    axs[2].set_xlabel('time [s]')
    fig.suptitle('Tool-centre-point motion: straight-line segments with a trapezoidal speed profile', y=0.995)
    fig.tight_layout(); fig.savefig(f'{a.outdir}/tcp_profile.png', dpi=140); plt.close(fig)

    # ------------------------------------------------------------ figure 2: joint speed vs spec
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
    for j, ax in enumerate(axs.flat):
        ax.plot(t, qd[:, j], lw=1.2)
        ax.axhline(QD_MAX[j], color='r', ls='--', lw=0.8); ax.axhline(-QD_MAX[j], color='r', ls='--', lw=0.8)
        shade(ax, t, seg); ax.grid(alpha=.3)
        ax.set_title(f'J{j + 1}: peak {np.abs(qd[:, j]).max():.0f} deg/s = {100 * np.abs(qd[:, j]).max() / QD_MAX[j]:.0f} % of spec ({QD_MAX[j]:.0f})', fontsize=10)
        ax.set_ylabel('deg/s')
    for ax in axs[-1]:
        ax.set_xlabel('time [s]')
    fig.tight_layout(); fig.savefig(f'{a.outdir}/joint_speed_vs_spec.png', dpi=140); plt.close(fig)

    # ------------------------------------------------------ figure 3: joint positions vs limits
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
    for j, ax in enumerate(axs.flat):
        ax.plot(t, qdeg[:, j], lw=1.2)
        ax.axhline(Q_LO[j], color='r', ls='--', lw=0.8); ax.axhline(Q_HI[j], color='r', ls='--', lw=0.8)
        shade(ax, t, seg); ax.grid(alpha=.3); ax.set_title(f'J{j + 1}  (limits {Q_LO[j]:.0f} … {Q_HI[j]:.0f} deg)', fontsize=10)
        ax.set_ylabel('deg')
    for ax in axs[-1]:
        ax.set_xlabel('time [s]')
    fig.tight_layout(); fig.savefig(f'{a.outdir}/joint_position_vs_limits.png', dpi=140); plt.close(fig)

    # --------------------------------------------------------------------------- diagnostics
    print(f'{len(t)} samples, dt {dt:.3f} s, duration {t[-1]:.2f} s')
    print('\nsegment      duration [s]   path length [m]   peak TCP speed [m/s]')
    names, starts, ends = shade(plt.figure().add_subplot(), t, seg)
    plt.close('all')
    for n, s, e in zip(names, starts, ends):
        m = (t >= s) & (t <= e)
        L = float(np.sum(np.linalg.norm(np.diff(P[m], axis=0), axis=1)))
        print(f'{n:<12} {e - s:10.2f} {L:15.3f} {speed[m].max():18.3f}')
    print('\njoint | peak speed [deg/s] | % of spec | min / max position [deg] | inside limits?')
    for j in range(6):
        ok = bool(np.all(qdeg[:, j] >= Q_LO[j] - 1e-6) and np.all(qdeg[:, j] <= Q_HI[j] + 1e-6))
        print(f'  J{j + 1}  | {np.abs(qd[:, j]).max():14.1f}     | {100 * np.abs(qd[:, j]).max() / QD_MAX[j]:7.1f}   | '
              f'{qdeg[:, j].min():8.1f} / {qdeg[:, j].max():8.1f}      | {"yes" if ok else "NO  <-- violates the limit"}')
    print('\nsegment boundaries (smooth plan: position jump ~0 and speed ~0):')
    print('boundary                  jump in q [deg]   max |joint speed| just after [deg/s]   TCP speed just after [m/s]')
    for i in range(1, len(seg)):
        if seg[i] != seg[i - 1]:
            jump = np.degrees(np.abs(q[i] - q[i - 1]).max())
            after = np.abs(np.degrees((q[min(i + 2, len(q) - 1)] - q[i]) / (2 * dt))).max()
            flag = '   <-- check' if jump > 0.2 or after > 15 else ''
            print(f'{seg[i - 1]:>9} -> {seg[i]:<9} {jump:14.3f} {after:30.1f} {speed[min(i + 2, len(q) - 1)]:30.3f}{flag}')
    print(f'\nfigures written to {a.outdir}/')


if __name__ == '__main__':
    main()