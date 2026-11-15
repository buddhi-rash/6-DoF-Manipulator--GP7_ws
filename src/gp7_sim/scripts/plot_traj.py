#!/usr/bin/env python3
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = "pick_place_log.csv"
FIG_PATH = "traj_plots.png"


def main():
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    t   = np.array([float(r["t"])       for r in rows])
    q   = np.array([[float(r[f"q{j}"])  for j in range(6)] for r in rows])
    qd  = np.array([[float(r[f"qd{j}"]) for j in range(6)] for r in rows])
    qdd = np.array([[float(r[f"qdd{j}"]) for j in range(6)] for r in rows])

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for j in range(6):
        axes[0].plot(t, q[:, j], label=f"q{j+1}")
        axes[1].plot(t, qd[:, j], label=f"qd{j+1}")
        axes[2].plot(t, qdd[:, j], label=f"qdd{j+1}")
    axes[0].set_ylabel("Position [rad]")
    axes[1].set_ylabel("Velocity [rad/s]")
    axes[2].set_ylabel("Acceleration [rad/s^2]")
    axes[2].set_xlabel("Time [s]")
    for ax in axes:
        ax.legend(loc="upper right", ncol=6, fontsize=7)
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150)
    print(f"Wrote {FIG_PATH}")


if __name__ == "__main__":
    main()
