#!/usr/bin/env python3
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = "pick_place_log.csv"
FIG_PATH = "traj_plots_per_joint.png"

def smooth(data, window=15):
    """Simple moving average to clean up acceleration noise."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='same')

def main():
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    t   = np.array([float(r["t"])       for r in rows])
    q   = np.array([[float(r[f"q{j}"])  for j in range(6)] for r in rows])
    qd  = np.array([[float(r[f"qd{j}"]) for j in range(6)] for r in rows])
    qdd = np.array([[float(r[f"qdd{j}"]) for j in range(6)] for r in rows])

    # Smooth acceleration for visualization
    for j in range(6):
        qdd[:, j] = smooth(qdd[:, j], window=15)

    # Create a 6x3 grid (6 joints, 3 columns for Pos/Vel/Acc)
    fig, axes = plt.subplots(6, 3, figsize=(18, 20), sharex=True)
    
    # Segment background shading boundaries
    seg_idx = [i for i in range(1, len(rows)) if rows[i]["segment"] != rows[i-1]["segment"]]

    for j in range(6):
        # Position (Left Column)
        axes[j, 0].plot(t, q[:, j], color='tab:blue')
        axes[j, 0].set_ylabel(f"Joint {j+1}\nPos [rad]", fontsize=11, fontweight='bold')
        
        # Velocity (Middle Column)
        axes[j, 1].plot(t, qd[:, j], color='tab:orange')
        axes[j, 1].set_ylabel("Vel [rad/s]", fontsize=11, fontweight='bold')
        
        # Acceleration (Right Column)
        axes[j, 2].plot(t, qdd[:, j], color='tab:green')
        axes[j, 2].set_ylabel("Acc [rad/s²]", fontsize=11, fontweight='bold')

        # Add grey shading to alternate phases for readability
        for idx, i in enumerate(seg_idx):
            if idx % 2 == 0:
                for col in range(3):
                    axes[j, col].axvspan(t[i-1], t[i], color='gray', alpha=0.15)

        # Formatting
        for col in range(3):
            axes[j, col].grid(True, linestyle='--', alpha=0.7)
            if j == 0:
                axes[j, col].set_title(["Position", "Velocity", "Acceleration"][col], 
                                       fontsize=14, fontweight='bold', pad=10)
            if j == 5:
                axes[j, col].set_xlabel("Time [s]", fontsize=12)

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150)
    print(f"Wrote {FIG_PATH}")

if __name__ == "__main__":
    main()