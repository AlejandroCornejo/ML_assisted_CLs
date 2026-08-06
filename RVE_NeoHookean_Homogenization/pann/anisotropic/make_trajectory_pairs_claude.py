#!/usr/bin/env python3
"""Five 3D panels of the full macro-strain training domain
(E11, E22, gamma12): the ten training trajectories form five
tension/compression pairs, one pair per shear (gamma12) level. Each
trajectory first ramps from the origin up (or down) to its target shear
level, then sweeps its (E11, E22) zig-zag within that shear plane -- a
genuinely 3D path, not a flat slice -- so every panel is a real 3D plot
to make that visible, with both members of the pair always starting at
the origin.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "trajectories" / "stage_1_training_set_fom"

# (tension trajectory id, compression trajectory id) per shear level,
# read off each trajectory's own final applied-strain gamma12 value.
PAIRS = [
    (1, 2, 0.0),
    (3, 4, 0.05),
    (5, 6, 0.1),
    (7, 8, -0.05),
    (9, 10, -0.1),
]

TENSION_COLOR = "#1f77b4"
COMPRESSION_COLOR = "#d62728"


def load(traj_id: int) -> np.ndarray:
    return np.load(DATA / f"trajectory_{traj_id}" / f"trajectory_{traj_id}_applied_strain.npy")


def main() -> None:
    fig = plt.figure(figsize=(19.5, 4.4))

    for panel_index, (tens_id, comp_id, gamma) in enumerate(PAIRS):
        axis = fig.add_subplot(1, 5, panel_index + 1, projection="3d")
        tens = load(tens_id)
        comp = load(comp_id)

        axis.plot(tens[:, 0], tens[:, 1], tens[:, 2], color=TENSION_COLOR, linewidth=1.8,
                  label=f"Traj. {tens_id} (tension)")
        axis.plot(comp[:, 0], comp[:, 1], comp[:, 2], color=COMPRESSION_COLOR, linewidth=1.8,
                  linestyle="--", label=f"Traj. {comp_id} (compression)")
        axis.scatter([0], [0], [0], color="black", s=18, zorder=5)

        axis.set_xlabel("$E_{11}$", labelpad=2, fontsize=8.5)
        axis.set_ylabel("$E_{22}$", labelpad=2, fontsize=8.5)
        axis.set_zlabel(r"$\gamma_{12}$", labelpad=-2, fontsize=8.5)
        axis.set_title(rf"$\gamma_{{12}}\to{gamma:g}$", fontsize=11, pad=-2)
        axis.set_xlim(0, 2)
        axis.set_ylim(-0.1, 2)
        axis.set_zlim(-0.1, 0.1)
        axis.tick_params(labelsize=6.5, pad=0)
        axis.view_init(elev=18, azim=-60)
        axis.xaxis.pane.set_alpha(0.05)
        axis.yaxis.pane.set_alpha(0.05)
        axis.zaxis.pane.set_alpha(0.05)

    handles = [
        plt.Line2D([], [], color=TENSION_COLOR, linewidth=1.8, label="tension branch ($E_{22}\\to 2$)"),
        plt.Line2D([], [], color=COMPRESSION_COLOR, linewidth=1.8, linestyle="--",
                   label="compression branch ($E_{22}\\to -0.1$)"),
        plt.Line2D([], [], color="black", marker="o", linestyle="None", markersize=5,
                   label="origin (start of every trajectory)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "training_trajectories_pairs_claude.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote training_trajectories_pairs_claude.png")


if __name__ == "__main__":
    main()
