#!/usr/bin/env python3
"""The held-out test trajectory (Section 2.2), shown purely as a
strain-space path -- no model predictions here, just the geometry of
the macro-strain sequence every reported accuracy number is evaluated
on. One 3D view plus three pairwise 2D projections, read directly from
the real applied-strain array used for this trajectory throughout the
project (not re-derived or re-sampled).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}",
})

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
STRAIN_PATH = (
    REPO_ROOT / "studies" / "fom_tangent_stability_test" / "reference_states"
    / "stage10_mixed_applied_strain.npy"
)

PATH_COLOR = "#6a3d9a"
START_COLOR = "black"


def main() -> None:
    strain = np.load(STRAIN_PATH)
    e11, e22, g12 = strain[:, 0], strain[:, 1], strain[:, 2]

    fig = plt.figure(figsize=(12.5, 3.4))

    ax3d = fig.add_subplot(1, 4, 1, projection="3d")
    ax3d.plot(e11, e22, g12, color=PATH_COLOR, linewidth=1.6)
    ax3d.scatter([e11[0]], [e22[0]], [g12[0]], color=START_COLOR, s=22, zorder=5)
    ax3d.set_xlabel(r"$E_{11}$", labelpad=1, fontsize=8.5)
    ax3d.set_ylabel(r"$E_{22}$", labelpad=1, fontsize=8.5)
    ax3d.set_zlabel(r"$\gamma_{12}$", labelpad=-3, fontsize=8.5)
    ax3d.tick_params(labelsize=6.5, pad=0)
    ax3d.view_init(elev=18, azim=-60)
    ax3d.set_title("3D path", fontsize=9.5)

    projections = [
        (e11, e22, r"$E_{11}$", r"$E_{22}$", "$E_{11}$-$E_{22}$"),
        (e11, g12, r"$E_{11}$", r"$\gamma_{12}$", "$E_{11}$-$\\gamma_{12}$"),
        (e22, g12, r"$E_{22}$", r"$\gamma_{12}$", "$E_{22}$-$\\gamma_{12}$"),
    ]
    for i, (x, y, xlabel, ylabel, title) in enumerate(projections):
        ax = fig.add_subplot(1, 4, i + 2)
        ax.plot(x, y, color=PATH_COLOR, linewidth=1.6)
        ax.scatter([x[0]], [y[0]], color=START_COLOR, s=22, zorder=5,
                   label="origin (start/end)" if i == 0 else None)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9.5)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)

    fig.suptitle("The held-out test trajectory: a single mixed loading-unloading cycle",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(HERE / "test_trajectory_claude.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote test_trajectory_claude.pdf")
    print("n_samples:", strain.shape[0])
    print("E11 range:", e11.min(), e11.max())
    print("E22 range:", e22.min(), e22.max())
    print("g12 range:", g12.min(), g12.max())


if __name__ == "__main__":
    main()
