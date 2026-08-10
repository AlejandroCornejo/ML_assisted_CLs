#!/usr/bin/env python3
"""The four closed strain loops used for the (C1) cyclic-work test
(Table~\\ref{tab:cyclic}), read directly from the same JSON the test
itself produced (results/c1_cyclic_claude_metrics.json), not redrawn
from memory. Four panels: one genuine 3D view of all four loops in
(E11, E22, gamma12) space, plus the three pairwise 2D projections, so
every loop's true shape is visible from at least one panel without
ambiguity (a loop that is flat in one 2D projection is a real 3D curve
in the other views).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

LOOP_LABELS = {
    "small_normal_square": "Small normal square",
    "origin_ellipse": "Origin-anchored ellipse",
    "shear_loop": "Shear loop",
    "mixed_triangle": "Mixed triangle",
}
LOOP_COLORS = {
    "small_normal_square": "#2ca02c",
    "origin_ellipse": "#9467bd",
    "shear_loop": "#1f77b4",
    "mixed_triangle": "#d62728",
}


def main() -> None:
    metrics = json.loads((RESULTS / "c1_cyclic_claude_metrics.json").read_text(encoding="utf-8"))
    loops = dict(metrics["loops_definition_physical_strain"])
    loops["origin_ellipse"] = metrics["origin_ellipse_path_for_plotting"]
    loop_order = metrics.get("loop_order", list(LOOP_LABELS))

    fig = plt.figure(figsize=(12.5, 9.6))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax_ee = fig.add_subplot(2, 2, 2)
    ax_eg = fig.add_subplot(2, 2, 3)
    ax_2g = fig.add_subplot(2, 2, 4)

    for key in loop_order:
        vertices = np.asarray(loops[key])
        e11, e22, g12 = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        color = LOOP_COLORS[key]
        label = LOOP_LABELS[key]
        ax3d.plot(e11, e22, g12, color=color, linewidth=2.0, label=label)
        ax_ee.plot(e11, e22, color=color, linewidth=2.0, marker="o", markersize=2.5, label=label)
        ax_eg.plot(e11, g12, color=color, linewidth=2.0, marker="o", markersize=2.5, label=label)
        ax_2g.plot(e22, g12, color=color, linewidth=2.0, marker="o", markersize=2.5, label=label)

    ax3d.scatter([0], [0], [0], color="black", s=35, zorder=5)
    ax3d.set_xlabel(r"$E_{11}$")
    ax3d.set_ylabel(r"$E_{22}$")
    ax3d.set_zlabel(r"$\gamma_{12}$")
    ax3d.set_title("3D view")
    ax3d.view_init(elev=22, azim=-60)

    panels = [
        (ax_ee, "e11", "e22", r"$E_{11}$", r"$E_{22}$", r"$(E_{11},E_{22})$ projection", (0, 1)),
        (ax_eg, "e11", "g12", r"$E_{11}$", r"$\gamma_{12}$", r"$(E_{11},\gamma_{12})$ projection", (0, 2)),
        (ax_2g, "e22", "g12", r"$E_{22}$", r"$\gamma_{12}$", r"$(E_{22},\gamma_{12})$ projection", (1, 2)),
    ]
    for ax, _, _, xlabel, ylabel, title, _ in panels:
        ax.scatter([0], [0], color="black", s=35, zorder=5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axhline(0, color="#dddddd", linewidth=0.8, zorder=0)
        ax.axvline(0, color="#dddddd", linewidth=0.8, zorder=0)
        ax.set_aspect("equal", adjustable="datalim")
    ax_ee.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    fig.suptitle("The four closed strain loops of the (C1) cyclic-work test")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(HERE / "c1_cyclic_loops_claude.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote c1_cyclic_loops_claude.pdf")


if __name__ == "__main__":
    main()
