#!/usr/bin/env python3
"""Un-meshed, dimensioned schematic of the RVE geometry: square unit cell
with a centered circular hole, annotated with the actual numeric
parameters read off core/rve_geometry.mdpa (side length 2, hole radius
0.5), for readers who want the plain geometric definition without a
finite-element mesh overlay.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

HERE = Path(__file__).resolve().parent

HALF_SIDE = 1.0          # square spans [-1, 1] x [-1, 1] -> side length 2
HOLE_RADIUS = 0.5


def main() -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.6))

    square = plt.Rectangle((-HALF_SIDE, -HALF_SIDE), 2 * HALF_SIDE, 2 * HALF_SIDE,
                            fill=False, edgecolor="black", linewidth=1.6)
    ax.add_patch(square)
    ax.add_patch(Circle((0.0, 0.0), HOLE_RADIUS, fill=False, edgecolor="black", linewidth=1.6))

    # --- dimension line: total side length (bottom edge) ---
    y_dim = -HALF_SIDE - 0.38
    ax.annotate("", xy=(HALF_SIDE, y_dim), xytext=(-HALF_SIDE, y_dim),
                arrowprops=dict(arrowstyle="<->", linewidth=1.0))
    for x in (-HALF_SIDE, HALF_SIDE):
        ax.plot([x, x], [-HALF_SIDE, y_dim], color="gray", linewidth=0.7, linestyle=(0, (3, 2)))
    ax.text(0.0, y_dim - 0.16, "side length $= 2$", ha="center", va="top", fontsize=10)

    # --- dimension line: hole radius, pointing straight down so the label
    # only ever has to clear the bottom edge (not a corner), with plenty of
    # margin from both the circle and the square boundary ---
    ang = np.deg2rad(-90)
    rx, ry = HOLE_RADIUS * np.cos(ang), HOLE_RADIUS * np.sin(ang)
    ax.annotate("", xy=(rx, ry), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="->", linewidth=1.0, color="#1f77b4"))
    label_x, label_y = 0.0, ry - 0.20
    ax.plot([rx, label_x], [ry, label_y], color="#1f77b4", linewidth=0.7)
    ax.text(label_x + 0.06, label_y, "$r = 0.5$", color="#1f77b4", fontsize=10,
            ha="left", va="center")
    ax.plot(0, 0, marker="+", color="#1f77b4", markersize=7)

    # --- axes through the center, reference frame ---
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.annotate("", xy=(HALF_SIDE + 0.28, 0), xytext=(HALF_SIDE - 0.05, 0),
                arrowprops=dict(arrowstyle="->", linewidth=1.0))
    ax.text(HALF_SIDE + 0.32, -0.05, "$X$", fontsize=11, va="top")
    ax.annotate("", xy=(0, HALF_SIDE + 0.28), xytext=(0, HALF_SIDE - 0.05),
                arrowprops=dict(arrowstyle="->", linewidth=1.0))
    ax.text(0.06, HALF_SIDE + 0.30, "$Y$", fontsize=11)

    # --- corner coordinate labels ---
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        ax.text(sx * HALF_SIDE + 0.06 * sx, sy * HALF_SIDE + 0.10 * sy,
                 f"$({sx:+.0f},{sy:+.0f})$", fontsize=7.5, color="gray",
                 ha="left" if sx > 0 else "right")

    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.9, 1.75)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(HERE / "rve_dimensions_claude.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote rve_dimensions_claude.png")


if __name__ == "__main__":
    main()
