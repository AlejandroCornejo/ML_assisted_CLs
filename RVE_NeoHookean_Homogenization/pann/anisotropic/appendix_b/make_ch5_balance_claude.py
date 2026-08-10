#!/usr/bin/env python3
"""Chapter 5 (Super Appendix B): a balanced direction system. Draws
three undirected directions (lines through the origin, since d and -d
give the same d ox d) at concrete angles, labeled with the weights
that solve the balance system A(theta) w = (1,1,0)^T, verified to give
sum_i w_i d_i ox d_i = I exactly.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}",
})

HERE = Path(__file__).resolve().parent

THETAS_DEG = [0, 55, 130]
COLORS = ["#d62728", "#1f5fa8", "#2ca02c"]


def main() -> None:
    thetas = np.deg2rad(THETAS_DEG)
    A = np.array([[np.cos(t) ** 2, np.sin(t) ** 2, np.sin(t) * np.cos(t)] for t in thetas]).T
    b = np.array([1.0, 1.0, 0.0])
    w = np.linalg.solve(A, b)

    fig, ax = plt.subplots(figsize=(5.6, 5.6))

    circle = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(circle), np.sin(circle), color="#dddddd", linewidth=1.0, zorder=0)

    for theta_deg, theta, wi, color in zip(THETAS_DEG, thetas, w, COLORS):
        d = np.array([np.cos(theta), np.sin(theta)])
        line_len = 1.55
        ax.plot([-line_len * d[0], line_len * d[0]], [-line_len * d[1], line_len * d[1]],
                color=color, linewidth=1.5 + 3.0 * wi, zorder=2, alpha=0.85)
        label_pos = 1.75 * d
        ax.text(label_pos[0], label_pos[1],
                rf"$\theta={theta_deg}^\circ$" "\n" rf"$w={wi:.3f}$",
                color=color, fontsize=10.5, ha="center", va="center")

    ax.scatter([0], [0], color="black", s=20, zorder=3)
    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-2.3, 2.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(r"Sistema balanceado de 3 direcciones: $\sum_i w_i\,\bm d_i\otimes\bm d_i=\bm I$" "\n"
                 r"(grosor de la l\'inea $\propto$ peso $w_i$)", fontsize=12)
    fig.tight_layout()
    fig.savefig(HERE / "ch5_balance_claude.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote ch5_balance_claude.png")
    print("weights:", w)


if __name__ == "__main__":
    main()
