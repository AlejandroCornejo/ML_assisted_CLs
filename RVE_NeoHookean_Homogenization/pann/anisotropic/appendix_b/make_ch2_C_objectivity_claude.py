#!/usr/bin/env python3
"""Chapter 2 (Super Appendix B): why F alone is not objective, and why
C = F^T F is. Two panels show the unit circle mapped by the same
material stretch F, once directly and once with an extra rigid
rotation Q applied on top (F' = QF): the ellipse has the identical
shape in both cases (same principal stretches), just rotated in space.
Both panels report the numerically computed C = F^T F to make the
invariance concrete, not just asserted.
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

E1_COLOR = "#d62728"
E2_COLOR = "#1f5fa8"
FILL_COLOR = "#9467bd"

F = np.array([[2.0, 0.5], [0.0, 1.0]])
THETA_DEG = 40.0
theta = np.deg2rad(THETA_DEG)
Q = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
F2 = Q @ F

t = np.linspace(0, 2 * np.pi, 400)
circle = np.stack([np.cos(t), np.sin(t)], axis=1)


def panel(ax, Fmat, label):
    ellipse = circle @ Fmat.T
    ax.plot(circle[:, 0], circle[:, 1], linestyle="--", color="#888888", linewidth=1.1, zorder=1)
    ax.plot(ellipse[:, 0], ellipse[:, 1], color=FILL_COLOR, linewidth=1.8, zorder=2)
    ax.fill(ellipse[:, 0], ellipse[:, 1], color=FILL_COLOR, alpha=0.18, zorder=1)

    e1 = Fmat @ np.array([1.0, 0.0])
    e2 = Fmat @ np.array([0.0, 1.0])
    ax.annotate("", xy=tuple(e1), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=E1_COLOR, linewidth=1.8, mutation_scale=14), zorder=3)
    ax.annotate("", xy=tuple(e2), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=E2_COLOR, linewidth=1.8, mutation_scale=14), zorder=3)
    ax.scatter([0], [0], color="black", s=14, zorder=4)

    C = Fmat.T @ Fmat
    c_str = (rf"$\bm C=\bm F^T\bm F=\begin{{pmatrix}}{C[0,0]:.2f}&{C[0,1]:.2f}"
             rf"\\{C[1,0]:.2f}&{C[1,1]:.2f}\end{{pmatrix}}$")
    ax.text(0.0, -2.55, c_str, ha="center", fontsize=10.5)

    ax.set_title(label, fontsize=11)
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-2.9, 2.6)
    ax.set_aspect("equal")
    ax.axhline(0, color="#dddddd", linewidth=0.6, zorder=0)
    ax.axvline(0, color="#dddddd", linewidth=0.6, zorder=0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#cccccc")


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.4))
    panel(axes[0], F, r"Solo $\bm F$" "\n" r"($\bm F\bm e_1$ rojo, $\bm F\bm e_2$ azul)")
    panel(axes[1], F2, rf"$\bm F'=\bm Q\bm F$, rotaci\'on de ${THETA_DEG:.0f}^\circ$" "\n"
                       r"(misma forma, distinta orientaci\'on)")
    fig.suptitle(
        r"El c\'irculo unidad deformado por $\bm F$ (izquierda) y por $\bm Q\bm F$ (derecha): "
        "\n"
        r"la elipse es id\'entica en forma, solo rotada -- y $\bm C=\bm F^T\bm F$ no distingue los dos casos",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(HERE / "ch2_C_objectivity_claude.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote ch2_C_objectivity_claude.png")
    print("F =\n", F)
    print("Q =\n", Q)
    print("F2 = QF =\n", F2)
    print("C1 = F^T F =\n", F.T @ F)
    print("C2 = F2^T F2 =\n", F2.T @ F2)


if __name__ == "__main__":
    main()
