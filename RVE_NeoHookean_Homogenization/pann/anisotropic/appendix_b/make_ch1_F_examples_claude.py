#!/usr/bin/env python3
"""Chapter 1 (Super Appendix B): six worked examples of the deformation
gradient F acting on the reference unit square, covering every regime
discussed in Chapter 1's text: identity, area-doubling stretch,
area-halving compression, area-collapsing (J=0) degeneracy,
orientation-reversing reflection (J<0), and area-preserving shear.
Each panel shows the reference unit square (dashed), its deformed image
under F (solid, filled), and the two column vectors F e1, F e2 as arrows.
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
FILL_COLOR = "#2ca02c"

CASES = [
    ("identity", np.array([[1.0, 0.0], [0.0, 1.0]]), r"$\bm F=\bm I$"),
    ("stretch", np.array([[2.0, 0.0], [0.0, 1.0]]), r"$\bm F=\mathrm{diag}(2,1)$"),
    ("compression", np.array([[0.5, 0.0], [0.0, 0.5]]), r"$\bm F=\mathrm{diag}(0.5,0.5)$"),
    ("collapse", np.array([[2.0, 0.0], [0.0, 0.0]]), r"$\bm F=\mathrm{diag}(2,0)$"),
    ("reflection", np.array([[-1.0, 0.0], [0.0, 1.0]]), r"$\bm F=\mathrm{diag}(-1,1)$"),
    ("shear", np.array([[1.0, 0.6], [0.0, 1.0]]), r"$\bm F=\begin{pmatrix}1&0.6\\0&1\end{pmatrix}$"),
]

REF_SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)


def panel(ax, F, label):
    J = np.linalg.det(F)
    deformed = REF_SQUARE @ F.T

    ax.plot(REF_SQUARE[:, 0], REF_SQUARE[:, 1], linestyle="--", color="#888888",
            linewidth=1.2, zorder=2)
    ax.fill(deformed[:, 0], deformed[:, 1], color=FILL_COLOR, alpha=0.30, zorder=1)
    ax.plot(deformed[:, 0], deformed[:, 1], color=FILL_COLOR, linewidth=1.6, zorder=3)

    e1 = F @ np.array([1.0, 0.0])
    e2 = F @ np.array([0.0, 1.0])
    ax.annotate("", xy=tuple(e1), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=E1_COLOR, linewidth=1.8, mutation_scale=14),
                zorder=4)
    ax.annotate("", xy=tuple(e2), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=E2_COLOR, linewidth=1.8, mutation_scale=14),
                zorder=4)
    if np.linalg.norm(e1) > 1e-6:
        ax.text(e1[0] * 1.12, e1[1] * 1.12, r"$\bm Fe_1$", color=E1_COLOR, fontsize=9, ha="center")
    if np.linalg.norm(e2) > 1e-6:
        ax.text(e2[0] * 1.12, e2[1] * 1.12 + 0.05, r"$\bm Fe_2$", color=E2_COLOR, fontsize=9, ha="center")

    ax.scatter([0], [0], color="black", s=14, zorder=5)
    ax.set_title(label + f"\n$J=\\det\\bm F={J:.2f}$", fontsize=10.5)
    ax.set_xlim(-1.6, 2.4)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.axhline(0, color="#dddddd", linewidth=0.6, zorder=0)
    ax.axvline(0, color="#dddddd", linewidth=0.6, zorder=0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#cccccc")


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 7.4))
    for ax, (_, F, label) in zip(axes.flat, CASES):
        panel(ax, F, label)
    fig.suptitle(
        r"El cuadrado de referencia (l\'{\i}nea discontinua) y su imagen bajo $\bm F$ "
        r"(verde), con las columnas $\bm Fe_1$ (rojo) y $\bm Fe_2$ (azul)",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(HERE / "ch1_F_examples_claude.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote ch1_F_examples_claude.png")


if __name__ == "__main__":
    main()
