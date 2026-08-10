#!/usr/bin/env python3
"""Chapter 5 (Super Appendix B): why classical invariants (I_1, I_2)
are not enough for an anisotropic material. C1 = diag(2, 0.5) and
C2 = Q C1 Q^T (same eigenvalues, rotated 45 degrees) have identical
I_1 = tr(C) and I_2 = det(C). But a FIXED direction d = e_1 (e.g. the
material's fixed fiber direction) sees a different stretch in each:
d^T C1 d = 2, d^T C2 d = 1.25. The figure draws both stretch ellipses
(image of the unit circle under U = sqrt(C)) with the same fixed ray d
marked in both, and the image point U d highlighted.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}\newcommand{\tr}{\operatorname{tr}}",
})

HERE = Path(__file__).resolve().parent

C1 = np.diag([2.0, 0.5])
theta = np.deg2rad(45)
Q = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
C2 = Q @ C1 @ Q.T
D_FIXED = np.array([1.0, 0.0])

t = np.linspace(0, 2 * np.pi, 300)
circle = np.stack([np.cos(t), np.sin(t)], axis=1)


def panel(ax, C, label):
    U = sqrtm(C).real
    ellipse = circle @ U.T
    ax.plot(ellipse[:, 0], ellipse[:, 1], color="#9467bd", linewidth=1.8, zorder=2)
    ax.fill(ellipse[:, 0], ellipse[:, 1], color="#9467bd", alpha=0.15, zorder=1)

    Ud = U @ D_FIXED
    ax.annotate("", xy=tuple(Ud), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#d62728", linewidth=2.2, mutation_scale=16), zorder=4)
    ax.plot([0, 2.0], [0, 0], color="#888888", linestyle=":", linewidth=1.3, zorder=1)
    ax.scatter([0], [0], color="black", s=18, zorder=5)

    dCd = D_FIXED @ C @ D_FIXED
    ax.text(Ud[0] + 0.08, Ud[1] + 0.30, rf"$\bm U\bm d$" "\n" rf"$|\bm U\bm d|^2=\bm d\cdot\bm C\bm d={dCd:.2f}$",
            color="#d62728", fontsize=10.5)
    ax.text(1.55, -0.18, r"$\bm d$ (fija)", color="#888888", fontsize=10.5)

    ax.text(-1.85, 1.6, rf"$I_1=\tr\bm C={np.trace(C):.1f}$" "\n" rf"$I_2=\det\bm C={np.linalg.det(C):.1f}$",
            fontsize=11, color="#333333")
    ax.set_title(label, fontsize=12)
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect("equal")
    ax.axhline(0, color="#eeeeee", linewidth=0.6, zorder=0)
    ax.axvline(0, color="#eeeeee", linewidth=0.6, zorder=0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#cccccc")


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.2))
    panel(axes[0], C1, r"$\bm C_1=\mathrm{diag}(2,0.5)$")
    panel(axes[1], C2, r"$\bm C_2=\bm Q\bm C_1\bm Q^T$, $\bm Q$ rotaci\'on de $45^\circ$")
    fig.suptitle(r"Mismos $I_1,I_2$ (misma elipse, solo rotada) -- pero $\bm d\cdot\bm C\bm d$ distinto "
                 r"para la misma direcci\'on fija $\bm d$", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(HERE / "ch5_invariants_claude.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote ch5_invariants_claude.png")


if __name__ == "__main__":
    main()
