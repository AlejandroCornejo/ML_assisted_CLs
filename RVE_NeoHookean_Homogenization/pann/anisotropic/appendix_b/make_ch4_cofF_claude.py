#!/usr/bin/env python3
"""Chapter 4 (Super Appendix B): the physical meaning of Cof F. Uses
the shear F=[[1,0.6],[0,1]] from Chapter 1. Left panel: a reference
line segment with its tangent t=(1,1) and normal n=(1,-1) (t.n=0).
Right panel: the deformed tangent F t (correct: tangents transform
with F), the deformed normal via Cof(F) n (correct: stays exactly
perpendicular to F t), and, for contrast, the naive/wrong F n (does
NOT stay perpendicular to F t) -- making concrete why normals need a
different transformation rule than tangents.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}\newcommand{\Cof}{\operatorname{cof}}",
})

HERE = Path(__file__).resolve().parent

F = np.array([[1.0, 0.6], [0.0, 1.0]])
COF_F = np.array([[F[1, 1], -F[1, 0]], [-F[0, 1], F[0, 0]]])

t_ref = np.array([1.0, 1.0])
n_ref = np.array([1.0, -1.0])

t_def = F @ t_ref
n_def_correct = COF_F @ n_ref
n_def_wrong = F @ n_ref

T_COLOR = "#1f5fa8"
N_COLOR = "#2ca02c"
WRONG_COLOR = "#d62728"


def draw_square(ax, Fmat, color):
    sq = np.array([[-1.4, -1.4], [1.4, -1.4], [1.4, 1.4], [-1.4, 1.4], [-1.4, -1.4]])
    img = sq @ Fmat.T
    ax.plot(img[:, 0], img[:, 1], color=color, linewidth=0.7, alpha=0.35, zorder=0)


def arrow(ax, v, color, label, label_offset=(0.1, 0.1), scale=1.0, ls="-"):
    v = v * scale
    ax.annotate("", xy=tuple(v), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=2.2, mutation_scale=16, linestyle=ls),
                zorder=5)
    ax.text(v[0] + label_offset[0], v[1] + label_offset[1], label, color=color, fontsize=11)


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 5.0))

    draw_square(ax1, np.eye(2), "#999999")
    arrow(ax1, t_ref, T_COLOR, r"$\bm t=(1,1)$", (0.08, 0.05))
    arrow(ax1, n_ref, N_COLOR, r"$\bm n=(1,-1)$", (0.08, -0.15))
    ax1.set_title(r"Referencia: $\bm t\cdot\bm n=0$" "\n" r"(l\'inea y su normal)")

    draw_square(ax2, F, "#999999")
    arrow(ax2, t_def, T_COLOR, r"$\bm F\bm t=(1.6,1)$", (0.08, 0.05))
    arrow(ax2, n_def_correct, N_COLOR, r"$\Cof(\bm F)\bm n=(1,-1.6)$", (0.08, -0.25))
    arrow(ax2, n_def_wrong, WRONG_COLOR, r"$\bm F\bm n=(0.4,-1)$ (\emph{incorrecto})", (-2.9, -0.35), ls="--")
    ax2.set_title(r"Deformado: $(\Cof(\bm F)\bm n)\cdot(\bm F\bm t)=0$ exactamente" "\n"
                  r"pero $(\bm F\bm n)\cdot(\bm F\bm t)=-0.36\neq0$")

    for ax in (ax1, ax2):
        ax.axhline(0, color="#dddddd", linewidth=0.6, zorder=0)
        ax.axvline(0, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_xlim(-2.4, 2.4)
        ax.set_ylim(-2.4, 2.4)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#cccccc")

    fig.suptitle(r"$\bm F=\begin{pmatrix}1&0.6\\0&1\end{pmatrix}$ (el corte del Cap\'itulo 1): "
                 r"las tangentes se transforman con $\bm F$, las normales con $\Cof(\bm F)$",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(HERE / "ch4_cofF_claude.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote ch4_cofF_claude.png")
    print("F =", F.tolist())
    print("Cof(F) =", COF_F.tolist())
    print("t_def =", t_def, "n_def_correct =", n_def_correct, "n_def_wrong =", n_def_wrong)
    print("dot(correct) =", n_def_correct @ t_def)
    print("dot(wrong)   =", n_def_wrong @ t_def)


if __name__ == "__main__":
    main()
