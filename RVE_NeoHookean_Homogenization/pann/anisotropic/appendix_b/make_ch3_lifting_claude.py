#!/usr/bin/env python3
"""Chapter 3 (Super Appendix B): non-convexity of det(F) and Ball's
lifting trick. Left panel: the saddle h(a,d)=ad over diagonal matrices,
with the running example F1=diag(2,0.5), F2=diag(0.5,2) and their
midpoint marked, connected by the straight segment used throughout the
chapter. Right panel: h(F(t)) along that exact segment (solid) versus
the flat chord connecting the two endpoint values (dashed) -- the gap
between them, shaded, is both the concrete non-convexity violation and
(reinterpreted with the added delta-axis) the lifting-space distinction
between a straight chord (delta stays 1) and the curved physical path
(delta = det F(t) bulges to 1.5625).
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

F1 = (2.0, 0.5)
F2 = (0.5, 2.0)
MID = (1.25, 1.25)


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.0))

    # --- Panel A: the saddle h(a,d) = a*d, with the running example ---
    a = np.linspace(-3, 3, 400)
    d = np.linspace(-3, 3, 400)
    A, D = np.meshgrid(a, d)
    H = A * D
    im = ax1.contourf(A, D, H, levels=30, cmap="RdBu_r", vmin=-6, vmax=6)
    ax1.contour(A, D, H, levels=[0], colors="black", linewidths=1.0)
    fig.colorbar(im, ax=ax1, shrink=0.85, label=r"$h(a,d)=ad$")

    seg_a = [F1[0], F2[0]]
    seg_d = [F1[1], F2[1]]
    ax1.plot(seg_a, seg_d, color="black", linewidth=1.6, zorder=5)
    ax1.scatter(*F1, color="#d62728", s=45, zorder=6)
    ax1.scatter(*F2, color="#1f5fa8", s=45, zorder=6)
    ax1.scatter(*MID, color="#2ca02c", s=55, marker="D", zorder=6)
    ax1.annotate(r"$\bm F_1$, $h=1$", F1,
                 xytext=(F1[0] + 0.15, F1[1] - 0.55), fontsize=9.5, color="#d62728")
    ax1.annotate(r"$\bm F_2$, $h=1$", F2,
                 xytext=(F2[0] - 1.55, F2[1] + 0.30), fontsize=9.5, color="#1f5fa8")
    ax1.annotate(r"punto medio, $h=1.5625$", MID,
                 xytext=(MID[0] - 0.15, MID[1] + 0.55), fontsize=9.5, color="#2ca02c")
    ax1.set_xlabel(r"$a$")
    ax1.set_ylabel(r"$d$")
    ax1.set_title(r"$h(a,d)=ad$ restringida a $\bm F=\mathrm{diag}(a,d)$: una silla de montar")
    ax1.set_aspect("equal")

    # --- Panel B: h(F(t)) along the segment vs. the flat chord ---
    t = np.linspace(0, 1, 200)
    h_t = 1.0 + 2.25 * t * (1.0 - t)
    chord = np.ones_like(t)

    ax2.plot(t, h_t, color="#2ca02c", linewidth=2.2,
             label=r"camino f\'isico: $\det\bm F(t)=1+2.25\,t(1-t)$")
    ax2.plot(t, chord, color="#888888", linestyle="--", linewidth=1.8,
             label=r"cuerda recta entre los extremos: $1$")
    ax2.fill_between(t, chord, h_t, color="#2ca02c", alpha=0.15)
    ax2.scatter([0, 0.5, 1], [1, 1.5625, 1], color=["#d62728", "#2ca02c", "#1f5fa8"], zorder=5, s=45)
    ax2.annotate(r"$t=0$: $\bm F_1$", (0, 1), xytext=(0.02, 1.06), fontsize=9, color="#d62728")
    ax2.annotate(r"$t=1$: $\bm F_2$", (1, 1), xytext=(0.80, 1.06), fontsize=9, color="#1f5fa8")
    ax2.annotate(r"$t=0.5$: punto medio," "\n" r"$\det\bm F_m=1.5625>1$", (0.5, 1.5625),
                 xytext=(0.16, 1.62), fontsize=9, color="#2ca02c")
    ax2.set_xlabel(r"$t$ (a lo largo del segmento $\bm F(t)=(1-t)\bm F_1+t\bm F_2$)")
    ax2.set_ylabel(r"valor")
    ax2.set_title(r"Violaci\'on concreta: el camino f\'isico queda \emph{por encima} de la cuerda")
    ax2.legend(loc="lower center", fontsize=8.5, framealpha=0.9)
    ax2.set_ylim(0.85, 1.75)

    fig.tight_layout()
    fig.savefig(HERE / "ch3_lifting_claude.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote ch3_lifting_claude.png")


if __name__ == "__main__":
    main()
