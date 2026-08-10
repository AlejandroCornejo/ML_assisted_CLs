#!/usr/bin/env python3
"""Chapter 9 (Super Appendix B): why "convex AND non-decreasing" is the
exact condition needed for h(g(x)) to be convex when g is convex.
Three panels: g(x)=x^2 (convex); h1(y)=e^y (convex, increasing) composed
with g, still convex; h2(y)=e^{-y} (convex, but DECREASING) composed
with g, a bump shape that is not convex at all -- the monotonicity
condition is not a technicality, dropping it breaks the rule outright.
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

x = np.linspace(-2.0, 2.0, 400)
g = x ** 2
h1_g = np.exp(g)
h2_g = np.exp(-g)


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6))

    axes[0].plot(x, g, color="#1f5fa8", linewidth=2.2)
    axes[0].set_title(r"$g(x)=x^2$" "\n" r"(convexa)", fontsize=10.5)

    axes[1].plot(x, h1_g, color="#2ca02c", linewidth=2.2)
    axes[1].set_title(r"$h_1(y)=e^{y}$,\ $h_1(g(x))=e^{x^2}$" "\n"
                       r"convexa \textbf{y no-decreciente}:" "\n" r"la composici\'on es convexa",
                       fontsize=10.5)

    axes[2].plot(x, h2_g, color="#d62728", linewidth=2.2)
    axes[2].set_title(r"$h_2(y)=e^{-y}$,\ $h_2(g(x))=e^{-x^2}$" "\n"
                       r"convexa pero \textbf{decreciente}:" "\n" r"la composici\'on NO es convexa",
                       fontsize=10.5)

    for ax in axes:
        ax.axvline(0, color="#dddddd", linewidth=0.8, zorder=0)
        ax.set_xlabel(r"$x$")
    fig.suptitle(r"Por qu\'e la regla de composici\'on necesita ``convexa \textbf{y} no-decreciente'', no solo ``convexa''",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=2.2)
    fig.savefig(HERE / "ch9_composition_claude.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote ch9_composition_claude.png")


if __name__ == "__main__":
    main()
