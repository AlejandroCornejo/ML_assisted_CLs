#!/usr/bin/env python3
"""Chapter 3 motivation figure (Super Appendix B): three toy 1D
functions that ground the "why do we even want convexity" discussion:
f(x)=x^2 (the clean, single-minimum case we'd love to have), g(x)=x^4-3x^2
(two genuine minima plus a misleading local max -- an optimizer can get
stuck on the "wrong side"), and h(x)=x^3, the reader's own example
(no global minimum at all: it is neither convex nor concave, and is
unbounded below as x -> -infinity).
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


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), gridspec_kw={"wspace": 0.32})

    # Panel A: f(x) = x^2
    x = np.linspace(-2.2, 2.2, 400)
    axes[0].plot(x, x**2, color="#1f5fa8", linewidth=2.2)
    axes[0].scatter([0], [0], color="#1f5fa8", zorder=5, s=45)
    axes[0].annotate(r"\'unico m\'inimo", (0, 0), xytext=(0.15, 1.3), fontsize=10, color="#1f5fa8")
    axes[0].set_title(r"$f(x)=x^2$" "\n" r"convexa: un m\'inimo," "\n" r"sin ambig\"uedad", fontsize=9.8)

    # Panel B: g(x) = x^4 - 3x^2
    x2 = np.linspace(-2.2, 2.2, 400)
    g = x2**4 - 3 * x2**2
    axes[1].plot(x2, g, color="#d62728", linewidth=2.2)
    mins_x = np.sqrt(1.5)
    g_val = mins_x**4 - 3 * mins_x**2
    axes[1].scatter([-mins_x, mins_x], [g_val, g_val], color="#d62728", zorder=5, s=45)
    axes[1].scatter([0], [0], facecolors="white", edgecolors="#d62728", zorder=5, s=45)
    axes[1].annotate(r"m\'inimo real", (mins_x, g_val), xytext=(1.35, 4.2),
                      fontsize=9.5, color="#d62728",
                      arrowprops=dict(arrowstyle="-", color="#d62728", lw=0.8))
    axes[1].annotate(r"m\'inimo real", (-mins_x, g_val), xytext=(-2.15, 4.2),
                      fontsize=9.5, color="#d62728",
                      arrowprops=dict(arrowstyle="-", color="#d62728", lw=0.8))
    axes[1].annotate(r"m\'aximo local" "\n" r"(falsa trampa)", (0, 0), xytext=(0.25, 6.2),
                      fontsize=9.5, color="#d62728",
                      arrowprops=dict(arrowstyle="-", color="#d62728", lw=0.8))
    axes[1].set_ylim(-3.2, 9.2)
    axes[1].set_title(r"$g(x)=x^4-3x^2$" "\n" r"no convexa: dos m\'inimos," "\n" r"un optimizador se puede quedar en cualquiera", fontsize=9.8)

    # Panel C: h(x) = x^3 (the reader's own example)
    x3 = np.linspace(-2.0, 2.0, 400)
    axes[2].plot(x3, x3**3, color="#2ca02c", linewidth=2.2)
    axes[2].scatter([0], [0], facecolors="white", edgecolors="#2ca02c", zorder=5, s=45)
    axes[2].annotate(r"$x=0$: ni m\'aximo ni m\'inimo" "\n" r"(punto de inflexi\'on)",
                      (0, 0), xytext=(-2.0, 6.3), fontsize=9.2, color="#2ca02c",
                      arrowprops=dict(arrowstyle="-", color="#2ca02c", lw=0.8))
    axes[2].annotate(r"$h(x)\to-\infty$ sin cota" "\n" r"conforme $x\to-\infty$",
                      (-1.85, (-1.85) ** 3), xytext=(-1.0, -8.3), fontsize=9.2, color="#2ca02c",
                      arrowprops=dict(arrowstyle="-", color="#2ca02c", lw=0.8))
    axes[2].set_ylim(-9.5, 9.5)
    axes[2].set_title(r"$h(x)=x^3$ (tu ejemplo)" "\n" r"ni convexa ni c\'oncava:" "\n" r"no tiene ning\'un m\'inimo global", fontsize=9.8)

    for ax in axes:
        ax.axhline(0, color="#dddddd", linewidth=0.6, zorder=0)
        ax.axvline(0, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_xlabel(r"$x$")

    fig.tight_layout()
    fig.savefig(HERE / "ch3_motivation_claude.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote ch3_motivation_claude.png")


if __name__ == "__main__":
    main()
