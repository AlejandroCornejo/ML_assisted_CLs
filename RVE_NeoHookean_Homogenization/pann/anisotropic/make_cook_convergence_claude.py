#!/usr/bin/env python3
"""Cook's membrane: Newton residual-norm convergence history at the
first load step, all four PANN tiers, same tolerance and line search.
Polyconvex ICNN and ICKAN (tiers 3a/3b) are nearly coincident and are
offset slightly and drawn with distinct markers/linestyles so neither
is hidden behind the other. Colors and tier order match every other
tier comparison in the paper (make_full_comparison_claude.py). Real
data from Cook.gid/cook_results_*_claude.npz.
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
COOK_DIR = HERE.parent.parent / "Cook.gid"

TIER_ORDER = ("regression", "free", "certified", "ickan")
COLORS = {"regression": "#9467bd", "free": "#7f7f7f", "certified": "#d62728", "ickan": "#1f77b4"}
LABELS = {"regression": "Regression (tier 1)", "free": "Free hyperelastic (tier 2)",
          "certified": "Polyconvex ICNN (tier 3a)", "ickan": "Polyconvex ICKAN (tier 3b)"}
MARKERS = {"regression": "^", "free": "X", "certified": "o", "ickan": "s"}
LINESTYLES = {"regression": "-.", "free": "-", "certified": "-", "ickan": "--"}
X_JITTER = {"regression": 0.0, "free": 0.0, "certified": -0.05, "ickan": 0.05}


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for which in TIER_ORDER:
        d = np.load(COOK_DIR / f"cook_results_{which}_claude.npz")
        res = d["residual_history_step1"]
        iters = np.arange(1, len(res) + 1) + X_JITTER[which]
        ax.semilogy(iters, res / res[0], MARKERS[which] + LINESTYLES[which], color=COLORS[which],
                    linewidth=2.0, markersize=6, markeredgecolor="black", markeredgewidth=0.4,
                    alpha=1.0, zorder=3, label=LABELS[which])

    ax.axhline(1.0e-4, color="#999999", linewidth=0.9, linestyle=":", label="convergence tolerance")
    ax.set_xlabel("Newton iteration (first load step, with line search)")
    ax.set_ylabel(r"$\|\bm r\|/\|\bm r_0\|$")
    ax.set_title("Cook's membrane: Newton convergence history at the\nfirst load step, same tolerance and line search")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(HERE / "cook_convergence_claude.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote cook_convergence_claude.pdf")


if __name__ == "__main__":
    main()
