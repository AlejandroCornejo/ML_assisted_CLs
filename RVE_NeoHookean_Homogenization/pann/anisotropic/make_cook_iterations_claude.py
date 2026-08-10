#!/usr/bin/env python3
"""Cook's membrane: Newton iterations needed per load step across the
full ramp, all four PANN tiers. Polyconvex ICNN and ICKAN (tiers 3a/3b)
coincide at every step (both flat at 3 iterations) and are plotted with
a small horizontal offset, distinct markers, and distinct linestyles so
neither is hidden behind the other; the two uncertified tiers both stall
already at the first step and are offset the same way. Colors and tier
order match every other tier comparison in the paper
(make_full_comparison_claude.py). Real data from
Cook.gid/cook_results_*_claude.npz.
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
X_JITTER = {"regression": 0.18, "free": -0.18, "certified": -0.15, "ickan": 0.15}


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for which in TIER_ORDER:
        d = np.load(COOK_DIR / f"cook_results_{which}_claude.npz")
        load = d["load_per_step"]
        iters = d["iters_per_step"]
        converged = d["converged_per_step"]
        steps = np.arange(1, len(load) + 1) + X_JITTER[which]
        ax.plot(steps, iters, MARKERS[which] + LINESTYLES[which], color=COLORS[which], linewidth=2.0,
                markersize=7, markeredgecolor="black", markeredgewidth=0.4, alpha=1.0, zorder=3,
                label=LABELS[which])
        if not bool(converged[-1]):
            ax.plot(steps[-1], iters[-1], "x", color=COLORS[which], markersize=13, markeredgewidth=3,
                    zorder=4)

    ax.annotate("did not converge\n(30-iteration cap)", xy=(0.82, 30), xytext=(4.0, 24),
                fontsize=9, color="black",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax.set_xlabel("Load step (1--20, ramped to the full target load)")
    ax.set_ylabel("Newton iterations to converge")
    ax.set_xticks(np.arange(1, 21, 2))
    ax.set_ylim(0, 33)
    ax.set_title("Cook's membrane: iterations needed across the load ramp")
    ax.legend(loc="center right", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(HERE / "cook_iterations_claude.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote cook_iterations_claude.pdf")


if __name__ == "__main__":
    main()
