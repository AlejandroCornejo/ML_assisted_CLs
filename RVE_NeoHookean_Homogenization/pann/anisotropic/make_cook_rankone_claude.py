#!/usr/bin/env python3
"""Cook's membrane: rank-one convexity falsification, sampled at the
real Gauss-point deformation states this structural problem actually
visits (not a synthetic global sample), certified/ICKAN (converged,
full load) vs. free (stalled, 5% of that load). Real data from
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

COLORS = {"certified": "#1f5fa8", "ickan": "#2ca02c", "free": "#d62728"}
LABELS = {
    "certified": "Certified (ICNN), converged, full load",
    "ickan": "Certified (ICKAN), converged, full load",
    "free": "Free (uncertified), stalled, 5\\% of that load",
}


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), sharey=True)
    for ax, which in zip(axes, ("certified", "ickan", "free")):
        d = np.load(COOK_DIR / f"cook_results_{which}_claude.npz")
        samples = d["rank_one_samples"]
        n_neg = int(np.sum(samples < 0.0))
        frac = n_neg / samples.size
        # symlog-style: split sign, log-scale magnitude, to show both the
        # bulk (near zero) and the tails on one readable axis
        sign = np.sign(samples)
        mag = np.log10(np.abs(samples) + 1.0)
        signed_log = sign * mag
        ax.hist(signed_log, bins=60, color=COLORS[which], alpha=0.85)
        ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax.set_title(LABELS[which], fontsize=9.8)
        ax.set_xlabel(r"$\mathrm{sign}(d^2)\log_{10}(1+|d^2|)$")
        ax.text(0.03, 0.92, rf"{n_neg}/{samples.size} negative" "\n" rf"({100 * frac:.1f}\%)",
                transform=ax.transAxes, fontsize=9.5, va="top",
                color="#8b0000" if n_neg > 0 else "#006400",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#999999"))
    axes[0].set_ylabel("count")
    fig.suptitle(r"Rank-one convexity check at real structural Gauss-point states: "
                 r"$d^2W(\bm F+t\,\bm a\otimes\bm b)/dt^2|_{t=0}$",
                 fontsize=12.0)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(HERE / "cook_rankone_claude.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote cook_rankone_claude.pdf")


if __name__ == "__main__":
    main()
