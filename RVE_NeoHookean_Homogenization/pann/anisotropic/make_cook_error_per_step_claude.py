#!/usr/bin/env python3
"""Regenerate cook_error_per_step_claude.png with the paper's canonical
model order (Linear-HPROM, HPROM--ANN, D-HPROM--ANN) in the legend,
matching the reordered prose/table row order elsewhere this session.
Same data, same visual encoding (color/marker/linestyle per model) as
the original figure -- only the plotting/legend order changes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FE2_DIR = HERE.parent.parent / "fe2_extension"

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}",
})


def midpoint(d):
    return 0.5 * (d["tip_uy_min_per_step"] + d["tip_uy_max_per_step"])


def main() -> None:
    fom = np.load(FE2_DIR / "cook_results_fom_nested_consistent_parallel_claude.npz", allow_pickle=True)
    linear = np.load(FE2_DIR / "cook_results_linear_hprom_parallel_continuation_claude.npz", allow_pickle=True)
    hprom = np.load(FE2_DIR / "cook_results_hprom_ann_parallel_continuation_claude.npz", allow_pickle=True)
    dhprom = np.load(FE2_DIR / "cook_results_dhprom_ann_parallel_claude.npz", allow_pickle=True)

    steps = np.arange(1, 21)
    mid_fom = midpoint(fom)

    series = [
        ("Linear-HPROM-FE$^2$", linear, "tab:green", "o", "-"),
        ("HPROM--ANN-FE$^2$ (20pt)", hprom, "tab:blue", "^", ":"),
        ("D-HPROM--ANN-FE$^2$ (10pt)", dhprom, "tab:red", "s", "--"),
    ]

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for label, data, color, marker, ls in series:
        mid = midpoint(data)
        rel_err = 100.0 * np.abs(mid - mid_fom) / np.abs(mid_fom)
        ax.semilogy(steps, rel_err, color=color, marker=marker, linestyle=ls,
                    label=label, markersize=5, linewidth=1.4)

    ax.set_xlabel(r"Cook load step")
    ax.set_ylabel(r"Tip $u_y$ relative error vs.\ FOM-FE$^2$ [\%]")
    ax.set_xlim(0.5, 20.5)
    ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.6)
    ax.legend(loc="lower left", frameon=True, fontsize=9.5)

    fig.tight_layout()
    out_path = HERE / "cook_error_per_step_claude.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
