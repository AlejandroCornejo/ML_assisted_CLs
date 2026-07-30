#!/usr/bin/env python3
"""Stage--10 diagnostic figures for the final (comp-stress-weight=0) polyconvex PANN.

Produces the ICNN-specific volumetric-barrier certificate figure. The
six-way Stage--10 trajectory comparison across all model variants lives in
make_full_comparison_claude.py (fig:full-comparison in the memo); this
script no longer duplicates it.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PREDICTIONS = RESULTS / "polyconvex_final_claude_predictions.npz"
METRICS = RESULTS / "polyconvex_final_claude_metrics.json"


def main() -> None:
    if not PREDICTIONS.exists() or not METRICS.exists():
        raise FileNotFoundError("Run evaluate_polyconvex_final_claude.py before creating figures.")
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))

    audit = metrics["guarantee_audit"]
    compression = audit["volumetric_barrier_J_to_zero"]
    expansion = audit["volumetric_growth_J_to_infinity"]
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.6))
    axes[0].plot(compression["J"], np.asarray(compression["energy"]) / 1.0e9, "o-", color="#d62728")
    axes[0].set_xscale("log")
    axes[0].invert_xaxis()
    axes[0].set_xlabel(r"$J=\det F$ (compression)")
    axes[0].set_ylabel(r"$W_{\rm pc}$ [GPa]")
    axes[0].set_title(r"Barrier as $J\to0^+$")
    axes[0].grid(alpha=0.25)
    axes[1].plot(expansion["J"], np.asarray(expansion["energy"]) / 1.0e9, "o-", color="#1f77b4")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$J=\det F$ (expansion)")
    axes[1].set_ylabel(r"$W_{\rm pc}$ [GPa]")
    axes[1].set_title(r"Growth as $J\to\infty$")
    axes[1].grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(HERE / "polyconvex_volumetric_claude.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
