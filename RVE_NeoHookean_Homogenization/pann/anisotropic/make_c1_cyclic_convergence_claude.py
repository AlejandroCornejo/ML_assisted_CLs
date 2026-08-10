#!/usr/bin/env python3
"""Convergence of the cyclic-work residual on the mixed-triangle loop as
the trapezoidal quadrature is refined (200, 800, 3200 points per edge),
read directly from results/c1_cyclic_claude_metrics.json (the same
numbers reported in the text, not recomputed or re-derived). Log-log
axes: the three energy-based tiers should fall on (approximately) a
slope-(-2) line, the signature of O(h^2) trapezoidal quadrature error
converging to the true value of exactly zero; the tier-1 baseline has no
potential anywhere, so its residual is not quadrature error and should
stay flat instead.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

TIER_LABELS = {
    "regression_baseline_tier1": "Pure regression (tier 1)",
    "free_hyperelastic_tier2": "Free hyperelastic (tier 2)",
    "polyconvex_icnn_tier3a": "Polyconvex ICNN (tier 3a)",
    "polyconvex_ickan_tier3b": "Polyconvex ICKAN (tier 3b)",
}
TIER_COLORS = {
    "regression_baseline_tier1": "#9467bd",
    "free_hyperelastic_tier2": "#7f7f7f",
    "polyconvex_icnn_tier3a": "#d62728",
    "polyconvex_ickan_tier3b": "#1f77b4",
}
TIER_MARKERS = {
    "regression_baseline_tier1": "o",
    "free_hyperelastic_tier2": "s",
    "polyconvex_icnn_tier3a": "^",
    "polyconvex_ickan_tier3b": "v",
}


def main() -> None:
    metrics = json.loads((RESULTS / "c1_cyclic_claude_metrics.json").read_text(encoding="utf-8"))
    refinement = metrics["discretization_refinement_check"]
    levels = np.asarray(refinement["points_per_edge_levels"], dtype=float)
    per_tier = refinement["cyclic_work_per_model_per_level"]

    fig, ax = plt.subplots(figsize=(6.4, 5.0))

    for key, label in TIER_LABELS.items():
        values = np.abs(np.asarray(per_tier[key], dtype=float))
        ax.plot(levels, values, color=TIER_COLORS[key], marker=TIER_MARKERS[key],
                markersize=7, linewidth=1.8, label=label)

    # Reference slope for O(h^2) trapezoidal convergence (16x drop per 4x
    # refinement), anchored near the free-hyperelastic tier's first point.
    anchor_x, anchor_y = levels[0], abs(per_tier["free_hyperelastic_tier2"][0])
    ref_x = levels
    ref_y = anchor_y * (anchor_x / ref_x) ** 2
    ax.plot(ref_x, ref_y, color="black", linestyle=":", linewidth=1.3,
             label="reference slope (quadrature error ~ h^2)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Points per edge")
    ax.set_ylabel("Cyclic-work residual, |cyclic work| (Pa)")
    ax.set_title("Mixed-triangle loop: residual vs. quadrature refinement")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(True, which="both", linestyle="-", linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    fig.savefig(HERE / "c1_cyclic_convergence_claude.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote c1_cyclic_convergence_claude.pdf")


if __name__ == "__main__":
    main()
