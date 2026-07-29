#!/usr/bin/env python3
"""Four-way Stage--10 comparison: certified polyconvex ICNN, certified
polyconvex ICKAN, the non-polyconvex "free" baseline, and the direct
HPROM--ANN reconstruction (aggregate numbers only, no per-step arrays
available for that one -- see evaluate_polyconvex_final_claude.py).

Produces one trajectory figure (von Mises equivalent stress: FOM vs. the
three PANNs) and one grouped bar chart of relative L2 errors, plus the
rank-one curvature contrast (the certified models' minimum sampled
curvature vs. the free model's confirmed violation rate).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def load(prefix: str):
    metrics = json.loads((RESULTS / f"{prefix}_metrics.json").read_text(encoding="utf-8"))
    predictions_path = RESULTS / f"{prefix}_predictions.npz"
    predictions = dict(np.load(predictions_path)) if predictions_path.exists() else None
    return metrics, predictions


def main() -> None:
    icnn_metrics, icnn_pred = load("polyconvex_final_claude")
    ickan_metrics, ickan_pred = load("polyconvex_ickan_final_claude")
    free_metrics, free_pred = load("free_compw000_ellipticity_claude")

    hprom = icnn_metrics["direct_hprom_ann_reference_same_path"]

    models = [
        ("Polyconvex ICNN", icnn_metrics, icnn_pred, "#d62728"),
        ("Polyconvex ICKAN", ickan_metrics, ickan_pred, "#1f77b4"),
        ("Non-polyconvex free ANN", free_metrics, free_pred, "#7f7f7f"),
    ]

    # --- Trajectory figure: von Mises stress, FOM vs all three PANNs ---
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(icnn_pred["direct_von_mises"]))
    axis.plot(x, icnn_pred["direct_von_mises"] / 1.0e9, color="black", linewidth=1.6, label="Direct FOM")
    for name, _metrics, pred, color in models:
        axis.plot(x, pred["predicted_von_mises"] / 1.0e9, color=color, linewidth=1.0, label=name)
    axis.set_xlabel("Stage--10 step index")
    axis.set_ylabel(r"$\sigma_{\rm eq}$ (von Mises) [GPa]")
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "full_comparison_vonmises_claude.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- Bar chart: relative L2 errors across all four ---
    categories = ("$W$", "$S$", "$S_{11}$", "$S_{22}$", "$S_{12}$", r"$\sigma_{\rm eq}$")

    def row(metrics):
        return [
            metrics["energy_relative_l2"],
            metrics["stress_relative_l2"],
            *metrics["stress_component_relative_l2"],
            metrics["von_mises_relative_l2"],
        ]

    hprom_row = [hprom["energy_relative_l2"], hprom["stress_relative_l2"], *hprom["stress_component_relative_l2"], None]

    fig, axis = plt.subplots(figsize=(9.0, 4.6))
    positions = np.arange(len(categories))
    width = 0.2
    bars = [
        ("Polyconvex ICNN", row(icnn_metrics), "#d62728"),
        ("Polyconvex ICKAN", row(ickan_metrics), "#1f77b4"),
        ("Non-polyconvex free ANN", row(free_metrics), "#7f7f7f"),
        ("Direct HPROM--ANN", hprom_row, "#2ca02c"),
    ]
    for i, (name, values, color) in enumerate(bars):
        offset = (i - 1.5) * width
        plot_positions = [p + offset for p, v in zip(positions, values) if v is not None]
        plot_values = [100.0 * v for v in values if v is not None]
        axis.bar(plot_positions, plot_values, width, color=color, label=name)
    axis.set_yscale("log")
    axis.set_xticks(positions)
    axis.set_xticklabels(categories)
    axis.set_ylabel("Relative $L^2$ error [\\%]")
    axis.set_title("Stage--10 accuracy: certified models vs. non-polyconvex baseline vs. direct HPROM--ANN")
    axis.grid(alpha=0.25, axis="y", which="both")
    axis.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "full_comparison_bars_claude.pdf", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "polyconvex_icnn": {k: icnn_metrics[k] for k in ("energy_relative_l2", "stress_relative_l2", "von_mises_relative_l2")},
        "polyconvex_ickan": {k: ickan_metrics[k] for k in ("energy_relative_l2", "stress_relative_l2", "von_mises_relative_l2")},
        "non_polyconvex_free": {k: free_metrics[k] for k in ("energy_relative_l2", "stress_relative_l2", "von_mises_relative_l2")},
        "direct_hprom_ann": {k: hprom[k] for k in ("energy_relative_l2", "stress_relative_l2")},
        "rank_one_curvature_contrast": {
            "polyconvex_icnn_minimum_curvature_gpa": icnn_metrics["guarantee_audit"]["rank_one_curvature_audit"]["minimum_second_derivative"] / 1.0e9,
            "polyconvex_ickan_minimum_curvature_gpa": ickan_metrics["guarantee_audit"]["rank_one_curvature_audit"]["minimum_second_derivative"] / 1.0e9,
            "non_polyconvex_free_violation_fraction": free_metrics["rank_one_curvature_audit"]["fraction_violations"],
            "non_polyconvex_free_minimum_curvature_gpa": free_metrics["rank_one_curvature_audit"]["minimum_curvature"] / 1.0e9,
        },
    }
    (RESULTS / "full_comparison_summary_claude.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
