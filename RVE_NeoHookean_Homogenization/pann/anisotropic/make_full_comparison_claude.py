#!/usr/bin/env python3
"""Six-way Stage--10 comparison: all four ANN tiers (regression, free
hyperelastic, polyconvex ICNN, polyconvex ICKAN) and both HPROM--ANN
operating modes (iterative and direct), matching Table~\\ref{tab:master}.

Produces one multi-panel trajectory figure (S11, S22, S12, and the von
Mises equivalent: direct FOM vs. all six variants; energy: direct FOM vs.
only the three energy-capable tiers, since the regression tier and both
HPROM--ANN modes are stress-only reduced-order/ANN maps with no scalar
potential) and one grouped bar chart of relative L2 errors, plus the
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
DATA = HERE.parent / "data"


def load(prefix: str):
    metrics = json.loads((RESULTS / f"{prefix}_metrics.json").read_text(encoding="utf-8"))
    predictions_path = RESULTS / f"{prefix}_predictions.npz"
    predictions = dict(np.load(predictions_path)) if predictions_path.exists() else None
    return metrics, predictions


def main() -> None:
    regression_metrics, regression_pred = load("regression_claude")
    free_metrics, free_pred = load("free_compw000_ellipticity_claude")
    icnn_metrics, icnn_pred = load("polyconvex_final_claude")
    ickan_metrics, ickan_pred = load("polyconvex_ickan_final_claude")

    hprom_direct = json.loads((DATA / "hprom_ann_direct_stage10_metrics.json").read_text(encoding="utf-8"))
    hprom_iter_metrics = hprom_direct["modes"]["hprom_ann_iterative"]
    dhprom_metrics = hprom_direct["modes"]["dhprom_ann_direct"]
    with np.load(DATA / "hprom_ann_direct_stage10_metrics.npz") as hprom_arrays:
        stage10_stress = np.asarray(hprom_arrays["stage10_stress"], dtype=float)
        hprom_iter_stress = np.asarray(hprom_arrays["hprom_ann_iterative_stress"], dtype=float)
        dhprom_stress = np.asarray(hprom_arrays["dhprom_ann_direct_stress"], dtype=float)

    def von_mises(stress: np.ndarray) -> np.ndarray:
        sxx, syy, sxy = stress[:, 0], stress[:, 1], stress[:, 2]
        return np.sqrt(np.maximum(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy, 0.0))

    fom_von_mises = von_mises(stage10_stress)
    hprom_iter_von_mises = von_mises(hprom_iter_stress)
    dhprom_von_mises = von_mises(dhprom_stress)

    stress_models = [
        ("Regression ANN (tier 1)", regression_pred["predicted_stress"], "#9467bd"),
        ("Free hyperelastic ANN (tier 2)", free_pred["predicted_stress"], "#7f7f7f"),
        ("Polyconvex ICNN (tier 3a)", icnn_pred["predicted_stress"], "#d62728"),
        ("Polyconvex ICKAN (tier 3b)", ickan_pred["predicted_stress"], "#1f77b4"),
        ("HPROM--ANN", hprom_iter_stress, "#ff7f0e"),
        ("D-HPROM--ANN", dhprom_stress, "#2ca02c"),
    ]
    von_mises_models = [
        ("Regression ANN (tier 1)", regression_pred["predicted_von_mises"], "#9467bd"),
        ("Free hyperelastic ANN (tier 2)", free_pred["predicted_von_mises"], "#7f7f7f"),
        ("Polyconvex ICNN (tier 3a)", icnn_pred["predicted_von_mises"], "#d62728"),
        ("Polyconvex ICKAN (tier 3b)", ickan_pred["predicted_von_mises"], "#1f77b4"),
        ("HPROM--ANN", hprom_iter_von_mises, "#ff7f0e"),
        ("D-HPROM--ANN", dhprom_von_mises, "#2ca02c"),
    ]
    energy_models = [
        ("Free hyperelastic ANN (tier 2)", free_pred["predicted_energy"], "#7f7f7f"),
        ("Polyconvex ICNN (tier 3a)", icnn_pred["predicted_energy"], "#d62728"),
        ("Polyconvex ICKAN (tier 3b)", ickan_pred["predicted_energy"], "#1f77b4"),
    ]

    # --- Trajectory figure: energy (3 series) + S11, S22, S12, von Mises (all 6) ---
    x = np.arange(len(fom_von_mises))
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.2))

    axis = axes[0, 0]
    axis.plot(x, icnn_pred["direct_energy"] / 1.0e9, color="black", linewidth=1.8, label="Direct FOM")
    for name, values, color in energy_models:
        axis.plot(x, np.asarray(values) / 1.0e9, color=color, linewidth=1.0, label=name)
    axis.set_ylabel(r"$W$ [GPa]")
    axis.set_xlabel("Held-out test-trajectory step index")
    axis.set_title("Energy (energy-capable tiers only)")
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=6.5)

    component_panels = (
        (axes[0, 1], 0, r"$S_{11}$ [GPa]"),
        (axes[0, 2], 1, r"$S_{22}$ [GPa]"),
        (axes[1, 0], 2, r"$S_{12}$ [GPa]"),
    )
    for axis, component_index, ylabel in component_panels:
        axis.plot(x, stage10_stress[:, component_index] / 1.0e9, color="black", linewidth=1.8, label="Direct FOM")
        for name, values, color in stress_models:
            axis.plot(x, np.asarray(values)[:, component_index] / 1.0e9, color=color, linewidth=1.0, label=name)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Held-out test-trajectory step index")
        axis.grid(alpha=0.25)

    axis = axes[1, 1]
    axis.plot(x, fom_von_mises / 1.0e9, color="black", linewidth=1.8, label="Direct FOM")
    for name, values, color in von_mises_models:
        axis.plot(x, np.asarray(values) / 1.0e9, color=color, linewidth=1.0, label=name)
    axis.set_ylabel(r"$\sigma_{\rm eq}$ (von Mises) [GPa]")
    axis.set_xlabel("Held-out test-trajectory step index")
    axis.grid(alpha=0.25)

    axes[1, 2].axis("off")
    handles, labels = axes[1, 1].get_legend_handles_labels()
    axes[1, 2].legend(handles, labels, loc="center", fontsize=9, title="All six model variants")

    fig.tight_layout()
    fig.savefig(HERE / "full_comparison_trajectories_claude.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- Bar chart: relative L2 errors across all six variants ---
    categories = ("$W$", "$S$", "$S_{11}$", "$S_{22}$", "$S_{12}$", r"$\sigma_{\rm eq}$")

    def row(metrics):
        return [
            metrics.get("energy_relative_l2"),
            metrics["stress_relative_l2"],
            *metrics["stress_component_relative_l2"],
            metrics["von_mises_relative_l2"],
        ]

    bars = [
        ("Regression (tier 1)", row(regression_metrics), "#9467bd"),
        ("Free hyperelastic (tier 2)", row(free_metrics), "#7f7f7f"),
        ("Polyconvex ICNN (tier 3a)", row(icnn_metrics), "#d62728"),
        ("Polyconvex ICKAN (tier 3b)", row(ickan_metrics), "#1f77b4"),
        ("HPROM--ANN", row(hprom_iter_metrics), "#ff7f0e"),
        ("D-HPROM--ANN", row(dhprom_metrics), "#2ca02c"),
    ]

    fig, axis = plt.subplots(figsize=(10.5, 4.8))
    positions = np.arange(len(categories))
    width = 0.13
    for i, (name, values, color) in enumerate(bars):
        offset = (i - (len(bars) - 1) / 2.0) * width
        plot_positions = [p + offset for p, v in zip(positions, values) if v is not None]
        plot_values = [100.0 * v for v in values if v is not None]
        axis.bar(plot_positions, plot_values, width, color=color, label=name)
    axis.set_yscale("log")
    axis.set_xticks(positions)
    axis.set_xticklabels(categories)
    axis.set_ylabel("Relative $L^2$ error [%]")
    axis.set_title("Held-out test-trajectory accuracy: all six model variants")
    axis.grid(alpha=0.25, axis="y", which="both")
    axis.legend(loc="best", fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(HERE / "full_comparison_bars_claude.pdf", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "regression": {k: regression_metrics[k] for k in ("stress_relative_l2", "von_mises_relative_l2")},
        "free_hyperelastic": {k: free_metrics[k] for k in ("energy_relative_l2", "stress_relative_l2", "von_mises_relative_l2")},
        "polyconvex_icnn": {k: icnn_metrics[k] for k in ("energy_relative_l2", "stress_relative_l2", "von_mises_relative_l2")},
        "polyconvex_ickan": {k: ickan_metrics[k] for k in ("energy_relative_l2", "stress_relative_l2", "von_mises_relative_l2")},
        "hprom_ann_iterative": {k: hprom_iter_metrics[k] for k in ("stress_relative_l2", "von_mises_relative_l2")},
        "dhprom_ann_direct": {k: dhprom_metrics[k] for k in ("stress_relative_l2", "von_mises_relative_l2")},
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
