#!/usr/bin/env python3
"""Stage--10 comparison figure for the final (comp-stress-weight=0) polyconvex PANN.

Five trajectory panels (direct FOM vs. the polyconvex PANN: energy, S11, S22,
S12, and the plane-stress von Mises equivalent) plus a sixth panel comparing
this PANN's aggregate relative-L2 errors against the officially recorded
direct HPROM-ANN reconstruction on the same Stage-10 path.
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
    with np.load(PREDICTIONS) as data:
        direct_energy = np.asarray(data["direct_energy"], dtype=float)
        direct_stress = np.asarray(data["direct_stress"], dtype=float)
        direct_von_mises = np.asarray(data["direct_von_mises"], dtype=float)
        predicted_energy = np.asarray(data["predicted_energy"], dtype=float)
        predicted_stress = np.asarray(data["predicted_stress"], dtype=float)
        predicted_von_mises = np.asarray(data["predicted_von_mises"], dtype=float)
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    hprom = metrics["direct_hprom_ann_reference_same_path"]

    x = np.arange(len(direct_energy))
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.8))

    panels = (
        (axes[0, 0], direct_energy, predicted_energy, r"$W$ [GPa]"),
        (axes[0, 1], direct_stress[:, 0], predicted_stress[:, 0], r"$S_{11}$ [GPa]"),
        (axes[0, 2], direct_stress[:, 1], predicted_stress[:, 1], r"$S_{22}$ [GPa]"),
        (axes[1, 0], direct_stress[:, 2], predicted_stress[:, 2], r"$S_{12}$ [GPa]"),
        (axes[1, 1], direct_von_mises, predicted_von_mises, r"$\sigma_{\rm eq}$ (von Mises) [GPa]"),
    )
    for axis, reference, prediction, ylabel in panels:
        axis.plot(x, reference / 1.0e9, color="black", linewidth=1.5, label="Direct FOM")
        axis.plot(x, prediction / 1.0e9, color="#d62728", linewidth=1.1, label="Polyconvex PANN")
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Stage--10 step index")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(loc="best", fontsize=8)

    bar_axis = axes[1, 2]
    categories = ("$W$", "$S$", "$S_{11}$", "$S_{22}$", "$S_{12}$", r"$\sigma_{\rm eq}$")
    pann_values = [
        metrics["energy_relative_l2"],
        metrics["stress_relative_l2"],
        *metrics["stress_component_relative_l2"],
        metrics["von_mises_relative_l2"],
    ]
    hprom_values = [
        hprom["energy_relative_l2"],
        hprom["stress_relative_l2"],
        *hprom["stress_component_relative_l2"],
        None,  # von Mises not available for the recorded direct HPROM-ANN reference
    ]
    positions = np.arange(len(categories))
    width = 0.36
    bar_axis.bar(positions - width / 2, [100.0 * v for v in pann_values], width, color="#d62728", label="Polyconvex PANN")
    hprom_positions = [p for p, v in zip(positions, hprom_values) if v is not None]
    hprom_plot_values = [100.0 * v for v in hprom_values if v is not None]
    bar_axis.bar(
        [p + width / 2 for p in hprom_positions], hprom_plot_values, width, color="#7f7f7f", label="Direct HPROM--ANN"
    )
    bar_axis.set_yscale("log")
    bar_axis.set_xticks(positions)
    bar_axis.set_xticklabels(categories)
    bar_axis.set_ylabel("Relative $L^2$ error [\\%]")
    bar_axis.set_title("Stage--10 accuracy vs. direct HPROM--ANN")
    bar_axis.grid(alpha=0.25, axis="y", which="both")
    bar_axis.legend(loc="best", fontsize=8)
    bar_axis.annotate(
        "N/A\n(energy only)",
        xy=(positions[-1] + width / 2, 1.0),
        xytext=(positions[-1] + width / 2, 1.0),
        ha="center", va="bottom", fontsize=6.5, color="#7f7f7f",
    )

    fig.tight_layout()
    fig.savefig(HERE / "polyconvex_stage10_comparison_claude.pdf", bbox_inches="tight")
    plt.close(fig)

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
