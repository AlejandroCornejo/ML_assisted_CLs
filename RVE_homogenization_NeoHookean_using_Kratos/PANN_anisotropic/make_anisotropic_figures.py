#!/usr/bin/env python3
"""Create Stage--10 and volumetric figures after the one declared evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PREDICTIONS = RESULTS / "anisotropic_stage10_predictions.npz"
METRICS = RESULTS / "anisotropic_stage10_metrics.json"


def main() -> None:
    if not PREDICTIONS.exists() or not METRICS.exists():
        raise FileNotFoundError("Run evaluate_anisotropic_pann.py before creating figures.")
    with np.load(PREDICTIONS) as data:
        direct_energy = np.asarray(data["direct_energy"], dtype=float)
        direct_stress = np.asarray(data["direct_stress"], dtype=float)
        free_energy = np.asarray(data["free_energy"], dtype=float)
        free_stress = np.asarray(data["free_stress"], dtype=float)
        metric_energy = np.asarray(data["metric_energy"], dtype=float)
        metric_stress = np.asarray(data["metric_stress"], dtype=float)
        poly_energy = np.asarray(data["polyconvex_energy"], dtype=float)
        poly_stress = np.asarray(data["polyconvex_stress"], dtype=float)
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    x = np.arange(len(direct_energy))
    labels = (r"$W$ [GPa]", r"$S_{11}$ [GPa]", r"$S_{22}$ [GPa]", r"$S_{12}$ [GPa]")
    sequences = ((direct_energy, free_energy, metric_energy, poly_energy),) + tuple(
        (direct_stress[:, component], free_stress[:, component], metric_stress[:, component], poly_stress[:, component])
        for component in range(3)
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.7), sharex=True)
    for axis, values, ylabel in zip(axes.ravel(), sequences, labels):
        axis.plot(x, values[0] / 1.0e9, color="black", linewidth=1.4, label="Direct FOM")
        axis.plot(x, values[1] / 1.0e9, color="#1f77b4", linewidth=1.0, label="Free anisotropic PANN")
        axis.plot(x, values[2] / 1.0e9, color="#2ca02c", linewidth=1.0, label="PANN-T (local metric)")
        axis.plot(x, values[3] / 1.0e9, color="#d62728", linewidth=1.0, label="Polyconvex PANN")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[1, 0].set_xlabel("Stage--10 step index")
    axes[1, 1].set_xlabel("Stage--10 step index")
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "anisotropic_stage10_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    audit = metrics["polyconvex_anisotropic_pann"]["guarantee_audit"]
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
    axes[1].set_xlabel(r"$J=\det F$ (expansión)")
    axes[1].set_ylabel(r"$W_{\rm pc}$ [GPa]")
    axes[1].set_title(r"Growth as $J\to\infty$")
    axes[1].grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(HERE / "anisotropic_polyconvex_volumetric.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
