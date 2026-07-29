#!/usr/bin/env python3
"""Figures for the clean direct-ICKAN and polyconvex-ICKAN comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pann_d4_model import load_selected_pann


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> None:
    with np.load(RESULTS / "ICKAN_D4_minor_features_stage10_predictions.npz") as data:
        strain = data["strain"]
        fom_energy, fom_stress = data["energy_reference"], data["stress_reference"]
        minor_energy, minor_stress = data["energy_prediction"], data["stress_prediction"]
    with np.load(RESULTS / "ICKAN_D4_direct_stage10_predictions.npz") as data:
        direct_energy, direct_stress = data["energy_prediction"], data["stress_prediction"]
    with np.load(RESULTS / "PANN_D4_polyconvex_stage10_predictions.npz") as data:
        pc_energy, pc_stress = data["polyconvex_energy"], data["polyconvex_stress"]
    free_model, free_scale, free_energy_scale, _ = load_selected_pann(
        HERE / "checkpoints" / "PANN_D4_best.pt", torch.device("cpu")
    )
    free_energy_parts, free_stress_parts = [], []
    with torch.enable_grad():
        for start in range(0, len(strain), 512):
            x = torch.as_tensor(strain[start : start + 512], dtype=torch.float32) / free_scale
            x = x.detach().clone().requires_grad_(True)
            energy, stress = free_model.energy_and_stress(x, create_graph=False)
            free_energy_parts.append((energy.detach().numpy() * free_energy_scale).reshape(-1))
            free_stress_parts.append(stress.detach().numpy() * (free_energy_scale / free_scale))
    free_energy, free_stress = np.concatenate(free_energy_parts), np.concatenate(free_stress_parts)

    x = np.arange(len(strain))
    labels = (r"$W$ [GPa]", r"$S_{xx}$ [GPa]", r"$S_{yy}$ [GPa]", r"$S_{xy}$ [GPa]")
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.7), sharex=True)
    sequences = ((fom_energy, free_energy, pc_energy, minor_energy, direct_energy),) + tuple(
        (fom_stress[:, c], free_stress[:, c], pc_stress[:, c], minor_stress[:, c], direct_stress[:, c])
        for c in range(3)
    )
    for axis, values, ylabel in zip(axes.ravel(), sequences, labels):
        axis.plot(x, values[0] / 1.0e9, color="black", linewidth=1.5, label="FOM directo")
        axis.plot(x, values[1] / 1.0e9, color="#1f77b4", linewidth=1.0, label="PANN-D4 libre")
        axis.plot(x, values[2] / 1.0e9, color="#d62728", linewidth=0.95, label="PConv-ICNN")
        axis.plot(x, values[3] / 1.0e9, color="#2ca02c", linewidth=1.05, label="PConv-ICKAN")
        axis.plot(x, values[4] / 1.0e9, color="#9467bd", linewidth=0.85, linestyle="--", label="ICKAN directo")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    for axis in axes[-1, :]:
        axis.set_xlabel("índice del paso Stage--10")
    axes[0, 0].legend(loc="best", fontsize=7.1)
    fig.tight_layout()
    fig.savefig(HERE / "ickan_d4_stage10_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    audit = json.loads((RESULTS / "ICKAN_D4_minor_features_stage10_metrics.json").read_text(encoding="utf-8"))["polyconvex_audit"]
    compression, expansion = audit["barrier_J_to_zero"], audit["growth_J_to_infinity"]
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.7))
    axes[0].plot(compression["J"], np.asarray(compression["energy_pa"]) / 1.0e9, "o-", color="#2ca02c")
    axes[0].set_xscale("log")
    axes[0].invert_xaxis()
    axes[0].set_xlabel(r"$J=\det F$ (compresión)")
    axes[0].set_ylabel("W PConv-ICKAN [GPa]")
    axes[0].set_title(r"Barrera: $J\rightarrow0^+$")
    axes[0].grid(alpha=0.25)
    axes[1].plot(expansion["J"], np.asarray(expansion["energy_pa"]) / 1.0e9, "o-", color="#2ca02c")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$J=\det F$ (expansión)")
    axes[1].set_ylabel("W PConv-ICKAN [GPa]")
    axes[1].set_title(r"Crecimiento: $J\rightarrow\infty$")
    axes[1].grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(HERE / "ickan_d4_barrier.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
