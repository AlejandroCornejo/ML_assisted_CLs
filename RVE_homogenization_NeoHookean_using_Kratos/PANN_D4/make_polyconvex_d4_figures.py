#!/usr/bin/env python3
"""Create the two figures used by the polyconvex-PANN addendum."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pann_d4_model import load_selected_pann
from polyconvex_d4_model import load_polyconvex_pann


HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "alltraj_stage10_direct_energy.npz"


def predict(model, strain: np.ndarray, strain_scale: float, energy_scale: float, *, dtype: torch.dtype) -> tuple[np.ndarray, np.ndarray]:
    energies, stresses = [], []
    with torch.enable_grad():
        for start in range(0, len(strain), 512):
            raw = torch.as_tensor(strain[start : start + 512], dtype=dtype)
            x = (raw / strain_scale).detach().clone().requires_grad_(True)
            energy, stress = model.energy_and_stress(x, create_graph=False)
            energies.append((energy.detach().cpu().numpy() * energy_scale).reshape(-1))
            stresses.append(stress.detach().cpu().numpy() * (energy_scale / strain_scale))
    return np.concatenate(energies), np.concatenate(stresses)


def main() -> None:
    free_model, free_scale, free_energy_scale, _ = load_selected_pann(HERE / "checkpoints" / "PANN_D4_best.pt", torch.device("cpu"))
    pc_model, pc_scale, pc_energy_scale, _ = load_polyconvex_pann(HERE / "checkpoints" / "PANN_D4_polyconvex_best.pt", torch.device("cpu"))
    with np.load(DATA) as data:
        strain = np.asarray(data["stage10_strain"], dtype=np.float64)
        fom_energy = np.asarray(data["stage10_energy"], dtype=np.float64)
        fom_stress = np.asarray(data["stage10_stress"], dtype=np.float64)
    free_energy, free_stress = predict(free_model, strain, free_scale, free_energy_scale, dtype=torch.float32)
    pc_energy, pc_stress = predict(pc_model, strain, pc_scale, pc_energy_scale, dtype=torch.float64)

    x = np.arange(len(strain))
    labels = (r"$W$ [GPa]", r"$S_{xx}$ [GPa]", r"$S_{yy}$ [GPa]", r"$S_{xy}$ [GPa]")
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.6), sharex=True)
    sequences = ((fom_energy, free_energy, pc_energy),) + tuple(
        (fom_stress[:, component], free_stress[:, component], pc_stress[:, component]) for component in range(3)
    )
    for axis, values, ylabel in zip(axes.ravel(), sequences, labels):
        axis.plot(x, values[0] / 1.0e9, color="black", linewidth=1.5, label="FOM directo")
        axis.plot(x, values[1] / 1.0e9, color="#1f77b4", linewidth=1.0, label="PANN-D4 libre")
        axis.plot(x, values[2] / 1.0e9, color="#d62728", linewidth=1.0, label="PANN-D4 policonvexa")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    for axis in axes[-1, :]:
        axis.set_xlabel("índice del paso Stage--10")
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "polyconvex_d4_stage10_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    audit = json.loads((HERE / "results" / "PANN_D4_polyconvex_stage10_metrics.json").read_text(encoding="utf-8"))["analytic_and_numerical_guarantee_audit"]
    compression = audit["volumetric_barrier_J_to_zero"]
    expansion = audit["volumetric_growth_J_to_infinity"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    axes[0].plot(compression["J"], np.asarray(compression["energy"]) / 1.0e9, "o-", color="#d62728")
    axes[0].set_xscale("log")
    axes[0].invert_xaxis()
    axes[0].set_xlabel(r"$J=\det F$ (compresión)")
    axes[0].set_ylabel(r"$W_{\rm pc}$ [GPa]")
    axes[0].set_title(r"Barrera: $J\rightarrow0^+$")
    axes[0].grid(alpha=0.25)
    axes[1].plot(expansion["J"], np.asarray(expansion["energy"]) / 1.0e9, "o-", color="#1f77b4")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$J=\det F$ (expansión)")
    axes[1].set_ylabel(r"$W_{\rm pc}$ [GPa]")
    axes[1].set_title(r"Crecimiento: $J\rightarrow\infty$")
    axes[1].grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(HERE / "polyconvex_d4_barrier.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()

