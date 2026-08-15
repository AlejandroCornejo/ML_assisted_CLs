#!/usr/bin/env python3
"""Exploratory, non-paper check: evaluate the new tangent0-loss variants of
certified ICNN and ICKAN against the SAME held-out stage-10 trajectory used
for the paper's own Table 6, side by side with the original checkpoints.
Same relative-L2 metric evaluate_anisotropic_pann.py uses. Also reports
the error restricted to the trajectory's own smallest-strain subset, since
that is the regime the new loss term targets.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / ".." / ".." / "Cook.gid"))

DATA_PATH = HERE.parent / "data" / "alltraj_stage10_direct_energy.npz"

from anisotropic_pann_model import load_anisotropic_polyconvex  # noqa: E402
from anisotropic_pann_model_ickan_claude import load_anisotropic_polyconvex_ickan  # noqa: E402


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - reference) / max(np.linalg.norm(reference), 1.0e-30))


def predict(model, strain: np.ndarray, *, strain_scale: float, energy_scale: float, dtype, batch_size=512):
    energies, stresses = [], []
    with torch.enable_grad():
        for start in range(0, len(strain), batch_size):
            stop = min(start + batch_size, len(strain))
            x = torch.as_tensor(strain[start:stop] / strain_scale, dtype=dtype)
            energy, stress = model.energy_and_stress(x.requires_grad_(True), create_graph=False)
            energies.append((energy.detach().numpy() * energy_scale).reshape(-1))
            stresses.append(stress.detach().numpy() * (energy_scale / strain_scale))
    return np.concatenate(energies), np.concatenate(stresses, axis=0)


def evaluate(loader, checkpoint_name, strain, energy_true, stress_true, small_idx):
    model, strain_scale, energy_scale, _ = loader(HERE / "checkpoints" / checkpoint_name, torch.device("cpu"))
    model.eval()
    dtype = next(model.parameters()).dtype
    energy_pred, stress_pred = predict(model, strain, strain_scale=strain_scale, energy_scale=energy_scale, dtype=dtype)
    full_e = relative_l2(energy_pred, energy_true)
    full_s = relative_l2(stress_pred, stress_true)
    small_e = relative_l2(energy_pred[small_idx], energy_true[small_idx])
    small_s = relative_l2(stress_pred[small_idx], stress_true[small_idx])
    return full_e, full_s, small_e, small_s


if __name__ == "__main__":
    with np.load(DATA_PATH) as data:
        strain = np.asarray(data["stage10_strain"], dtype=np.float64)
        energy_true = np.asarray(data["stage10_energy"], dtype=np.float64)
        stress_true = np.asarray(data["stage10_stress"], dtype=np.float64)

    mags = np.abs(strain).max(axis=1)
    small_idx = np.flatnonzero((mags > 0) & (mags <= 0.02))
    print(f"n_total={len(strain)}, n_small(|E|max in (0,0.02])={len(small_idx)}")

    rows = [
        ("certified (original)", load_anisotropic_polyconvex, "PANN_anisotropic_polyconvex_final_claude.pt"),
        ("certified (smallstrain)", load_anisotropic_polyconvex, "PANN_anisotropic_polyconvex_smallstrain_claude.pt"),
        ("ickan (original)", load_anisotropic_polyconvex_ickan, "PANN_anisotropic_polyconvex_ickan_final_claude.pt"),
        ("ickan (smallstrain)", load_anisotropic_polyconvex_ickan, "PANN_anisotropic_polyconvex_ickan_smallstrain_claude.pt"),
    ]
    print(f"{'model':26s} {'full E':>8s} {'full S':>8s} {'small E':>9s} {'small S':>9s}")
    for label, loader, ckpt in rows:
        full_e, full_s, small_e, small_s = evaluate(loader, ckpt, strain, energy_true, stress_true, small_idx)
        print(f"{label:26s} {full_e:8.4%} {full_s:8.4%} {small_e:9.4%} {small_s:9.4%}")
