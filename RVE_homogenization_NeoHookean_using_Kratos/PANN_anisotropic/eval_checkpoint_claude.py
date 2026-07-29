#!/usr/bin/env python3
"""Quick Stage-10 energy/stress/von-Mises + certificate check for one polyconvex checkpoint."""
import sys
import numpy as np, torch
from pathlib import Path
from anisotropic_pann_model import load_anisotropic_polyconvex
from evaluate_anisotropic_pann import polyconvex_audit


def sigma_eq(sig):
    sxx, syy, sxy = sig[:, 0], sig[:, 1], sig[:, 2]
    return np.sqrt(np.maximum(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy, 0.0))


def rel_l2(pred, ref):
    return np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), 1e-30)


def main(checkpoint_name):
    DATA_PATH = Path("../PANN_D4/data/alltraj_stage10_direct_energy.npz")
    with np.load(DATA_PATH) as data:
        strain = np.asarray(data["stage10_strain"], dtype=np.float64)
        direct_energy = np.asarray(data["stage10_energy"], dtype=np.float64)
        direct_stress = np.asarray(data["stage10_stress"], dtype=np.float64)

    model, strain_scale, energy_scale, ckpt = load_anisotropic_polyconvex(
        Path("checkpoints") / checkpoint_name, torch.device("cpu")
    )

    energies, stresses = [], []
    with torch.enable_grad():
        for start in range(0, len(strain), 512):
            raw = torch.as_tensor(strain[start:start + 512], dtype=torch.float64)
            normalised = (raw / strain_scale).detach().clone().requires_grad_(True)
            e, s = model.energy_and_stress(normalised, create_graph=False)
            energies.append((e.detach().numpy() * energy_scale).reshape(-1))
            stresses.append(s.detach().numpy() * (energy_scale / strain_scale))
    energy_pred = np.concatenate(energies)
    stress_pred = np.concatenate(stresses)

    eq_direct, eq_pred = sigma_eq(direct_stress), sigma_eq(stress_pred)

    print(f"=== {checkpoint_name} ===")
    print(f"best_epoch: {ckpt['best_epoch']}  weights: {ckpt['loss']}  widths: {ckpt['model_configuration']['icnn_widths']}")
    print(f"training_metrics: {ckpt['training_metrics']}")
    print(f"energy rel L2: {rel_l2(energy_pred, direct_energy)*100:.3f}%")
    print(f"stress rel L2 (all): {rel_l2(stress_pred, direct_stress)*100:.3f}%")
    for i, name in enumerate(["S11", "S22", "S12"]):
        print(f"  {name}: {rel_l2(stress_pred[:, i], direct_stress[:, i])*100:.3f}%")
    print(f"von Mises rel L2: {rel_l2(eq_pred, eq_direct)*100:.3f}%")

    audit = polyconvex_audit(model, strain_scale=strain_scale, energy_scale=energy_scale)
    print(f"reference stress norm: {audit['reference']['stress_norm']:.3e}")
    print(f"rank-one curvature min: {audit['rank_one_curvature_audit']['minimum_second_derivative']:.3e}")
    print(f"sampled min energy: {audit['broad_energy_sampling_audit']['minimum_energy']:.3e}")
    print()


if __name__ == "__main__":
    main(sys.argv[1])
