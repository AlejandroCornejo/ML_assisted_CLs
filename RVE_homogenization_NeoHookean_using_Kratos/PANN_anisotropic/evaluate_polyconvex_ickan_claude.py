#!/usr/bin/env python3
"""Evaluate the anisotropic polyconvex ICKAN on Stage--10.

Mirrors evaluate_polyconvex_final_claude.py exactly, but for the ICKAN
(input-convex Kolmogorov-Arnold network) core instead of the ICNN core --
same features, same reference-state and polyconvexity certificate, same
audits (evaluate_anisotropic_pann.polyconvex_audit is agnostic to the
concrete convex core, since it only calls the model's public
energy/energy_and_stress/certificate_summary interface).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from anisotropic_pann_model_ickan_claude import load_anisotropic_polyconvex_ickan
from evaluate_anisotropic_pann import polyconvex_audit


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "PANN_D4" / "data" / "alltraj_stage10_direct_energy.npz"
HPROM_REFERENCE = HERE.parent / "PANN_D4" / "results" / "stage10_hprom_direct_reference.json"
RESULT_DIR = HERE / "results"


def sigma_eq(stress: np.ndarray) -> np.ndarray:
    sxx, syy, sxy = stress[:, 0], stress[:, 1], stress[:, 2]
    return np.sqrt(np.maximum(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy, 0.0))


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - reference) / max(np.linalg.norm(reference), 1.0e-30))


def predict(model, strain: np.ndarray, *, strain_scale: float, energy_scale: float, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    energies, stresses = [], []
    with torch.enable_grad():
        for start in range(0, len(strain), batch_size):
            raw = torch.as_tensor(strain[start:start + batch_size], dtype=torch.float64)
            normalised = (raw / strain_scale).detach().clone().requires_grad_(True)
            energy, stress = model.energy_and_stress(normalised, create_graph=False)
            energies.append((energy.detach().cpu().numpy() * energy_scale).reshape(-1))
            stresses.append(stress.detach().cpu().numpy() * (energy_scale / strain_scale))
    return np.concatenate(energies), np.concatenate(stresses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="PANN_anisotropic_polyconvex_ickan_claude.pt")
    parser.add_argument("--output-prefix", default="polyconvex_ickan_claude")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    with np.load(DATA_PATH) as data:
        strain = np.asarray(data["stage10_strain"], dtype=np.float64)
        direct_energy = np.asarray(data["stage10_energy"], dtype=np.float64)
        direct_stress = np.asarray(data["stage10_stress"], dtype=np.float64)

    model, strain_scale, energy_scale, checkpoint = load_anisotropic_polyconvex_ickan(
        HERE / "checkpoints" / args.checkpoint, torch.device("cpu")
    )
    predicted_energy, predicted_stress = predict(
        model, strain, strain_scale=strain_scale, energy_scale=energy_scale, batch_size=512
    )

    direct_eq = sigma_eq(direct_stress)
    predicted_eq = sigma_eq(predicted_stress)

    audit = polyconvex_audit(model, strain_scale=strain_scale, energy_scale=energy_scale)
    hprom = json.loads(HPROM_REFERENCE.read_text(encoding="utf-8"))["hprom_ann_vs_direct_fom"]

    result = {
        "protocol": "Train on Stage-1 trajectories 1-10 only; evaluate once on untouched Stage-10.",
        "checkpoint": args.checkpoint,
        "best_epoch": checkpoint["best_epoch"],
        "loss_weights": checkpoint["loss"],
        "ickan_hidden_widths": checkpoint["model_configuration"]["ickan_hidden_widths"],
        "ickan_grid": checkpoint["model_configuration"]["ickan_grid"],
        "ickan_spline_order": checkpoint["model_configuration"]["ickan_spline_order"],
        "n_stage10": int(len(strain)),
        "energy_relative_l2": relative_l2(predicted_energy, direct_energy),
        "stress_relative_l2": relative_l2(predicted_stress, direct_stress),
        "stress_component_relative_l2": [
            relative_l2(predicted_stress[:, component], direct_stress[:, component]) for component in range(3)
        ],
        "von_mises_relative_l2": relative_l2(predicted_eq, direct_eq),
        "direct_hprom_ann_reference_same_path": hprom,
        "guarantee_audit": audit,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "guarantee_audit"}, indent=2))

    if not args.no_write:
        RESULT_DIR.mkdir(exist_ok=True)
        (RESULT_DIR / f"{args.output_prefix}_metrics.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        np.savez_compressed(
            RESULT_DIR / f"{args.output_prefix}_predictions.npz",
            stage10_strain=strain,
            direct_energy=direct_energy,
            direct_stress=direct_stress,
            direct_von_mises=direct_eq,
            predicted_energy=predicted_energy,
            predicted_stress=predicted_stress,
            predicted_von_mises=predicted_eq,
        )


if __name__ == "__main__":
    main()
