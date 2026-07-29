#!/usr/bin/env python3
"""Evaluate the selected PANN-D4 once on the untouched Stage-10 path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pann_d4_model import load_selected_pann


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "alltraj_stage10_direct_energy.npz"
CHECKPOINT_PATH = HERE / "checkpoints" / "PANN_D4_best.pt"
RESULT_PATH = HERE / "results" / "stage10_metrics.json"


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - reference) / max(np.linalg.norm(reference), 1.0e-30))


def predict(model, strain: np.ndarray, strain_scale: float, energy_scale: float, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    energies, stresses = [], []
    with torch.enable_grad():
        for start in range(0, len(strain), batch_size):
            raw = torch.as_tensor(strain[start : start + batch_size], dtype=torch.float32)
            normalised = (raw / strain_scale).detach().clone().requires_grad_(True)
            energy, stress = model.energy_and_stress(normalised, create_graph=False)
            energies.append((energy.detach().cpu().numpy() * energy_scale).reshape(-1))
            stresses.append(stress.detach().cpu().numpy() * (energy_scale / strain_scale))
    return np.concatenate(energies), np.concatenate(stresses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
        help="PANN-D4 checkpoint to evaluate.",
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=RESULT_PATH,
        help="JSON path written unless --no-write is passed.",
    )
    args = parser.parse_args()

    model, strain_scale, energy_scale, checkpoint = load_selected_pann(args.checkpoint, torch.device("cpu"))
    with np.load(DATA_PATH) as data:
        strain = np.asarray(data["stage10_strain"], dtype=np.float32)
        direct_energy = np.asarray(data["stage10_energy"], dtype=float)
        direct_stress = np.asarray(data["stage10_stress"], dtype=float)

    pann_energy, pann_stress = predict(model, strain, strain_scale, energy_scale, args.batch_size)
    result = {
        "protocol": "Train on Stage-1 trajectories 1--10; evaluate once on untouched Stage-10.",
        "model": "Selected direct-input PANN-D4, one seed (20260731).",
        "n_stage10": int(len(strain)),
        "pann_d4_vs_direct_fom": {
            "energy_relative_l2": relative_l2(pann_energy, direct_energy),
            "stress_relative_l2": relative_l2(pann_stress, direct_stress),
            "stress_component_relative_l2": [
                relative_l2(pann_stress[:, component], direct_stress[:, component])
                for component in range(3)
            ],
        },
        "important_measure_note": (
            "The stored HPROM-ANN and D-HPROM-ANN stresses use the historical "
            "volume-average measure, whereas the PANN target is S=dW/dE. They "
            "are intentionally not ranked here as if they were identical stresses."
        ),
        "normalisation": {
            "strain_scale": strain_scale,
            "energy_scale": energy_scale,
        },
        "checkpoint_best_epoch": int(checkpoint["best_epoch"]),
    }
    print(json.dumps(result, indent=2))
    if not args.no_write:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
