#!/usr/bin/env python3
"""Fit the fixed polyconvex D4 energy on all ten Stage-1 trajectories only.

The architecture is selected analytically before this script is run.  It
never opens the held-out Stage-10 arrays.  Energy and its energy-conjugate
second Piola stress are fitted simultaneously.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from polyconvex_d4_model import PolyconvexD4Energy


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "alltraj_stage10_direct_energy.npz"
CHECKPOINT_DIR = HERE / "checkpoints"
RESULT_DIR = HERE / "results"

# These settings are fixed independently of Stage-10.  The directional-minors
# class is deliberately compact: adding arbitrary couplings would invalidate
# its simple analytic polyconvexity certificate.
ICNN_WIDTHS = (64, 64, 32)
ENERGY_WEIGHT = 0.65
STRESS_WEIGHT = 1.00
COMPONENT_STRESS_WEIGHT = 0.65
LEARNING_RATE = 2.0e-3


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--output-name", default="PANN_D4_polyconvex_best.pt")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def relative_l2(prediction: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = torch.clamp(torch.linalg.vector_norm(reference), min=1.0e-30)
    return float(torch.linalg.vector_norm(prediction - reference) / denominator)


def training_metrics(
    model: PolyconvexD4Energy,
    strain: torch.Tensor,
    energy: torch.Tensor,
    stress: torch.Tensor,
    *,
    strain_scale: float,
    energy_scale: float,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    predicted_energy, predicted_stress = [], []
    with torch.enable_grad():
        for start in range(0, len(strain), batch_size):
            stop = min(start + batch_size, len(strain))
            x = (strain[start:stop] / strain_scale).detach().clone().requires_grad_(True)
            w, s = model.energy_and_stress(x, create_graph=False)
            predicted_energy.append(w * energy_scale)
            predicted_stress.append(s * (energy_scale / strain_scale))
    return {
        "energy_relative_l2": relative_l2(torch.cat(predicted_energy), energy),
        "stress_relative_l2": relative_l2(torch.cat(predicted_stress), stress),
    }


def main() -> None:
    args = arguments()
    seed_everything(args.seed)
    device = torch.device("cpu")
    with np.load(DATA_PATH) as data:
        trajectory_ids = np.asarray(data["train_trajectory_id"], dtype=int)
        strain = torch.as_tensor(np.asarray(data["train_strain"], dtype=np.float64), dtype=torch.float64, device=device)
        energy = torch.as_tensor(np.asarray(data["train_energy"], dtype=np.float64).reshape(-1, 1), dtype=torch.float64, device=device)
        stress = torch.as_tensor(np.asarray(data["train_stress"], dtype=np.float64), dtype=torch.float64, device=device)

    if set(np.unique(trajectory_ids)) != set(range(1, 11)):
        raise RuntimeError("The polyconvex PANN must use all ten Stage-1 trajectories.")

    strain_scale = float(torch.max(torch.abs(strain)).item())
    energy_scale = float(torch.max(torch.abs(energy)).item())
    x_all = strain / strain_scale
    energy_target = energy / energy_scale
    stress_target = stress * (strain_scale / energy_scale)
    energy_denominator = torch.clamp(torch.mean(energy_target.square()), min=1.0e-12)
    stress_denominator = torch.clamp(torch.mean(stress_target.square()), min=1.0e-12)
    component_denominator = torch.clamp(torch.mean(stress_target.square(), dim=0), min=1.0e-12)

    model = PolyconvexD4Energy(strain_scale=strain_scale, widths=ICNN_WIDTHS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1.0e-9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, min_lr=2.0e-6
    )
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)
    history_path = RESULT_DIR / "PANN_D4_polyconvex_training_history.csv"
    fields = [
        "epoch", "loss", "energy_loss", "stress_global_loss", "stress_component_loss",
        "training_energy_relative_l2", "training_stress_relative_l2", "learning_rate",
    ]
    best_state, best_metrics, best_epoch, best_score = None, None, 0, float("inf")
    started = time.perf_counter()

    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            permutation = torch.randperm(len(strain), device=device)
            sums = np.zeros(4, dtype=float)
            n_batches = 0
            for start in range(0, len(strain), args.batch_size):
                index = permutation[start : start + args.batch_size]
                x = x_all[index].detach().clone().requires_grad_(True)
                optimizer.zero_grad(set_to_none=True)
                predicted_energy, predicted_stress = model.energy_and_stress(x, create_graph=True)
                energy_loss = torch.mean((predicted_energy - energy_target[index]).square()) / energy_denominator
                global_stress_loss = torch.mean((predicted_stress - stress_target[index]).square()) / stress_denominator
                component_stress_loss = torch.mean(
                    torch.mean((predicted_stress - stress_target[index]).square(), dim=0) / component_denominator
                )
                stress_loss = (1.0 - COMPONENT_STRESS_WEIGHT) * global_stress_loss + COMPONENT_STRESS_WEIGHT * component_stress_loss
                loss = ENERGY_WEIGHT * energy_loss + STRESS_WEIGHT * stress_loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at epoch {epoch}.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0)
                optimizer.step()
                sums += [
                    float(loss.detach()),
                    float(energy_loss.detach()),
                    float(global_stress_loss.detach()),
                    float(component_stress_loss.detach()),
                ]
                n_batches += 1

            metrics = None
            if epoch == 1 or epoch % args.checkpoint_interval == 0 or epoch == args.epochs:
                metrics = training_metrics(
                    model, strain, energy, stress,
                    strain_scale=strain_scale, energy_scale=energy_scale, batch_size=args.batch_size,
                )
                score = metrics["energy_relative_l2"] + metrics["stress_relative_l2"]
                scheduler.step(score)
                if score < best_score:
                    best_state = copy.deepcopy(model.state_dict())
                    best_metrics, best_epoch, best_score = metrics, epoch, score
                print(
                    f"epoch={epoch:3d} train(W,S)=({metrics['energy_relative_l2']:.4e}, "
                    f"{metrics['stress_relative_l2']:.4e})",
                    flush=True,
                )
            writer.writerow(
                {
                    "epoch": epoch,
                    "loss": sums[0] / n_batches,
                    "energy_loss": sums[1] / n_batches,
                    "stress_global_loss": sums[2] / n_batches,
                    "stress_component_loss": sums[3] / n_batches,
                    "training_energy_relative_l2": float("nan") if metrics is None else metrics["energy_relative_l2"],
                    "training_stress_relative_l2": float("nan") if metrics is None else metrics["stress_relative_l2"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            file.flush()

    if best_state is None or best_metrics is None:
        raise RuntimeError("No training checkpoint was selected.")
    model.load_state_dict(best_state)
    summary = {
        "model": "Polyconvex D4 directional-minors PANN",
        "model_configuration": {
            "kind": "polyconvex_d4_directional_minors",
            "icnn_widths": list(ICNN_WIDTHS),
            "structural_directions": "axes, diagonals, and D4 orbits at 22.5 and 15 degrees",
            "direct_minor_terms": True,
            "cofactor_minor_terms": True,
        },
        "loss": {
            "energy_weight": ENERGY_WEIGHT,
            "stress_weight": STRESS_WEIGHT,
            "component_stress_weight": COMPONENT_STRESS_WEIGHT,
        },
        "strain_scale": strain_scale,
        "energy_scale": energy_scale,
        "best_epoch": best_epoch,
        "best_training_score": best_score,
        "training_metrics": best_metrics,
        "analytic_certificate": model.certificate_summary(),
        "protocol": {
            "n_train": int(len(strain)),
            "trajectory_ids": sorted(np.unique(trajectory_ids).tolist()),
            "stage10_used": False,
            "seed": args.seed,
            "epochs": args.epochs,
            "wall_seconds": time.perf_counter() - started,
        },
    }
    checkpoint_path = CHECKPOINT_DIR / args.output_name
    torch.save(summary | {"model_state_dict": model.state_dict()}, checkpoint_path)
    (RESULT_DIR / "PANN_D4_polyconvex_training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
