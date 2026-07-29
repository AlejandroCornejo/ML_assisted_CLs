#!/usr/bin/env python3
"""Train a D4-wrapped *actual ICKAN* on the direct Stage--1 labels only.

This is intentionally the same protocol as the final PANN-D4: all ten
Stage--1 trajectories train the potential; Stage--10 is not opened by this
script.  It supports two predeclared, non-Stage--10-selected modes:
``direct`` and ``minor_features`` (see ``ickan_d4_model.py``).
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

from ickan_d4_model import D4ICKANEnergy


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "alltraj_stage10_direct_energy.npz"
CHECKPOINT_DIR = HERE / "checkpoints"
RESULT_DIR = HERE / "results"

ENERGY_WEIGHT = 0.65
STRESS_WEIGHT = 1.00
COMPONENT_STRESS_WEIGHT = 0.65


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("direct", "minor_features"), default="direct")
    parser.add_argument("--width", type=int, default=0, help="Hidden KAN width; 0 selects a one-layer KAN.")
    parser.add_argument("--grid", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--evaluation-batch-size", type=int, default=512,
        help="Smaller batch for full-dataset derivative metrics, which retain an autograd graph.",
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument(
        "--threads", type=int, default=1,
        help="PyTorch CPU threads. One avoids severe B-spline/autograd oversubscription.",
    )
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--grid-samples", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--output-name", default=None)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def relative_l2(prediction: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(prediction - reference)
        / torch.clamp(torch.linalg.vector_norm(reference), min=1.0e-30)
    )


def training_metrics(model, strain, energy, stress, strain_scale, energy_scale, batch_size) -> dict[str, float]:
    model.eval()
    energies, stresses = [], []
    with torch.enable_grad():
        for start in range(0, len(strain), batch_size):
            stop = min(start + batch_size, len(strain))
            x = (strain[start:stop] / strain_scale).detach().clone().requires_grad_(True)
            predicted_energy, predicted_stress = model.energy_and_stress(x, create_graph=False)
            # ``energy`` retains the parameter graph even when the stress
            # derivative itself does not.  Detach before accumulating all
            # batches, otherwise a full metric pass needlessly keeps the
            # entire ICKAN graph in memory.
            energies.append((predicted_energy * energy_scale).detach())
            stresses.append((predicted_stress * (energy_scale / strain_scale)).detach())
    return {
        "energy_relative_l2": relative_l2(torch.cat(energies), energy),
        "stress_relative_l2": relative_l2(torch.cat(stresses), stress),
    }


def main() -> None:
    args = arguments()
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    with np.load(DATA_PATH) as data:
        trajectory_ids = np.asarray(data["train_trajectory_id"], dtype=int)
        strain = torch.as_tensor(np.asarray(data["train_strain"], dtype=np.float32), device=device)
        energy = torch.as_tensor(np.asarray(data["train_energy"], dtype=np.float32).reshape(-1, 1), device=device)
        stress = torch.as_tensor(np.asarray(data["train_stress"], dtype=np.float32), device=device)
    if set(np.unique(trajectory_ids)) != set(range(1, 11)):
        raise RuntimeError("The ICKAN-D4 comparison must use all ten Stage-1 trajectories.")

    strain_scale = float(torch.max(torch.abs(strain)).item())
    energy_scale = float(torch.max(torch.abs(energy)).item())
    x_all = strain / strain_scale
    energy_target = energy / energy_scale
    stress_target = stress * (strain_scale / energy_scale)
    energy_denominator = torch.clamp(torch.mean(energy_target.square()), min=1.0e-12)
    stress_denominator = torch.clamp(torch.mean(stress_target.square()), min=1.0e-12)
    component_denominator = torch.clamp(torch.mean(stress_target.square(), dim=0), min=1.0e-12)

    model = D4ICKANEnergy(
        strain_scale=strain_scale,
        mode=args.mode,
        widths=() if args.width == 0 else (args.width,),
        grid=args.grid,
        spline_order=3,
        seed=args.seed,
    ).to(device)
    # The ICKAN fork's *multi-layer* ``update_grid_from_samples`` can turn
    # its reference-state derivative into NaN after sequentially remapping
    # hidden grids.  Its default grid already spans the normalized direct
    # input range [-1,1], so the reproducible baseline fixes that grid.  A
    # positive value remains available for diagnosing a future upstream fix,
    # but is deliberately not used in the reported comparison.
    grid_count = min(args.grid_samples, len(x_all))
    if grid_count:
        grid_indices = torch.randperm(len(x_all), device=device)[:grid_count]
        model.initialise_grid(x_all[grid_indices])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=18, min_lr=2.0e-6
    )
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)
    tag = f"ICKAN_D4_{args.mode}"
    history_path = RESULT_DIR / f"{tag}_training_history.csv"
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
            totals, n_batches = np.zeros(4, dtype=float), 0
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)
                optimizer.step()
                totals += [
                    float(loss.detach()), float(energy_loss.detach()),
                    float(global_stress_loss.detach()), float(component_stress_loss.detach()),
                ]
                n_batches += 1

            metrics = None
            if epoch == 1 or epoch % args.checkpoint_interval == 0 or epoch == args.epochs:
                metrics = training_metrics(
                    model, strain, energy, stress, strain_scale, energy_scale, args.evaluation_batch_size
                )
                score = metrics["energy_relative_l2"] + metrics["stress_relative_l2"]
                scheduler.step(score)
                if score < best_score:
                    best_state = copy.deepcopy(model.state_dict())
                    best_metrics, best_epoch, best_score = metrics, epoch, score
                print(
                    f"mode={args.mode} epoch={epoch:3d} train(W,S)="
                    f"({metrics['energy_relative_l2']:.4e}, {metrics['stress_relative_l2']:.4e})",
                    flush=True,
                )
            writer.writerow(
                {
                    "epoch": epoch,
                    "loss": totals[0] / n_batches,
                    "energy_loss": totals[1] / n_batches,
                    "stress_global_loss": totals[2] / n_batches,
                    "stress_component_loss": totals[3] / n_batches,
                    "training_energy_relative_l2": float("nan") if metrics is None else metrics["energy_relative_l2"],
                    "training_stress_relative_l2": float("nan") if metrics is None else metrics["stress_relative_l2"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            file.flush()

    if best_state is None or best_metrics is None:
        raise RuntimeError("No ICKAN-D4 checkpoint was selected.")
    model.load_state_dict(best_state)
    output_name = args.output_name or f"{tag}_best.pt"
    summary = {
        "model": "D4-wrapped ICKAN spline energy",
        "model_configuration": {
            "kind": "d4_ickan",
            "mode": args.mode,
            "widths": [] if args.width == 0 else [args.width],
            "grid": args.grid,
            "spline_order": 3,
            "seed": args.seed,
            "torch_threads": args.threads,
            "evaluation_batch_size": args.evaluation_batch_size,
            "base_fun": "zero",
            "guarantee_note": (
                "Exact objectivity through E, D4 group averaging, reference state and energy-stress consistency; "
                "convexity in E for the direct-input ICKAN."
                if args.mode == "direct"
                else "Exact objectivity, D4 symmetry, reference state, energy-stress consistency, and polyconvexity: "
                "the monotone-convex ICKAN is evaluated on directional F/cof(F) measures and J, with a -r log(J) barrier."
            ),
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
        "protocol": {
            "n_train": int(len(strain)),
            "trajectory_ids": sorted(np.unique(trajectory_ids).tolist()),
            "stage10_used": False,
            "grid_samples": grid_count,
            "grid_update_note": (
                "0 in the reported run: fixed initial grids. The fork's multi-layer adaptive "
                "grid update was found to create a NaN reference derivative."
            ),
            "seed": args.seed,
            "epochs": args.epochs,
            "wall_seconds": time.perf_counter() - started,
        },
    }
    torch.save(summary | {"model_state_dict": model.state_dict()}, CHECKPOINT_DIR / output_name)
    (RESULT_DIR / f"{tag}_training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
