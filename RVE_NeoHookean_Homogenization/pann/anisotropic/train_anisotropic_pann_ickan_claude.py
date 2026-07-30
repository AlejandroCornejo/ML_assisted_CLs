#!/usr/bin/env python3
"""Train the anisotropic polyconvex ICKAN (input-convex Kolmogorov-Arnold
network) on the ten Stage--1 trajectories only.

Same balanced-direction features, reference-state cancellation, and
volumetric barrier as the certified ICNN model
(``anisotropic_pann_model.AnisotropicPolyconvexEnergy``); only the convex
core ``Phi_theta`` is replaced by an ICKAN
(``anisotropic_pann_model_ickan_claude.AnisotropicPolyconvexEnergyICKAN``),
using the reference implementation from Thakolkaran et al. 2025 ("Can KAN
CANs?", CMAME 443:118089) at https://github.com/mmc-group/ICKANs.

The script reads no Stage--10 arrays.  The held-out trajectory is opened
only by ``evaluate_polyconvex_ickan_claude.py``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from anisotropic_pann_model_ickan_claude import AnisotropicPolyconvexEnergyICKAN
from train_anisotropic_pann_claude import (
    derive_polyconvex_feature_scale,
    relative_l2,
    seed_everything,
    training_metrics,
)


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "data" / "alltraj_stage10_direct_energy.npz"
CHECKPOINT_DIR = HERE / "checkpoints"
RESULT_DIR = HERE / "results"

ENERGY_WEIGHT = 0.65
STRESS_WEIGHT = 1.00


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument(
        "--ickan-hidden", default="8", help="Comma-separated ICKAN hidden widths, e.g. 8,8."
    )
    parser.add_argument("--ickan-grid", type=int, default=6, help="Number of B-spline grid intervals.")
    parser.add_argument("--ickan-spline-order", type=int, default=3)
    parser.add_argument(
        "--ickan-grid-range", default="0.0,1.3", help="Comma-separated (min,max) for the shared spline grid range."
    )
    parser.add_argument("--component-stress-weight", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    ickan_hidden = tuple(int(v) for v in args.ickan_hidden.split(","))
    grid_lo, grid_hi = (float(v) for v in args.ickan_grid_range.split(","))
    dtype = torch.float64
    device = torch.device("cpu")
    seed_everything(args.seed)
    print(
        f"Training polyconvex ICKAN on {device}, dtype={dtype}, "
        f"hidden={ickan_hidden}, grid={args.ickan_grid}, k={args.ickan_spline_order}.",
        flush=True,
    )

    with np.load(DATA_PATH) as data:
        trajectory_ids = np.asarray(data["train_trajectory_id"], dtype=int)
        strain = torch.as_tensor(np.asarray(data["train_strain"], dtype=np.float64), dtype=dtype, device=device)
        energy = torch.as_tensor(np.asarray(data["train_energy"], dtype=np.float64).reshape(-1, 1), dtype=dtype, device=device)
        stress = torch.as_tensor(np.asarray(data["train_stress"], dtype=np.float64), dtype=dtype, device=device)
    if set(np.unique(trajectory_ids)) != set(range(1, 11)):
        raise RuntimeError("The final anisotropic models must use all ten Stage-1 trajectories.")

    strain_scale = float(torch.max(torch.abs(strain)).item())
    energy_scale = float(torch.max(torch.abs(energy)).item())
    x_all = strain / strain_scale
    energy_target = energy / energy_scale
    stress_target = stress * (strain_scale / energy_scale)
    energy_denominator = torch.clamp(torch.mean(energy_target.square()), min=1.0e-12)
    stress_denominator = torch.clamp(torch.mean(stress_target.square()), min=1.0e-12)
    component_denominator = torch.clamp(torch.mean(stress_target.square(), dim=0), min=1.0e-12)

    feature_scale = derive_polyconvex_feature_scale(x_all, strain_scale=strain_scale, widths=ickan_hidden)
    model = AnisotropicPolyconvexEnergyICKAN(
        strain_scale=strain_scale,
        ickan_hidden=ickan_hidden,
        ickan_grid=args.ickan_grid,
        ickan_spline_order=args.ickan_spline_order,
        ickan_grid_range=(grid_lo, grid_hi),
        ickan_seed=args.seed,
        feature_scale=feature_scale,
    ).to(device)
    configuration = {
        "kind": "anisotropic_polyconvex_ickan_directional_minors",
        "ickan_hidden_widths": list(ickan_hidden),
        "ickan_grid": args.ickan_grid,
        "ickan_spline_order": args.ickan_spline_order,
        "ickan_grid_range": [grid_lo, grid_hi],
        "features": "tr(C), balanced quartic and sixth-power directional invariants sum_i w_ki|F d_ki|^(4,6) and sum_i w_ki|cof(F)d_ki|^(4,6), plus J and J^2",
        "feature_scale": feature_scale.detach().cpu().tolist(),
        "d4_average": False,
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=18, min_lr=2.0e-6)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)
    output_name = args.output_name or "PANN_anisotropic_polyconvex_ickan_claude.pt"
    stem = Path(output_name).stem
    history_path = RESULT_DIR / f"{stem}_training_history.csv"
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
                index = permutation[start:start + args.batch_size]
                x = x_all[index].detach().clone().requires_grad_(True)
                optimizer.zero_grad(set_to_none=True)
                predicted_energy, predicted_stress = model.energy_and_stress(x, create_graph=True)
                energy_loss = torch.mean((predicted_energy - energy_target[index]).square()) / energy_denominator
                global_stress_loss = torch.mean((predicted_stress - stress_target[index]).square()) / stress_denominator
                component_stress_loss = torch.mean(
                    torch.mean((predicted_stress - stress_target[index]).square(), dim=0) / component_denominator
                )
                stress_loss = (
                    (1.0 - args.component_stress_weight) * global_stress_loss
                    + args.component_stress_weight * component_stress_loss
                )
                loss = ENERGY_WEIGHT * energy_loss + STRESS_WEIGHT * stress_loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at epoch {epoch}.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)
                optimizer.step()
                sums += [float(loss.detach()), float(energy_loss.detach()), float(global_stress_loss.detach()), float(component_stress_loss.detach())]
                n_batches += 1

            metrics = None
            if epoch == 1 or epoch % args.checkpoint_interval == 0 or epoch == args.epochs:
                metrics = training_metrics(
                    model, strain, energy, stress, strain_scale=strain_scale, energy_scale=energy_scale, batch_size=args.batch_size
                )
                score = metrics["energy_relative_l2"] + metrics["stress_relative_l2"]
                scheduler.step(score)
                if score < best_score:
                    best_state = copy.deepcopy(model.state_dict())
                    best_metrics, best_epoch, best_score = metrics, epoch, score
                elapsed = time.perf_counter() - started
                print(
                    f"epoch={epoch:4d} train(W,S)=({metrics['energy_relative_l2']:.4e}, {metrics['stress_relative_l2']:.4e}) "
                    f"elapsed={elapsed:.0f}s",
                    flush=True,
                )
            writer.writerow({
                "epoch": epoch,
                "loss": sums[0] / n_batches,
                "energy_loss": sums[1] / n_batches,
                "stress_global_loss": sums[2] / n_batches,
                "stress_component_loss": sums[3] / n_batches,
                "training_energy_relative_l2": float("nan") if metrics is None else metrics["energy_relative_l2"],
                "training_stress_relative_l2": float("nan") if metrics is None else metrics["stress_relative_l2"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            })
            file.flush()

    if best_state is None or best_metrics is None:
        raise RuntimeError("No training checkpoint was selected.")
    model.load_state_dict(best_state)
    summary = {
        "model": "Anisotropic polyconvex ICKAN directional-minors PANN",
        "model_configuration": configuration,
        "loss": {
            "energy_weight": ENERGY_WEIGHT,
            "stress_weight": STRESS_WEIGHT,
            "component_stress_weight": args.component_stress_weight,
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
            "batch_size": args.batch_size,
            "torch_threads": torch.get_num_threads(),
            "wall_seconds": time.perf_counter() - started,
        },
    }
    torch.save(summary | {"model_state_dict": model.state_dict()}, CHECKPOINT_DIR / output_name)
    (RESULT_DIR / f"{stem}_training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "model_state_dict"}, indent=2))


if __name__ == "__main__":
    main()
