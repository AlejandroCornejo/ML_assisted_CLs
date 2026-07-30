#!/usr/bin/env python3
"""Train the tier-1 direct-regression baseline (no energy potential) on the
ten Stage--1 trajectories only, exactly like the other three models in this
study.  The held-out trajectory is opened only by the evaluation scripts.
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

from anisotropic_pann_model_regression_claude import AnisotropicRegressionStress


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "data" / "alltraj_stage10_direct_energy.npz"
CHECKPOINT_DIR = HERE / "checkpoints"
RESULT_DIR = HERE / "results"


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def relative_l2(prediction: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(prediction - reference) / torch.clamp(torch.linalg.vector_norm(reference), min=1.0e-30))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--learning-rate", type=float, default=4.0e-4)
    parser.add_argument("--widths", default="128,128,64")
    parser.add_argument("--component-stress-weight", type=float, default=0.0)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-name", default="PANN_anisotropic_regression_claude.pt")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    widths = tuple(int(v) for v in args.widths.split(","))
    dtype = torch.float32
    device = torch.device("cpu")
    seed_everything(args.seed)
    print(f"Training direct-regression baseline on {device}, dtype={dtype}, widths={widths}.", flush=True)

    with np.load(DATA_PATH) as data:
        trajectory_ids = np.asarray(data["train_trajectory_id"], dtype=int)
        strain = torch.as_tensor(np.asarray(data["train_strain"], dtype=np.float64), dtype=dtype, device=device)
        stress = torch.as_tensor(np.asarray(data["train_stress"], dtype=np.float64), dtype=dtype, device=device)
        energy = torch.as_tensor(np.asarray(data["train_energy"], dtype=np.float64).reshape(-1, 1), dtype=dtype, device=device)
    if set(np.unique(trajectory_ids)) != set(range(1, 11)):
        raise RuntimeError("The final models must use all ten Stage-1 trajectories.")

    strain_scale = float(torch.max(torch.abs(strain)).item())
    energy_scale = float(torch.max(torch.abs(energy)).item())
    x_all = strain / strain_scale
    stress_target = stress * (strain_scale / energy_scale)
    stress_denominator = torch.clamp(torch.mean(stress_target.square()), min=1.0e-12)
    component_denominator = torch.clamp(torch.mean(stress_target.square(), dim=0), min=1.0e-12)

    model = AnisotropicRegressionStress(strain_scale=strain_scale, widths=widths).to(device)
    configuration = {
        "kind": "anisotropic_regression_baseline",
        "widths": list(widths),
        "features": "normalized [E11, E22, gamma12] directly, no potential",
        "d4_average": False,
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=18, min_lr=2.0e-6)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)
    stem = Path(args.output_name).stem
    history_path = RESULT_DIR / f"{stem}_training_history.csv"
    fields = ["epoch", "loss", "training_stress_relative_l2", "learning_rate"]
    best_state, best_metrics, best_epoch, best_score = None, None, 0, float("inf")
    started = time.perf_counter()

    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            permutation = torch.randperm(len(strain), device=device)
            loss_sum, n_batches = 0.0, 0
            for start in range(0, len(strain), args.batch_size):
                index = permutation[start:start + args.batch_size]
                x = x_all[index]
                optimizer.zero_grad(set_to_none=True)
                predicted_stress = model.stress(x)
                global_loss = torch.mean((predicted_stress - stress_target[index]).square()) / stress_denominator
                component_loss = torch.mean(
                    torch.mean((predicted_stress - stress_target[index]).square(), dim=0) / component_denominator
                )
                loss = (1.0 - args.component_stress_weight) * global_loss + args.component_stress_weight * component_loss
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach())
                n_batches += 1

            metrics = None
            if epoch == 1 or epoch % args.checkpoint_interval == 0 or epoch == args.epochs:
                model.eval()
                with torch.no_grad():
                    predicted_all = model.stress(x_all)
                score = relative_l2(predicted_all, stress_target)
                metrics = {"training_stress_relative_l2": score}
                scheduler.step(score)
                if score < best_score:
                    best_state = copy.deepcopy(model.state_dict())
                    best_metrics, best_epoch, best_score = metrics, epoch, score
                print(f"epoch={epoch:4d} train(S)={score:.4e}", flush=True)
            writer.writerow({
                "epoch": epoch,
                "loss": loss_sum / n_batches,
                "training_stress_relative_l2": float("nan") if metrics is None else metrics["training_stress_relative_l2"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            })
            file.flush()

    if best_state is None or best_metrics is None:
        raise RuntimeError("No training checkpoint was selected.")
    model.load_state_dict(best_state)
    summary = {
        "model": "Anisotropic direct-regression baseline (tier 1: no energy potential)",
        "model_configuration": configuration,
        "loss": {"component_stress_weight": args.component_stress_weight},
        "strain_scale": strain_scale,
        "energy_scale": energy_scale,
        "best_epoch": best_epoch,
        "best_training_score": best_score,
        "training_metrics": best_metrics,
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
    torch.save(summary | {"model_state_dict": model.state_dict()}, CHECKPOINT_DIR / args.output_name)
    (RESULT_DIR / f"{stem}_training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
