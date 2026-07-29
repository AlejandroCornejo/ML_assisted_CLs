#!/usr/bin/env python3
"""Train the one selected PANN-D4 using all ten Stage-1 trajectories.

Stage 10 is deliberately not opened.  This is the final training protocol
used for the selected direct-input PANN, not a new model-selection study.
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

from pann_d4_model import PANN_D4Energy


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "alltraj_stage10_direct_energy.npz"
CHECKPOINT_DIR = HERE / "checkpoints"
RESULT_DIR = HERE / "results"

# Fixed before the final all-trajectory training.  These are the settings that
# won the Stage-1-only selection (train trajectories 1--8, validate 9--10).
WIDTHS = (128, 128, 64)
ENERGY_WEIGHT = 0.65
STRESS_WEIGHT = 1.00
COMPONENT_STRESS_WEIGHT = 0.65
LEARNING_RATE = 4.0e-4


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto selects CUDA when available, otherwise CPU.",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--output-name", default="PANN_D4_retrained.pt")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue the one saved training state instead of restarting the fit.",
    )
    parser.add_argument(
        "--state-name",
        default="PANN_D4_training_state.pt",
        help="Filename for the resumable state inside checkpoints/.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def feature_scale(strain: torch.Tensor, strain_scale: float) -> torch.Tensor:
    """Store the historical feature scale in the checkpoint metadata."""

    normalised = strain / strain_scale
    features = torch.stack(
        (
            normalised[:, 0] + normalised[:, 1],
            (normalised[:, 0] - normalised[:, 1]) ** 2,
            normalised[:, 2] ** 2,
        ),
        dim=1,
    )
    return torch.clamp(torch.max(torch.abs(features), dim=0).values, min=1.0e-6)


def relative_l2(prediction: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(prediction - reference) / torch.clamp(torch.linalg.vector_norm(reference), min=1.0e-30))


def training_metrics(model, strain, energy, stress, strain_scale, energy_scale, batch_size) -> dict[str, float]:
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
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    print(f"Training device: {device}.", flush=True)
    with np.load(DATA_PATH) as data:
        trajectory_ids = np.asarray(data["train_trajectory_id"], dtype=int)
        strain = torch.as_tensor(np.asarray(data["train_strain"], dtype=np.float32), device=device)
        energy = torch.as_tensor(np.asarray(data["train_energy"], dtype=np.float32).reshape(-1, 1), device=device)
        stress = torch.as_tensor(np.asarray(data["train_stress"], dtype=np.float32), device=device)

    if set(np.unique(trajectory_ids)) != set(range(1, 11)):
        raise RuntimeError("The final PANN-D4 must use all ten Stage-1 trajectories.")

    strain_scale = float(torch.max(torch.abs(strain)).item())
    energy_scale = float(torch.max(torch.abs(energy)).item())
    x_all = strain / strain_scale
    energy_target = energy / energy_scale
    stress_target = stress * (strain_scale / energy_scale)
    energy_denominator = torch.clamp(torch.mean(energy_target**2), min=1.0e-12)
    stress_denominator = torch.clamp(torch.mean(stress_target**2), min=1.0e-12)
    component_denominator = torch.clamp(torch.mean(stress_target**2, dim=0), min=1.0e-12)

    model = PANN_D4Energy(WIDTHS, feature_scale(strain, strain_scale)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1.0e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=14, min_lr=2.0e-6
    )
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)
    state_path = CHECKPOINT_DIR / args.state_name
    history_path = RESULT_DIR / "PANN_D4_retraining_history.csv"
    fields = [
        "epoch", "loss", "energy_loss", "stress_global_loss", "stress_component_loss",
        "training_energy_relative_l2", "training_stress_relative_l2", "learning_rate",
    ]
    best_state, best_metrics, best_epoch, best_score = None, None, 0, float("inf")
    first_epoch = 1
    elapsed_before = 0.0
    history_mode = "w"
    if args.resume:
        if not state_path.exists():
            raise FileNotFoundError(f"No resumable state found at {state_path}.")
        saved = torch.load(state_path, map_location=device, weights_only=False)
        if int(saved["target_epochs"]) != args.epochs:
            raise ValueError(
                f"Saved state targets {saved['target_epochs']} epochs, but --epochs={args.epochs} was requested."
            )
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        best_state = saved["best_model_state_dict"]
        best_metrics = saved["best_metrics"]
        best_epoch = int(saved["best_epoch"])
        best_score = float(saved["best_score"])
        first_epoch = int(saved["completed_epoch"]) + 1
        elapsed_before = float(saved["elapsed_seconds"])
        torch.set_rng_state(saved["torch_rng_state"])
        if device.type == "cuda" and saved["cuda_rng_state_all"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng_state_all"])
        history_mode = "a"
        print(f"Resuming after epoch {first_epoch - 1}.", flush=True)
    started = time.perf_counter()

    with history_path.open(history_mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if history_mode == "w":
            writer.writeheader()
        for epoch in range(first_epoch, args.epochs + 1):
            model.train()
            permutation = torch.randperm(len(strain), device=device)
            sums = np.zeros(4, dtype=float)
            n_batches = 0
            for start in range(0, len(strain), args.batch_size):
                index = permutation[start : start + args.batch_size]
                x = x_all[index].detach().clone().requires_grad_(True)
                optimizer.zero_grad(set_to_none=True)
                predicted_energy, predicted_stress = model.energy_and_stress(x, create_graph=True)
                energy_loss = torch.mean((predicted_energy - energy_target[index]) ** 2) / energy_denominator
                global_stress_loss = torch.mean((predicted_stress - stress_target[index]) ** 2) / stress_denominator
                component_stress_loss = torch.mean(
                    torch.mean((predicted_stress - stress_target[index]) ** 2, dim=0) / component_denominator
                )
                stress_loss = (1.0 - COMPONENT_STRESS_WEIGHT) * global_stress_loss + COMPONENT_STRESS_WEIGHT * component_stress_loss
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
                metrics = training_metrics(model, strain, energy, stress, strain_scale, energy_scale, args.batch_size)
                score = metrics["energy_relative_l2"] + metrics["stress_relative_l2"]
                scheduler.step(score)
                if score < best_score:
                    best_state, best_metrics, best_epoch, best_score = copy.deepcopy(model.state_dict()), metrics, epoch, score
                print(
                    f"epoch={epoch:3d} train(W,S)=({metrics['energy_relative_l2']:.4e}, {metrics['stress_relative_l2']:.4e})",
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
            torch.save(
                {
                    "target_epochs": args.epochs,
                    "completed_epoch": epoch,
                    "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_model_state_dict": best_state,
                    "best_metrics": best_metrics,
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state_all": (
                        torch.cuda.get_rng_state_all() if device.type == "cuda" else None
                    ),
                },
                state_path,
            )

    if best_state is None or best_metrics is None:
        raise RuntimeError("No training checkpoint was selected.")
    model.load_state_dict(best_state)
    summary = {
        "model": "Selected direct-input PANN-D4",
        "model_configuration": {
            "kind": "d4_mlp",
            "widths": list(WIDTHS),
            "invariant_feature_scale": model.invariant_feature_scale.detach().cpu().tolist(),
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
            "seed": args.seed,
            "epochs": args.epochs,
            "wall_seconds": elapsed_before + time.perf_counter() - started,
        },
    }
    checkpoint_path = CHECKPOINT_DIR / args.output_name
    torch.save(summary | {"model_state_dict": model.state_dict()}, checkpoint_path)
    state_path.unlink(missing_ok=True)
    (RESULT_DIR / "PANN_D4_retraining_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
