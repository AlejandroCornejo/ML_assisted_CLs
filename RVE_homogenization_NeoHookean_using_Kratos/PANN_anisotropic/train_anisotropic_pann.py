#!/usr/bin/env python3
"""Train one anisotropic energy PANN on the ten Stage--1 trajectories only.

Two architectures are available, selected before any Stage--10 evaluation:

``free``
    Smooth MLP in the material components of C and J.  It is the intended
    high-fidelity anisotropic PANN.
``metric``
    Smooth MLP in the fixed locally-isotropised tangent coordinates T e and J.
    It tests the conditioning map proposed by Rossi--Zorrilla--Codina.
``polyconvex``
    Positive-ICNN directional-minors architecture with an analytic
    polyconvexity certificate.

The script reads no Stage--10 arrays.  The held-out trajectory is opened only
by ``evaluate_anisotropic_pann.py``.
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

from anisotropic_pann_model import (
    AnisotropicFreeEnergy,
    AnisotropicPolyconvexEnergy,
    MetricPreconditionedFreeEnergy,
    material_c_features,
    strain_to_c,
)


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "PANN_D4" / "data" / "alltraj_stage10_direct_energy.npz"
METRIC_PATH = HERE / "data" / "local_isotropic_metric.json"
CHECKPOINT_DIR = HERE / "checkpoints"
RESULT_DIR = HERE / "results"

FREE_WIDTHS = (128, 128, 64)
METRIC_WIDTHS = (128, 128, 64)
POLYCONVEX_WIDTHS = (24, 24)
ENERGY_WEIGHT = 0.65
STRESS_WEIGHT = 1.00
COMPONENT_STRESS_WEIGHT = 0.65


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("free", "metric", "polyconvex"), required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--max-per-trajectory", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--polyconvex-widths",
        default=None,
        help="Comma-separated hidden widths for the certified ICNN, e.g. 48,48,32.",
    )
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--device", default="cpu", help="Torch device; the certified model uses CPU float64.")
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Explicit Torch CPU thread count; useful for reproducible small-batch second derivatives.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the exact saved optimizer/model/RNG state after an interrupted CPU run.",
    )
    parser.add_argument("--state-name", default=None, help="Resumable state filename inside checkpoints/.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def relative_l2(prediction: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(prediction - reference) / torch.clamp(torch.linalg.vector_norm(reference), min=1.0e-30))


def derive_free_feature_scale(strain: torch.Tensor, strain_scale: float) -> torch.Tensor:
    normalised = strain / strain_scale
    c = strain_to_c(normalised, torch.tensor(strain_scale, dtype=strain.dtype, device=strain.device))
    scale = torch.max(torch.abs(material_c_features(c)), dim=0).values
    return torch.clamp(scale, min=1.0e-6)


def derive_polyconvex_feature_scale(
    normalised_strain: torch.Tensor, *, strain_scale: float, widths: tuple[int, ...]
) -> torch.Tensor:
    """Positive Stage--1-only scaling of certified invariant inputs.

    Scaling a convex feature by a positive constant does not alter the
    polyconvexity construction.  It merely prevents quartic features and J^2
    from dominating the first ICNN layers numerically.
    """

    probe = AnisotropicPolyconvexEnergy(strain_scale=strain_scale, widths=widths).to(normalised_strain.device)
    maxima = None
    with torch.no_grad():
        for start in range(0, len(normalised_strain), 8192):
            features, _ = probe.structural_features(normalised_strain[start:start + 8192])
            current = torch.max(torch.abs(features), dim=0).values
            maxima = current if maxima is None else torch.maximum(maxima, current)
    if maxima is None:
        raise RuntimeError("Cannot derive feature scales from an empty training set.")
    return torch.clamp(maxima, min=1.0e-6)


def training_metrics(model, strain, energy, stress, *, strain_scale, energy_scale, batch_size) -> dict[str, float]:
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
    if args.threads is not None:
        if args.threads < 1:
            raise ValueError("--threads must be positive.")
        torch.set_num_threads(args.threads)
    if args.kind in ("free", "metric"):
        epochs = 350 if args.epochs is None else args.epochs
        batch_size = 2048 if args.batch_size is None else args.batch_size
        default_seed = 20260811 if args.kind == "free" else 20260814
        seed = default_seed if args.seed is None else args.seed
        learning_rate = 4.0e-4 if args.learning_rate is None else args.learning_rate
        default_output = "PANN_anisotropic_free.pt" if args.kind == "free" else "PANN_anisotropic_metric.pt"
        output_name = default_output if args.output_name is None else args.output_name
        dtype = torch.float32
    else:
        epochs = 700 if args.epochs is None else args.epochs
        batch_size = 4096 if args.batch_size is None else args.batch_size
        seed = 20260812 if args.seed is None else args.seed
        learning_rate = 2.0e-3 if args.learning_rate is None else args.learning_rate
        output_name = "PANN_anisotropic_polyconvex.pt" if args.output_name is None else args.output_name
        dtype = torch.float64
    if args.polyconvex_widths is not None:
        if args.kind != "polyconvex":
            raise ValueError("--polyconvex-widths applies only to --kind polyconvex.")
        try:
            polyconvex_widths = tuple(int(value) for value in args.polyconvex_widths.split(","))
        except ValueError as error:
            raise ValueError("--polyconvex-widths must be a comma-separated list of positive integers.") from error
        if not polyconvex_widths or any(value < 1 for value in polyconvex_widths):
            raise ValueError("--polyconvex-widths must contain positive integers.")
    else:
        polyconvex_widths = POLYCONVEX_WIDTHS
    if args.device != "cpu" and args.kind == "polyconvex":
        raise ValueError("The polyconvex implementation is intentionally run in CPU float64 for its audits.")
    device = torch.device(args.device)
    seed_everything(seed)
    print(f"Training {args.kind} anisotropic PANN on {device}, dtype={dtype}.", flush=True)

    with np.load(DATA_PATH) as data:
        trajectory_ids = np.asarray(data["train_trajectory_id"], dtype=int)
        strain = torch.as_tensor(np.asarray(data["train_strain"], dtype=np.float64), dtype=dtype, device=device)
        energy = torch.as_tensor(np.asarray(data["train_energy"], dtype=np.float64).reshape(-1, 1), dtype=dtype, device=device)
        stress = torch.as_tensor(np.asarray(data["train_stress"], dtype=np.float64), dtype=dtype, device=device)
    if set(np.unique(trajectory_ids)) != set(range(1, 11)):
        raise RuntimeError("The final anisotropic models must use all ten Stage-1 trajectories.")
    if args.max_per_trajectory is not None:
        if args.max_per_trajectory < 2:
            raise ValueError("--max-per-trajectory must be at least two.")
        selected = []
        ids_cpu = trajectory_ids
        for trajectory in range(1, 11):
            candidates = np.flatnonzero(ids_cpu == trajectory)
            selected.append(candidates[np.linspace(0, len(candidates) - 1, min(args.max_per_trajectory, len(candidates)), dtype=int)])
        index = torch.as_tensor(np.concatenate(selected), dtype=torch.long, device=device)
        strain, energy, stress = strain[index], energy[index], stress[index]
        trajectory_ids = trajectory_ids[index.cpu().numpy()]

    strain_scale = float(torch.max(torch.abs(strain)).item())
    energy_scale = float(torch.max(torch.abs(energy)).item())
    x_all = strain / strain_scale
    energy_target = energy / energy_scale
    stress_target = stress * (strain_scale / energy_scale)
    energy_denominator = torch.clamp(torch.mean(energy_target.square()), min=1.0e-12)
    stress_denominator = torch.clamp(torch.mean(stress_target.square()), min=1.0e-12)
    component_denominator = torch.clamp(torch.mean(stress_target.square(), dim=0), min=1.0e-12)

    if args.kind == "free":
        feature_scale = derive_free_feature_scale(strain, strain_scale).to(torch.float32)
        model = AnisotropicFreeEnergy(
            strain_scale=strain_scale, feature_scale=feature_scale, widths=FREE_WIDTHS
        ).to(device)
        configuration = {
            "kind": "anisotropic_c_mlp",
            "widths": list(FREE_WIDTHS),
            "features": ["a0.C.a0 - 1", "b0.C.b0 - 1", "a0.C.b0", "J - 1"],
            "feature_scale": feature_scale.detach().cpu().tolist(),
            "d4_average": False,
        }
    elif args.kind == "metric":
        if not METRIC_PATH.exists():
            raise FileNotFoundError(
                f"Missing {METRIC_PATH}. Run make_local_isotropic_mapping.py before training PANN-T."
            )
        metric = json.loads(METRIC_PATH.read_text(encoding="utf-8"))
        transform = torch.tensor(metric["metric_transform_T"], dtype=torch.float32)
        model = MetricPreconditionedFreeEnergy(
            strain_scale=strain_scale, metric_transform=transform, widths=METRIC_WIDTHS
        ).to(device)
        configuration = {
            "kind": "anisotropic_metric_mlp",
            "widths": list(METRIC_WIDTHS),
            "features": ["(T e)_11 / strain_scale", "(T e)_22 / strain_scale", "(T e)_12 / strain_scale", "J - 1"],
            "metric_transform": metric["metric_transform_T"],
            "metric_source": metric["source"],
            "closest_isotropic_projection": metric["closest_isotropic_projection"],
            "d4_average": False,
        }
    else:
        feature_scale = derive_polyconvex_feature_scale(
            x_all, strain_scale=strain_scale, widths=polyconvex_widths
        )
        model = AnisotropicPolyconvexEnergy(
            strain_scale=strain_scale, widths=polyconvex_widths, feature_scale=feature_scale
        ).to(device)
        configuration = {
            "kind": "anisotropic_polyconvex_directional_minors",
            "icnn_widths": list(polyconvex_widths),
            "features": "tr(C), balanced quartic and sixth-power directional invariants sum_i w_ki|F d_ki|^(4,6) and sum_i w_ki|cof(F)d_ki|^(4,6), plus J and J^2",
            "feature_scale": feature_scale.detach().cpu().tolist(),
            "d4_average": False,
        }

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=18, min_lr=2.0e-6
    )
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)
    stem = {
        "free": "PANN_anisotropic_free",
        "metric": "PANN_anisotropic_metric",
        "polyconvex": "PANN_anisotropic_polyconvex",
    }[args.kind]
    history_path = RESULT_DIR / f"{stem}_training_history.csv"
    state_name = f"{stem}_training_state.pt" if args.state_name is None else args.state_name
    state_path = CHECKPOINT_DIR / state_name
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
        if saved["kind"] != args.kind or int(saved["target_epochs"]) != epochs:
            raise ValueError("The resumable state belongs to a different model kind or target epoch count.")
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
        for epoch in range(first_epoch, epochs + 1):
            model.train()
            permutation = torch.randperm(len(strain), device=device)
            sums = np.zeros(4, dtype=float)
            n_batches = 0
            for start in range(0, len(strain), batch_size):
                index = permutation[start:start + batch_size]
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
                sums += [float(loss.detach()), float(energy_loss.detach()), float(global_stress_loss.detach()), float(component_stress_loss.detach())]
                n_batches += 1

            metrics = None
            if epoch == 1 or epoch % args.checkpoint_interval == 0 or epoch == epochs:
                metrics = training_metrics(
                    model, strain, energy, stress, strain_scale=strain_scale, energy_scale=energy_scale, batch_size=batch_size
                )
                score = metrics["energy_relative_l2"] + metrics["stress_relative_l2"]
                scheduler.step(score)
                if score < best_score:
                    best_state = copy.deepcopy(model.state_dict())
                    best_metrics, best_epoch, best_score = metrics, epoch, score
                print(
                    f"epoch={epoch:4d} train(W,S)=({metrics['energy_relative_l2']:.4e}, {metrics['stress_relative_l2']:.4e})",
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
            # Persist together with the declared metric checks. This preserves
            # resumability without turning a small constitutive fit into an
            # I/O-bound workload on the mounted workspace.
            if epoch % args.checkpoint_interval == 0 or epoch == epochs:
                torch.save(
                {
                    "kind": args.kind,
                    "target_epochs": epochs,
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
                    "cuda_rng_state_all": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                },
                    state_path,
                )

    if best_state is None or best_metrics is None:
        raise RuntimeError("No training checkpoint was selected.")
    model.load_state_dict(best_state)
    summary = {
        "model": {
            "free": "Free anisotropic C-based PANN",
            "metric": "Metric-preconditioned anisotropic PANN-T",
            "polyconvex": "Anisotropic polyconvex directional-minors PANN",
        }[args.kind],
        "model_configuration": configuration,
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
        "analytic_certificate": model.certificate_summary() if args.kind == "polyconvex" else None,
        "protocol": {
            "n_train": int(len(strain)),
            "trajectory_ids": sorted(np.unique(trajectory_ids).tolist()),
            "stage10_used": False,
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "max_per_trajectory": args.max_per_trajectory,
            "torch_threads": torch.get_num_threads(),
            "wall_seconds": elapsed_before + time.perf_counter() - started,
        },
    }
    torch.save(summary | {"model_state_dict": model.state_dict()}, CHECKPOINT_DIR / output_name)
    state_path.unlink(missing_ok=True)
    (RESULT_DIR / f"{stem}_training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
