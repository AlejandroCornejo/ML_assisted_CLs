#!/usr/bin/env python3
"""Show what happens without a polyconvexity certificate: the non-polyconvex
'free' anisotropic PANN (unconstrained Softplus MLP on C, no convexity
constraint anywhere) versus the certified polyconvex PANNs.

This is the counterpart, on this project's own RVE, of the ellipticity-loss
demonstration Klein et al. (2022, JMPS 159:104703, their Fig. 5) show for an
uncertified flexible ANN, and of the divergence/instability behaviour As'ad,
Avery and Farhat (2022, IJNME 123:2738-2759) show in Section 4.4 for their
'standard regression ANN' baseline.

For a fixed sampling budget it (1) evaluates Stage-10 accuracy exactly like
the certified models, for a fair side-by-side comparison, and (2) samples
many more, and more aggressive, rank-one directions than the certified
models' falsification-attempt audit -- since here we expect, rather than
attempt to falsify, a violation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from anisotropic_pann_model import c_to_strain, load_anisotropic_free


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "data" / "alltraj_stage10_direct_energy.npz"
RESULT_DIR = HERE / "results"


def sigma_eq(stress: np.ndarray) -> np.ndarray:
    sxx, syy, sxy = stress[:, 0], stress[:, 1], stress[:, 2]
    return np.sqrt(np.maximum(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy, 0.0))


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - reference) / max(np.linalg.norm(reference), 1.0e-30))


def energy_from_f(model, f: torch.Tensor, *, strain_scale: float) -> torch.Tensor:
    c = f.transpose(-1, -2) @ f
    physical_strain = c_to_strain(c.reshape(-1, 2, 2))
    return model.energy(physical_strain / strain_scale)


def predict(model, strain: np.ndarray, *, strain_scale: float, energy_scale: float, dtype: torch.dtype, batch_size: int):
    energies, stresses = [], []
    with torch.enable_grad():
        for start in range(0, len(strain), batch_size):
            raw = torch.as_tensor(strain[start:start + batch_size], dtype=dtype)
            normalised = (raw / strain_scale).detach().clone().requires_grad_(True)
            energy, stress = model.energy_and_stress(normalised, create_graph=False)
            energies.append((energy.detach().cpu().numpy() * energy_scale).reshape(-1))
            stresses.append(stress.detach().cpu().numpy() * (energy_scale / strain_scale))
    return np.concatenate(energies), np.concatenate(stresses)


def rank_one_curvature_audit(
    model, *, strain_scale: float, energy_scale: float, dtype: torch.dtype,
    n_samples: int, stretch_log_range: float, seed: int,
) -> dict:
    """Sample many rank-one perturbations F + t (a⊗b) and record the sign of
    d^2/dt^2 W at t=0.  A single negative value is an ellipticity violation:
    a real (necessary-condition) failure, not a numerical-precision artifact,
    since it is checked at machine-precision float64.
    """

    generator = torch.Generator().manual_seed(seed)
    curvatures = []
    failures = []
    valid_paths = 0
    for _ in range(n_samples):
        f = torch.tensor(
            ((float(torch.exp(torch.empty((), dtype=dtype).uniform_(-stretch_log_range, stretch_log_range, generator=generator))),
              float(torch.empty((), dtype=dtype).uniform_(-0.4, 0.4, generator=generator))),
             (float(torch.empty((), dtype=dtype).uniform_(-0.4, 0.4, generator=generator)),
              float(torch.exp(torch.empty((), dtype=dtype).uniform_(-stretch_log_range, stretch_log_range, generator=generator)))),),
            dtype=dtype,
        )
        a = torch.randn(2, dtype=dtype, generator=generator)
        b = torch.randn(2, dtype=dtype, generator=generator)
        a = a / torch.linalg.vector_norm(a)
        b = b / torch.linalg.vector_norm(b)
        t = torch.zeros((), dtype=dtype, requires_grad=True)
        f_t = f + t * torch.outer(a, b)
        if torch.det(f_t).detach() <= 0.0:
            continue
        value = energy_from_f(model, f_t.reshape(1, 2, 2), strain_scale=strain_scale).sum()
        first = torch.autograd.grad(value, t, create_graph=True)[0]
        second = torch.autograd.grad(first, t)[0]
        curvature = float(second.detach() * energy_scale)
        curvatures.append(curvature)
        valid_paths += 1
        if curvature < 0.0:
            failures.append({"F": f.tolist(), "a": a.tolist(), "b": b.tolist(), "curvature": curvature})

    curvatures_arr = np.asarray(curvatures)
    return {
        "n_requested": n_samples,
        "n_valid_paths": valid_paths,
        "n_violations": int((curvatures_arr < 0.0).sum()),
        "fraction_violations": float((curvatures_arr < 0.0).mean()) if valid_paths else float("nan"),
        "minimum_curvature": float(curvatures_arr.min()) if valid_paths else float("nan"),
        "median_curvature": float(np.median(curvatures_arr)) if valid_paths else float("nan"),
        "example_violations": failures[:5],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="PANN_anisotropic_free_compw000_claude.pt")
    parser.add_argument("--output-prefix", default="free_ellipticity_claude")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--stretch-log-range", type=float, default=1.0, help="log-stretch sampling half-range; 1.0 => stretches in [e^-1, e^1] ~= [0.37, 2.72].")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    with np.load(DATA_PATH) as data:
        strain = np.asarray(data["stage10_strain"], dtype=np.float64)
        direct_energy = np.asarray(data["stage10_energy"], dtype=np.float64)
        direct_stress = np.asarray(data["stage10_stress"], dtype=np.float64)

    model, strain_scale, energy_scale, checkpoint = load_anisotropic_free(
        HERE / "checkpoints" / args.checkpoint, torch.device("cpu")
    )
    model = model.double()
    dtype = torch.float64

    predicted_energy, predicted_stress = predict(
        model, strain, strain_scale=strain_scale, energy_scale=energy_scale, dtype=dtype, batch_size=512
    )
    direct_eq, predicted_eq = sigma_eq(direct_stress), sigma_eq(predicted_stress)

    audit = rank_one_curvature_audit(
        model, strain_scale=strain_scale, energy_scale=energy_scale, dtype=dtype,
        n_samples=args.n_samples, stretch_log_range=args.stretch_log_range, seed=args.seed,
    )

    result = {
        "protocol": "Train on Stage-1 trajectories 1-10 only; evaluate once on untouched Stage-10.",
        "checkpoint": args.checkpoint,
        "best_epoch": checkpoint["best_epoch"],
        "widths": checkpoint["model_configuration"]["widths"],
        "n_stage10": int(len(strain)),
        "energy_relative_l2": relative_l2(predicted_energy, direct_energy),
        "stress_relative_l2": relative_l2(predicted_stress, direct_stress),
        "stress_component_relative_l2": [
            relative_l2(predicted_stress[:, component], direct_stress[:, component]) for component in range(3)
        ],
        "von_mises_relative_l2": relative_l2(predicted_eq, direct_eq),
        "rank_one_curvature_audit": audit,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "rank_one_curvature_audit"}, indent=2))
    print(json.dumps({k: v for k, v in audit.items() if k != "example_violations"}, indent=2))
    if audit["example_violations"]:
        print(f"First example violation (of {audit['n_violations']} found):")
        print(json.dumps(audit["example_violations"][0], indent=2))

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
