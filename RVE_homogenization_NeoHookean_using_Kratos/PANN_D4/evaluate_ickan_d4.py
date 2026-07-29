#!/usr/bin/env python3
"""Evaluate a preselected ICKAN-D4 checkpoint on untouched Stage--10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ickan_d4_model import load_ickan_d4


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "alltraj_stage10_direct_energy.npz"


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - reference) / max(np.linalg.norm(reference), 1.0e-30))


def predict(model, strain: np.ndarray, strain_scale: float, energy_scale: float, batch_size: int):
    energies, stresses = [], []
    with torch.enable_grad():
        for start in range(0, len(strain), batch_size):
            raw = torch.as_tensor(strain[start : start + batch_size], dtype=torch.float32)
            x = (raw / strain_scale).detach().clone().requires_grad_(True)
            energy, stress = model.energy_and_stress(x, create_graph=False)
            energies.append((energy.detach().cpu().numpy() * energy_scale).reshape(-1))
            stresses.append(stress.detach().cpu().numpy() * (energy_scale / strain_scale))
    return np.concatenate(energies), np.concatenate(stresses)


def reference_and_d4_audit(model, strain_scale: float, energy_scale: float) -> dict[str, float]:
    """Check the constraints enforced by the wrapper, independently of error."""

    with torch.enable_grad():
        zero = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)
        energy_zero, stress_zero = model.energy_and_stress(zero, create_graph=False)
        # A generic state and its 90 degree square action.
        x = torch.tensor([[0.18, -0.04, 0.035]], dtype=torch.float32, requires_grad=True)
        x_rotated = torch.tensor([[-0.04, 0.18, -0.035]], dtype=torch.float32, requires_grad=True)
        w, s = model.energy_and_stress(x, create_graph=False)
        w_rotated, s_rotated = model.energy_and_stress(x_rotated, create_graph=False)
    expected_rotated_stress = torch.stack((s[:, 1], s[:, 0], -s[:, 2]), dim=1)
    return {
        "reference_energy_abs_pa": float(torch.abs(energy_zero * energy_scale).item()),
        "reference_stress_norm_pa": float(torch.linalg.vector_norm(stress_zero * energy_scale / strain_scale).item()),
        "d4_energy_abs_pa": float(torch.abs((w - w_rotated) * energy_scale).item()),
        "d4_stress_covariance_norm_pa": float(
            torch.linalg.vector_norm((s_rotated - expected_rotated_stress) * energy_scale / strain_scale).item()
        ),
    }


def polyconvex_audit(model, strain_scale: float, energy_scale: float) -> dict:
    """Numerical checks that accompany (but do not replace) the proof."""

    if model.mode != "minor_features":
        return {}

    def diagonal_energy(j_values: list[float]) -> list[float]:
        strain = torch.tensor(
            [[0.5 * (j * j - 1.0), 0.0, 0.0] for j in j_values], dtype=torch.float32
        ) / strain_scale
        return (model.energy(strain).detach().cpu().numpy().reshape(-1) * energy_scale).tolist()

    # The model is float32; constructing E=0.5(J^2-1) below J=1e-3 loses
    # det(C) to round-off.  The analytic -r log(J) term still establishes the
    # limit J->0+, while this finite audit stays in representable arithmetic.
    compression_j = [1.0, 1.0e-1, 1.0e-2, 1.0e-3]
    expansion_j = [1.0, 2.0, 10.0, 100.0]
    generator = torch.Generator().manual_seed(20260805)
    min_curvature, n_paths = float("inf"), 0
    # PConv is already guaranteed analytically.  These 32 paths are merely a
    # transparent numerical attempt to falsify rank-one convexity.
    for _ in range(32):
        f = torch.tensor(
            [[float(torch.exp(torch.empty(()).uniform_(-1.0, 1.0, generator=generator))),
              float(torch.empty(()).uniform_(-0.4, 0.4, generator=generator))],
             [0.0, float(torch.exp(torch.empty(()).uniform_(-1.0, 1.0, generator=generator)))]],
            dtype=torch.float32,
        )
        a, b = torch.randn(2, generator=generator), torch.randn(2, generator=generator)
        a, b = a / torch.linalg.vector_norm(a), b / torch.linalg.vector_norm(b)
        t = torch.zeros((), dtype=torch.float32, requires_grad=True)
        f_t = f + t * torch.outer(a, b)
        c = f_t.T @ f_t
        e = torch.stack((0.5 * (c[0, 0] - 1.0), 0.5 * (c[1, 1] - 1.0), c[0, 1])).reshape(1, 3)
        value = model.energy(e / strain_scale).sum()
        first = torch.autograd.grad(value, t, create_graph=True)[0]
        second = torch.autograd.grad(first, t)[0]
        min_curvature = min(min_curvature, float(second.detach() * energy_scale))
        n_paths += 1
    h_zero, pressure = model.minor_reference_terms()
    return {
        "certificate": (
            "With base_fun=zero, ICKAN's projected spline coefficients make every edge convex and "
            "non-decreasing. Its input is an affine positive scaling of directional |F a|^2, "
            "|cof(F)a|^2, J and J^2. Hence the D4-averaged structural term is convex in "
            "(F,cof(F),J); -r log(J) plus the positive quadratic term retains polyconvexity and "
            "makes energy diverge as J tends to zero."
        ),
        "pressure_r": float(pressure.detach()),
        "volumetric_quadratic_beta": float(model.volumetric_quadratic.detach()),
        "barrier_J_to_zero": {"J": compression_j, "energy_pa": diagonal_energy(compression_j)},
        "growth_J_to_infinity": {"J": expansion_j, "energy_pa": diagonal_energy(expansion_j)},
        "rank_one_curvature_audit": {"number_of_paths": n_paths, "minimum_second_derivative_pa": min_curvature},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=HERE / "checkpoints" / "ICKAN_D4_direct_best.pt"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    model, strain_scale, energy_scale, checkpoint = load_ickan_d4(args.checkpoint, torch.device("cpu"))
    with np.load(DATA_PATH) as data:
        strain = np.asarray(data["stage10_strain"], dtype=np.float32)
        reference_energy = np.asarray(data["stage10_energy"], dtype=float)
        reference_stress = np.asarray(data["stage10_stress"], dtype=float)
    energy, stress = predict(model, strain, strain_scale, energy_scale, args.batch_size)
    result = {
        "protocol": "Train on Stage-1 trajectories 1--10; evaluate once on untouched Stage-10.",
        "model": checkpoint["model"],
        "mode": checkpoint["model_configuration"]["mode"],
        "n_stage10": int(len(strain)),
        "ickan_d4_vs_direct_fom": {
            "energy_relative_l2": relative_l2(energy, reference_energy),
            "stress_relative_l2": relative_l2(stress, reference_stress),
            "stress_component_relative_l2": [
                relative_l2(stress[:, component], reference_stress[:, component]) for component in range(3)
            ],
        },
        "constraint_audit": reference_and_d4_audit(model, strain_scale, energy_scale),
        "polyconvex_audit": polyconvex_audit(model, strain_scale, energy_scale),
        "model_guarantee_note": checkpoint["model_configuration"]["guarantee_note"],
        "checkpoint_best_epoch": int(checkpoint["best_epoch"]),
    }
    prefix = args.output_prefix or f"ICKAN_D4_{checkpoint['model_configuration']['mode']}_stage10"
    result_dir = HERE / "results"
    result_dir.mkdir(exist_ok=True)
    (result_dir / f"{prefix}_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        result_dir / f"{prefix}_predictions.npz",
        strain=strain,
        energy_prediction=energy,
        stress_prediction=stress,
        energy_reference=reference_energy,
        stress_reference=reference_stress,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
