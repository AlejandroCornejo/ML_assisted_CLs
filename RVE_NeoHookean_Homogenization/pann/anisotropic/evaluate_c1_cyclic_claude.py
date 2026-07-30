#!/usr/bin/env python3
"""Direct numerical test of (C1) thermodynamic consistency: the cyclic work
integral oint S . dE around several small closed strain loops.

For any hyperelastic model (stress obtained by differentiating a scalar
potential W), this integral is exactly zero for every closed loop, by the
fundamental theorem of calculus -- W is single-valued, so the net change of
W around a loop that returns to its starting point is zero, and
oint dW = oint S.dE. This holds regardless of which admissible model
computes S; it is not specific to polyconvexity.

For a direct strain-to-stress regression model (no potential anywhere),
nothing forces this. This script quantifies exactly how far from zero it
is for the tier-1 baseline, and confirms it is at floating-point zero for
the three energy-based models (free, polyconvex ICNN, polyconvex ICKAN).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from anisotropic_pann_model import load_anisotropic_free, load_anisotropic_polyconvex
from anisotropic_pann_model_ickan_claude import load_anisotropic_polyconvex_ickan
from anisotropic_pann_model_regression_claude import load_anisotropic_regression


HERE = Path(__file__).resolve().parent
RESULT_DIR = HERE / "results"

# Each loop is a closed polygon in *physical* [E11, E22, gamma12] space,
# visited in order and returned to its start. Chosen inside the Stage-1
# training range (E11, E22 in [-0.1, 2.0], gamma12 in [-0.1, 0.1]) so this
# is not an extrapolation artifact.
LOOPS = {
    "small_normal_square": [
        (0.10, 0.10, 0.00), (0.30, 0.10, 0.00), (0.30, 0.30, 0.00), (0.10, 0.30, 0.00), (0.10, 0.10, 0.00),
    ],
    "large_normal_square": [
        (0.20, 0.20, 0.00), (0.80, 0.20, 0.00), (0.80, 0.80, 0.00), (0.20, 0.80, 0.00), (0.20, 0.20, 0.00),
    ],
    "shear_loop": [
        (0.10, 0.05, -0.05), (0.30, 0.05, -0.05), (0.30, 0.05, 0.05), (0.10, 0.05, 0.05), (0.10, 0.05, -0.05),
    ],
    "mixed_triangle": [
        (0.05, 0.05, 0.00), (0.40, 0.10, 0.03), (0.15, 0.35, -0.03), (0.05, 0.05, 0.00),
    ],
}
POINTS_PER_EDGE = 200


def densify_loop(vertices: list[tuple[float, float, float]], points_per_edge: int) -> np.ndarray:
    path = []
    for start, end in zip(vertices[:-1], vertices[1:]):
        start_arr, end_arr = np.asarray(start), np.asarray(end)
        for t in np.linspace(0.0, 1.0, points_per_edge, endpoint=False):
            path.append(start_arr + t * (end_arr - start_arr))
    path.append(np.asarray(vertices[-1]))
    return np.asarray(path, dtype=np.float64)


def cyclic_work(stress_path: np.ndarray, strain_path: np.ndarray) -> float:
    """Trapezoidal oint S . dE along a closed path (last point == first)."""

    d_strain = np.diff(strain_path, axis=0)
    stress_mid = 0.5 * (stress_path[:-1] + stress_path[1:])
    return float(np.sum(stress_mid * d_strain))


def stress_along_loop_regression(model, strain_scale: float, path: np.ndarray) -> np.ndarray:
    x = torch.as_tensor(path, dtype=torch.float32) / strain_scale
    with torch.no_grad():
        stress_normalised = model.stress(x)
    return stress_normalised.numpy()


def stress_along_loop_energy_based(model, strain_scale: float, path: np.ndarray, dtype: torch.dtype) -> np.ndarray:
    x = torch.as_tensor(path, dtype=dtype) / strain_scale
    x = x.requires_grad_(True)
    _, stress_normalised = model.energy_and_stress(x, create_graph=False)
    return stress_normalised.detach().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regression-checkpoint", default="PANN_anisotropic_regression_claude.pt")
    parser.add_argument("--free-checkpoint", default="PANN_anisotropic_free_compw000_claude.pt")
    parser.add_argument("--icnn-checkpoint", default="PANN_anisotropic_polyconvex_final_claude.pt")
    parser.add_argument("--ickan-checkpoint", default="PANN_anisotropic_polyconvex_ickan_final_claude.pt")
    parser.add_argument("--output-prefix", default="c1_cyclic_claude")
    args = parser.parse_args()

    reg_model, reg_strain_scale, reg_energy_scale, _ = load_anisotropic_regression(
        HERE / "checkpoints" / args.regression_checkpoint, torch.device("cpu")
    )
    free_model, free_strain_scale, free_energy_scale, _ = load_anisotropic_free(
        HERE / "checkpoints" / args.free_checkpoint, torch.device("cpu")
    )
    icnn_model, icnn_strain_scale, icnn_energy_scale, _ = load_anisotropic_polyconvex(
        HERE / "checkpoints" / args.icnn_checkpoint, torch.device("cpu")
    )
    ickan_model, ickan_strain_scale, ickan_energy_scale, _ = load_anisotropic_polyconvex_ickan(
        HERE / "checkpoints" / args.ickan_checkpoint, torch.device("cpu")
    )

    results: dict[str, dict[str, float]] = {
        "regression_baseline_tier1": {},
        "free_hyperelastic_tier2": {},
        "polyconvex_icnn_tier3a": {},
        "polyconvex_ickan_tier3b": {},
    }

    def one_loop_all_models(path: np.ndarray) -> dict[str, float]:
        reg_stress_n = stress_along_loop_regression(reg_model, reg_strain_scale, path)
        reg_stress = reg_stress_n * (reg_energy_scale / reg_strain_scale)

        free_stress_n = stress_along_loop_energy_based(free_model, free_strain_scale, path, torch.float32)
        free_stress = free_stress_n * (free_energy_scale / free_strain_scale)

        icnn_stress_n = stress_along_loop_energy_based(icnn_model, icnn_strain_scale, path, torch.float64)
        icnn_stress = icnn_stress_n * (icnn_energy_scale / icnn_strain_scale)

        ickan_stress_n = stress_along_loop_energy_based(ickan_model, ickan_strain_scale, path, torch.float64)
        ickan_stress = ickan_stress_n * (ickan_energy_scale / ickan_strain_scale)

        return {
            "regression_baseline_tier1": cyclic_work(reg_stress, path),
            "free_hyperelastic_tier2": cyclic_work(free_stress, path),
            "polyconvex_icnn_tier3a": cyclic_work(icnn_stress, path),
            "polyconvex_ickan_tier3b": cyclic_work(ickan_stress, path),
        }

    for loop_name, vertices in LOOPS.items():
        path = densify_loop(vertices, POINTS_PER_EDGE)
        per_model = one_loop_all_models(path)
        for key, value in per_model.items():
            results[key][loop_name] = value

    # Discretization-refinement check: the trapezoidal cyclic-work integral of a
    # genuine gradient field converges to exactly zero as the path is refined
    # (its residual is pure quadrature error, O(h^2)); the residual of a
    # non-conservative field (no potential anywhere, as in the tier-1 baseline)
    # converges to a fixed nonzero constant instead. This distinguishes a real
    # (C1) violation from a quadrature artifact without relying on the choice
    # of POINTS_PER_EDGE.
    refinement_loop = "mixed_triangle"
    refinement_path = LOOPS[refinement_loop]
    refinement_levels = [POINTS_PER_EDGE, 4 * POINTS_PER_EDGE, 16 * POINTS_PER_EDGE]
    refinement_study = {
        "loop": refinement_loop,
        "points_per_edge_levels": refinement_levels,
        "cyclic_work_per_model_per_level": {
            key: [one_loop_all_models(densify_loop(refinement_path, ppe))[key] for ppe in refinement_levels]
            for key in results
        },
    }

    # Contextualize magnitude: compare against a typical energy scale for this RVE.
    typical_energy_scale = icnn_energy_scale

    summary = {
        "protocol": (
            "oint S.dE (trapezoidal, physical units) around several closed strain "
            f"loops entirely inside the Stage-1 training range, {POINTS_PER_EDGE} points per edge."
        ),
        "points_per_edge": POINTS_PER_EDGE,
        "loops_definition_physical_strain": LOOPS,
        "cyclic_work_per_model_per_loop": results,
        "discretization_refinement_check": refinement_study,
        "typical_energy_scale_for_context": typical_energy_scale,
        "regression_worst_case_fraction_of_typical_energy": max(
            abs(v) for v in results["regression_baseline_tier1"].values()
        ) / typical_energy_scale,
    }
    print(json.dumps(summary, indent=2))
    RESULT_DIR.mkdir(exist_ok=True)
    (RESULT_DIR / f"{args.output_prefix}_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
