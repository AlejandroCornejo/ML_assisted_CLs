#!/usr/bin/env python3
"""Evaluate and audit the polyconvex D4 PANN on the untouched Stage-10 path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from polyconvex_d4_model import PolyconvexD4Energy, load_polyconvex_pann


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "alltraj_stage10_direct_energy.npz"
CHECKPOINT_PATH = HERE / "checkpoints" / "PANN_D4_polyconvex_best.pt"
RESULT_PATH = HERE / "results" / "PANN_D4_polyconvex_stage10_metrics.json"
PREDICTION_PATH = HERE / "results" / "PANN_D4_polyconvex_stage10_predictions.npz"


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - reference) / max(np.linalg.norm(reference), 1.0e-30))


def predict(
    model: PolyconvexD4Energy,
    strain: np.ndarray,
    *,
    strain_scale: float,
    energy_scale: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    energies, stresses = [], []
    with torch.enable_grad():
        for start in range(0, len(strain), batch_size):
            raw = torch.as_tensor(strain[start : start + batch_size], dtype=torch.float64)
            normalised = (raw / strain_scale).detach().clone().requires_grad_(True)
            energy, stress = model.energy_and_stress(normalised, create_graph=False)
            energies.append((energy.detach().cpu().numpy() * energy_scale).reshape(-1))
            stresses.append(stress.detach().cpu().numpy() * (energy_scale / strain_scale))
    return np.concatenate(energies), np.concatenate(stresses)


def d4_actions(strain: torch.Tensor) -> torch.Tensor:
    exx, eyy, gamma = strain[:, 0], strain[:, 1], strain[:, 2]
    return torch.stack(
        (
            torch.stack((exx, eyy, gamma), dim=1),
            torch.stack((eyy, exx, -gamma), dim=1),
            torch.stack((exx, eyy, -gamma), dim=1),
            torch.stack((eyy, exx, gamma), dim=1),
        ),
        dim=1,
    )


def guarantee_audit(model: PolyconvexD4Energy, *, strain_scale: float, energy_scale: float) -> dict:
    """Numerically audit identities implied by the analytic construction."""

    # Reference state and D4 covariance of the energy gradient.
    reference = torch.zeros((1, 3), dtype=torch.float64, requires_grad=True)
    reference_energy, reference_stress = model.energy_and_stress(reference, create_graph=False)
    probe = torch.tensor(
        [[0.12, -0.04, 0.10], [-0.03, 0.16, -0.08], [0.08, 0.05, 0.18]],
        dtype=torch.float64,
    ) / strain_scale
    orbit = d4_actions(probe)
    n_samples, n_actions, _ = orbit.shape
    orbit_flat = orbit.reshape(n_samples * n_actions, 3).detach().clone().requires_grad_(True)
    orbit_energy, orbit_stress = model.energy_and_stress(orbit_flat, create_graph=False)
    orbit_energy = orbit_energy.reshape(n_samples, n_actions)
    orbit_stress = orbit_stress.reshape(n_samples, n_actions, 3)
    expected_stress = d4_actions(probe).detach()
    # d4_actions acts linearly and orthogonally, so it is also the correct
    # covariant action on the stress vector in engineering-shear coordinates.
    base = orbit_stress[:, 0, :]
    transformed_base = d4_actions(base)

    # Values along F=diag(J,1).  The energy is evaluated from E, but J is the
    # physical determinant of this deformation gradient.
    def energy_at_diagonal_j(j_values: list[float]) -> list[float]:
        strains = torch.tensor(
            [[0.5 * (j * j - 1.0), 0.0, 0.0] for j in j_values],
            dtype=torch.float64,
        ) / strain_scale
        return (model.energy(strains).detach().cpu().numpy().reshape(-1) * energy_scale).tolist()

    compression_j = [1.0, 1.0e-1, 1.0e-3, 1.0e-6]
    expansion_j = [1.0, 2.0, 10.0, 100.0]
    compression_energy = energy_at_diagonal_j(compression_j)
    expansion_energy = energy_at_diagonal_j(expansion_j)

    # A broad but finite sampling of valid C verifies the non-negativity that
    # also follows analytically from the convex construction and AM--GM.
    generator = torch.Generator().manual_seed(20260803)
    l11 = torch.exp(torch.empty(4000, dtype=torch.float64).uniform_(-2.0, 2.0, generator=generator))
    l22 = torch.exp(torch.empty(4000, dtype=torch.float64).uniform_(-2.0, 2.0, generator=generator))
    l21 = torch.empty(4000, dtype=torch.float64).uniform_(-2.0, 2.0, generator=generator)
    # C=L L^T, with L lower triangular and positive diagonal.
    c11 = l11.square()
    c12 = l11 * l21
    c22 = l21.square() + l22.square()
    valid_strain = torch.stack((0.5 * (c11 - 1.0), 0.5 * (c22 - 1.0), c12), dim=1) / strain_scale
    sampled_energy = model.energy(valid_strain).detach().cpu().numpy().reshape(-1) * energy_scale

    # Polyconvexity implies rank-one convexity.  This numerical audit is not
    # the proof (the architectural proof is above), but it actively searches
    # for a failure along 128 random F+t(a outer b) paths on J>0.
    rank_one_generator = torch.Generator().manual_seed(20260804)
    minimum_rank_one_curvature = float("inf")
    valid_rank_one_paths = 0
    for _ in range(128):
        f = torch.tensor(
            [[float(torch.exp(torch.empty((), dtype=torch.float64).uniform_(-1.2, 1.2, generator=rank_one_generator))),
              float(torch.empty((), dtype=torch.float64).uniform_(-0.6, 0.6, generator=rank_one_generator))],
             [0.0,
              float(torch.exp(torch.empty((), dtype=torch.float64).uniform_(-1.2, 1.2, generator=rank_one_generator))) ]],
            dtype=torch.float64,
        )
        a = torch.randn(2, dtype=torch.float64, generator=rank_one_generator)
        b = torch.randn(2, dtype=torch.float64, generator=rank_one_generator)
        a = a / torch.linalg.vector_norm(a)
        b = b / torch.linalg.vector_norm(b)
        t = torch.zeros((), dtype=torch.float64, requires_grad=True)
        deformed = f + t * torch.outer(a, b)
        c_tensor = deformed.T @ deformed
        physical_e = torch.stack((0.5 * (c_tensor[0, 0] - 1.0), 0.5 * (c_tensor[1, 1] - 1.0), c_tensor[0, 1]))
        value = model.energy((physical_e / strain_scale).reshape(1, 3)).sum()
        first = torch.autograd.grad(value, t, create_graph=True)[0]
        second = torch.autograd.grad(first, t)[0]
        minimum_rank_one_curvature = min(minimum_rank_one_curvature, float(second.detach() * energy_scale))
        valid_rank_one_paths += 1

    # This is intentionally a separate diagnostic: positive semidefiniteness
    # of dS/dE is *not* equivalent to polyconvexity.  We report it rather than
    # silently claiming that the seminar-slide condition follows from PConv.
    minimum_strain_hessian_eigenvalue = float("inf")
    for state in valid_strain[:64]:
        state = state.detach().clone().requires_grad_(True)
        hessian = torch.autograd.functional.hessian(lambda x: model.energy(x.reshape(1, 3)).sum(), state)
        eigenvalue = torch.linalg.eigvalsh(0.5 * (hessian + hessian.T))[0]
        minimum_strain_hessian_eigenvalue = min(
            minimum_strain_hessian_eigenvalue,
            float(eigenvalue.detach() * energy_scale / (strain_scale * strain_scale)),
        )

    return {
        "reference": {
            "energy": float(reference_energy.item() * energy_scale),
            "stress_norm": float(torch.linalg.vector_norm(reference_stress).item() * energy_scale / strain_scale),
        },
        "d4": {
            "maximum_absolute_energy_orbit_error": float(torch.max(torch.abs(orbit_energy - orbit_energy[:, :1])).item() * energy_scale),
            "maximum_absolute_stress_covariance_error": float(torch.max(torch.abs(orbit_stress - transformed_base)).item() * energy_scale / strain_scale),
        },
        "analytic_coefficients": model.certificate_summary(),
        "volumetric_barrier_J_to_zero": {
            "J": compression_j,
            "energy": compression_energy,
        },
        "volumetric_growth_J_to_infinity": {
            "J": expansion_j,
            "energy": expansion_energy,
        },
        "nonnegative_energy_sampling": {
            "number_of_valid_random_states": int(len(sampled_energy)),
            "minimum_energy": float(np.min(sampled_energy)),
        },
        "rank_one_curvature_audit": {
            "number_of_F_paths": valid_rank_one_paths,
            "minimum_second_derivative": minimum_rank_one_curvature,
        },
        "strain_hessian_diagnostic": {
            "minimum_eigenvalue_of_dS_dE": minimum_strain_hessian_eigenvalue,
            "interpretation": (
                "Reported only as a diagnostic. Polyconvexity is a condition in F and does not, by itself, require dS/dE to be positive semidefinite."
            ),
        },
        "certificate_statement": (
            "Each directional term is convex and non-decreasing in |F a|^2 or "
            "|cof(F) a|^2; the volumetric term is convex in J. Hence the energy "
            "is polyconvex in (F, cof(F), J) on J>0."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    model, strain_scale, energy_scale, checkpoint = load_polyconvex_pann(CHECKPOINT_PATH, torch.device("cpu"))
    with np.load(DATA_PATH) as data:
        strain = np.asarray(data["stage10_strain"], dtype=np.float64)
        direct_energy = np.asarray(data["stage10_energy"], dtype=np.float64)
        direct_stress = np.asarray(data["stage10_stress"], dtype=np.float64)

    polyconvex_energy, polyconvex_stress = predict(
        model, strain, strain_scale=strain_scale, energy_scale=energy_scale, batch_size=args.batch_size,
    )
    hprom_path = HERE / "results" / "stage10_hprom_direct_reference.json"
    selected_pann_path = HERE / "results" / "stage10_metrics.json"
    result = {
        "protocol": "Train on Stage-1 trajectories 1--10; evaluate once on untouched Stage-10.",
        "model": "Polyconvex D4 directional-minors PANN.",
        "n_stage10": int(len(strain)),
        "polyconvex_pann_vs_direct_fom": {
            "energy_relative_l2": relative_l2(polyconvex_energy, direct_energy),
            "stress_relative_l2": relative_l2(polyconvex_stress, direct_stress),
            "stress_component_relative_l2": [
                relative_l2(polyconvex_stress[:, component], direct_stress[:, component])
                for component in range(3)
            ],
        },
        "analytic_and_numerical_guarantee_audit": guarantee_audit(
            model, strain_scale=strain_scale, energy_scale=energy_scale,
        ),
        "normalisation": {"strain_scale": strain_scale, "energy_scale": energy_scale},
        "checkpoint_best_epoch": int(checkpoint["best_epoch"]),
        "reference_comparisons": {
            "selected_unconstrained_pann_d4": json.loads(selected_pann_path.read_text(encoding="utf-8")),
            "hprom_ann_direct_reconstruction": json.loads(hprom_path.read_text(encoding="utf-8")),
        },
    }
    print(json.dumps(result, indent=2))
    if not args.no_write:
        RESULT_PATH.parent.mkdir(exist_ok=True)
        RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        np.savez_compressed(
            PREDICTION_PATH,
            stage10_strain=strain,
            direct_energy=direct_energy,
            direct_stress=direct_stress,
            polyconvex_energy=polyconvex_energy,
            polyconvex_stress=polyconvex_stress,
        )


if __name__ == "__main__":
    main()
