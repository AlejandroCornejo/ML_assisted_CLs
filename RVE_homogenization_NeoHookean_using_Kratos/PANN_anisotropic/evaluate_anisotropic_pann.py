#!/usr/bin/env python3
"""Evaluate the anisotropic PANNs on the untouched Stage--10 trajectory.

This is the only script in this directory that opens Stage--10.  Besides the
energy/stress error, it audits the identities that are architectural facts:
objectivity, reference state, coordinate-covariant material features, and for
the second model the polyconvex construction and volumetric barrier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from anisotropic_pann_model import (
    AnisotropicPolyconvexEnergy,
    c_to_strain,
    load_anisotropic_free,
    load_anisotropic_polyconvex,
    load_metric_preconditioned_free,
    material_c_features,
)


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "PANN_D4" / "data" / "alltraj_stage10_direct_energy.npz"
FREE_CHECKPOINT = HERE / "checkpoints" / "PANN_anisotropic_free.pt"
METRIC_CHECKPOINT = HERE / "checkpoints" / "PANN_anisotropic_metric.pt"
POLYCONVEX_CHECKPOINT = HERE / "checkpoints" / "PANN_anisotropic_polyconvex.pt"
RESULT_PATH = HERE / "results" / "anisotropic_stage10_metrics.json"
PREDICTION_PATH = HERE / "results" / "anisotropic_stage10_predictions.npz"


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - reference) / max(np.linalg.norm(reference), 1.0e-30))


def predict(model, strain: np.ndarray, *, strain_scale: float, energy_scale: float, dtype: torch.dtype, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    energies, stresses = [], []
    with torch.enable_grad():
        for start in range(0, len(strain), batch_size):
            raw = torch.as_tensor(strain[start:start + batch_size], dtype=dtype)
            normalised = (raw / strain_scale).detach().clone().requires_grad_(True)
            energy, stress = model.energy_and_stress(normalised, create_graph=False)
            energies.append((energy.detach().cpu().numpy() * energy_scale).reshape(-1))
            stresses.append(stress.detach().cpu().numpy() * (energy_scale / strain_scale))
    return np.concatenate(energies), np.concatenate(stresses)


def energy_from_f(model, f: torch.Tensor, *, strain_scale: float) -> torch.Tensor:
    """Evaluate an objective energy along a deformation-gradient path."""

    c = f.transpose(-1, -2) @ f
    physical_strain = c_to_strain(c.reshape(-1, 2, 2))
    return model.energy(physical_strain / strain_scale)


def rotation(theta: float, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(
        ((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))), dtype=dtype
    )


def common_audit(model, *, strain_scale: float, energy_scale: float, dtype: torch.dtype, representation: str) -> dict:
    """Numerical checks of statements that follow from the representation."""

    reference = torch.zeros((1, 3), dtype=dtype, requires_grad=True)
    reference_energy, reference_stress = model.energy_and_stress(reference, create_graph=False)
    f = torch.tensor(
        (((1.25, 0.18), (-0.07, 0.86)), ((0.92, -0.11), (0.14, 1.17))), dtype=dtype
    )
    q = rotation(0.73, dtype)
    f_rotated = q.unsqueeze(0) @ f
    original_energy = energy_from_f(model, f, strain_scale=strain_scale)
    rotated_energy = energy_from_f(model, f_rotated, strain_scale=strain_scale)
    original_e = c_to_strain(f.transpose(-1, -2) @ f) / strain_scale
    rotated_e = c_to_strain(f_rotated.transpose(-1, -2) @ f_rotated) / strain_scale
    _, original_s = model.energy_and_stress(original_e.detach().clone().requires_grad_(True), create_graph=False)
    _, rotated_s = model.energy_and_stress(rotated_e.detach().clone().requires_grad_(True), create_graph=False)

    # Coordinate covariance of the scalar material features.  No D4 transform
    # is used: this is a generic relabelling of the material coordinate basis.
    c = f.transpose(-1, -2) @ f
    r = rotation(-0.41, dtype)
    c_prime = r.T.unsqueeze(0) @ c @ r.unsqueeze(0)
    a = torch.tensor((1.0, 0.0), dtype=dtype)
    b = torch.tensor((0.0, 1.0), dtype=dtype)
    original_features = material_c_features(c, a, b)
    transformed_features = material_c_features(c_prime, r.T @ a, r.T @ b)

    # This number is diagnostic only.  A nonzero value proves that the model
    # did not receive an architectural D4 group average; a near-zero value can
    # arise naturally if the FOM happens to be close to square symmetric.
    e_probe = torch.tensor(((0.18, -0.04, 0.10), (0.04, 0.22, -0.13)), dtype=dtype)
    d4_rotated = torch.stack((e_probe[:, 1], e_probe[:, 0], -e_probe[:, 2]), dim=1)
    e0 = model.energy(e_probe / strain_scale)
    e90 = model.energy(d4_rotated / strain_scale)
    return {
        "reference": {
            "energy": float(reference_energy.detach().item() * energy_scale),
            "stress_norm": float(torch.linalg.vector_norm(reference_stress).detach().item() * energy_scale / strain_scale),
        },
        "spatial_objectivity": {
            "maximum_absolute_energy_error_under_F_to_QF": float(torch.max(torch.abs(original_energy - rotated_energy)).detach() * energy_scale),
            "maximum_absolute_second_piola_error_under_F_to_QF": float(torch.max(torch.abs(original_s - rotated_s)).detach() * energy_scale / strain_scale),
        },
        "material_coordinate_covariance": {
            "maximum_absolute_joint_invariant_error": float(torch.max(torch.abs(original_features - transformed_features)).detach()),
            "statement": (
                "For the C-based PANN, C and material directions are rotated together; a.C.a, b.C.b, a.C.b and J are unchanged. "
                "For the polyconvex PANN, the same relabelling applies to every structural direction. "
                "For PANN-T, the three-by-three tangent map T must be transformed with the material strain coordinates."
            ),
            "model_representation": representation,
        },
        "d4_not_imposed": {
            "maximum_absolute_energy_difference_for_one_90_degree_material_rotation": float(torch.max(torch.abs(e0 - e90)).detach() * energy_scale),
            "statement": "This is not an error test: anisotropic architecture does not require it to vanish.",
        },
    }


def polyconvex_audit(model: AnisotropicPolyconvexEnergy, *, strain_scale: float, energy_scale: float) -> dict:
    dtype = torch.float64
    common = common_audit(
        model, strain_scale=strain_scale, energy_scale=energy_scale, dtype=dtype,
        representation="polyconvex directional-minor invariants",
    )

    def diagonal_j_energy(j_values: list[float]) -> list[float]:
        # F=diag(J,1), hence E11=(J^2-1)/2 and det(F)=J.
        states = torch.tensor(
            [(0.5 * (j * j - 1.0), 0.0, 0.0) for j in j_values], dtype=dtype
        )
        return (model.energy(states / strain_scale).detach().cpu().numpy().reshape(-1) * energy_scale).tolist()

    compression_j = [1.0, 1.0e-1, 1.0e-3, 1.0e-6]
    expansion_j = [1.0, 2.0, 10.0, 100.0]

    # PConv => rank-one convex.  This is a numerical falsification attempt,
    # not the proof: the proof is the model construction documented in source.
    generator = torch.Generator().manual_seed(20260813)
    minimum_rank_one_curvature = float("inf")
    valid_paths = 0
    for _ in range(64):
        f = torch.tensor(
            ((float(torch.exp(torch.empty((), dtype=dtype).uniform_(-0.7, 0.7, generator=generator))),
              float(torch.empty((), dtype=dtype).uniform_(-0.25, 0.25, generator=generator))),
             (0.0, float(torch.exp(torch.empty((), dtype=dtype).uniform_(-0.7, 0.7, generator=generator)))),),
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
        minimum_rank_one_curvature = min(minimum_rank_one_curvature, float(second.detach() * energy_scale))
        valid_paths += 1

    # This deliberately does not claim D_E positive definite.  The FOM audit
    # already showed that it is not a valid global property of this RVE.
    hessian_minimum = float("inf")
    probes = torch.tensor(
        ((0.0, 0.0, 0.0), (0.15, 0.05, 0.04), (0.5, 0.3, -0.06), (0.9, 0.7, 0.03)), dtype=dtype
    )
    for state in probes:
        x = (state / strain_scale).detach().clone().requires_grad_(True)
        hessian = torch.autograd.functional.hessian(lambda z: model.energy(z.reshape(1, 3)).sum(), x)
        hessian_minimum = min(hessian_minimum, float(torch.linalg.eigvalsh(0.5 * (hessian + hessian.T))[0].detach() * energy_scale / (strain_scale * strain_scale)))

    # This is not a global non-negativity proof.  It is a deterministic broad
    # sampling audit, recorded because W(I)=0 makes negative sampled energies
    # physically suspicious even when polyconvexity itself is satisfied.
    sampler = torch.Generator().manual_seed(20260818)
    sampled_strain = torch.empty((20000, 3), dtype=dtype)
    sampled_strain[:, :2].uniform_(-0.4, 1.5, generator=sampler)
    sampled_strain[:, 2].uniform_(-0.7, 0.7, generator=sampler)
    c11 = 1.0 + 2.0 * sampled_strain[:, 0]
    c22 = 1.0 + 2.0 * sampled_strain[:, 1]
    sampled_strain = sampled_strain[c11 * c22 - sampled_strain[:, 2].square() > 1.0e-6]
    sampled_energy = torch.cat(
        [model.energy((chunk / strain_scale)).detach().reshape(-1) for chunk in sampled_strain.split(2048)]
    )
    minimum_sampled_energy = float(torch.min(sampled_energy).detach() * energy_scale)
    negative_fraction = float(torch.mean((sampled_energy < 0.0).to(dtype)).detach())

    return common | {
        "analytic_polyconvex_certificate": model.certificate_summary(),
        "certificate_statement": (
            "tr(C), each balanced quartic/sixth-power invariant Q^F_{k,p}=sum_i w_ki|F d_ki|^(2p), and each Q^H_{k,p}=sum_i w_ki|cof(F)d_ki|^(2p), p=2,3, are convex in their minor arguments. "
            "The positive ICNN is convex and non-decreasing in those features; -r log(J) and (J-1)^2 are convex in J. "
            "Therefore W(F)=G(F,cof(F),J) with convex G on J>0."
        ),
        "volumetric_barrier_J_to_zero": {"J": compression_j, "energy": diagonal_j_energy(compression_j)},
        "volumetric_growth_J_to_infinity": {"J": expansion_j, "energy": diagonal_j_energy(expansion_j)},
        "rank_one_curvature_audit": {"number_of_paths": valid_paths, "minimum_second_derivative": minimum_rank_one_curvature},
        "strain_hessian_diagnostic": {
            "minimum_eigenvalue_of_dS_dE": hessian_minimum,
            "interpretation": "Diagnostic only: polyconvexity does not imply dS/dE positive semidefinite.",
        },
        "broad_energy_sampling_audit": {
            "number_of_positive_definite_samples": int(len(sampled_strain)),
            "minimum_energy": minimum_sampled_energy,
            "negative_energy_fraction": negative_fraction,
            "interpretation": "Sampling diagnostic only; it is not a global non-negativity theorem.",
        },
    }


def evaluate_one(kind: str, model, strain: np.ndarray, direct_energy: np.ndarray, direct_stress: np.ndarray, *, strain_scale: float, energy_scale: float) -> tuple[dict, np.ndarray, np.ndarray]:
    dtype = torch.float64 if kind == "polyconvex" else torch.float32
    predicted_energy, predicted_stress = predict(
        model, strain, strain_scale=strain_scale, energy_scale=energy_scale, dtype=dtype, batch_size=512
    )
    result = {
        "energy_relative_l2": relative_l2(predicted_energy, direct_energy),
        "stress_relative_l2": relative_l2(predicted_stress, direct_stress),
        "stress_component_relative_l2": [
            relative_l2(predicted_stress[:, component], direct_stress[:, component]) for component in range(3)
        ],
        "guarantee_audit": (
            common_audit(
                model, strain_scale=strain_scale, energy_scale=energy_scale, dtype=dtype,
                representation="C material-component MLP" if kind == "free" else "locally-isotropised tangent coordinates T e",
            )
            if kind == "free"
            else (
                common_audit(
                    model, strain_scale=strain_scale, energy_scale=energy_scale, dtype=dtype,
                    representation="locally-isotropised tangent coordinates T e",
                ) if kind == "metric" else polyconvex_audit(model, strain_scale=strain_scale, energy_scale=energy_scale)
            )
        ),
    }
    return result, predicted_energy, predicted_stress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("free", "metric", "polyconvex", "all"), default="all")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    with np.load(DATA_PATH) as data:
        strain = np.asarray(data["stage10_strain"], dtype=np.float64)
        direct_energy = np.asarray(data["stage10_energy"], dtype=np.float64)
        direct_stress = np.asarray(data["stage10_stress"], dtype=np.float64)

    output: dict = {
        "protocol": "Train on Stage-1 trajectories 1--10; evaluate once on untouched Stage-10.",
        "material_model": "2D anisotropic PANN; no D4 group average.",
        "n_stage10": int(len(strain)),
    }
    predictions: dict[str, np.ndarray] = {
        "stage10_strain": strain,
        "direct_energy": direct_energy,
        "direct_stress": direct_stress,
    }
    if args.kind in ("free", "all"):
        model, strain_scale, energy_scale, checkpoint = load_anisotropic_free(FREE_CHECKPOINT, torch.device("cpu"))
        result, energy, stress = evaluate_one("free", model, strain, direct_energy, direct_stress, strain_scale=strain_scale, energy_scale=energy_scale)
        output["free_anisotropic_c_pann"] = result | {"checkpoint_best_epoch": int(checkpoint["best_epoch"])}
        predictions["free_energy"], predictions["free_stress"] = energy, stress
    if args.kind in ("metric", "all"):
        model, strain_scale, energy_scale, checkpoint = load_metric_preconditioned_free(METRIC_CHECKPOINT, torch.device("cpu"))
        result, energy, stress = evaluate_one("metric", model, strain, direct_energy, direct_stress, strain_scale=strain_scale, energy_scale=energy_scale)
        output["metric_preconditioned_anisotropic_pann"] = result | {"checkpoint_best_epoch": int(checkpoint["best_epoch"])}
        predictions["metric_energy"], predictions["metric_stress"] = energy, stress
    if args.kind in ("polyconvex", "all"):
        model, strain_scale, energy_scale, checkpoint = load_anisotropic_polyconvex(POLYCONVEX_CHECKPOINT, torch.device("cpu"))
        result, energy, stress = evaluate_one("polyconvex", model, strain, direct_energy, direct_stress, strain_scale=strain_scale, energy_scale=energy_scale)
        output["polyconvex_anisotropic_pann"] = result | {"checkpoint_best_epoch": int(checkpoint["best_epoch"])}
        predictions["polyconvex_energy"], predictions["polyconvex_stress"] = energy, stress

    hprom_reference = HERE.parent / "PANN_D4" / "results" / "stage10_hprom_direct_reference.json"
    if hprom_reference.exists():
        output["compatible_direct_hprom_ann_reference"] = json.loads(hprom_reference.read_text(encoding="utf-8"))
    print(json.dumps(output, indent=2))
    if not args.no_write:
        RESULT_PATH.parent.mkdir(exist_ok=True)
        RESULT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        np.savez_compressed(PREDICTION_PATH, **predictions)


if __name__ == "__main__":
    main()
