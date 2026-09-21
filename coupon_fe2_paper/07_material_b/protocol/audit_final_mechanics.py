"""Post-evaluation numerical QA; not training, selection or a stability proof.

Uses deterministic test-input indices and all ten path endpoints, never target
responses. Checks physical-unit energy/stress derivatives and sampled rank-one
curvature on the frozen final checkpoints.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from protocol import train_material_b as base
from protocol.evaluate_final_models import (LABELS, LOCK, RESULTS, load_model,
                                            verify_gate)
from protocol.prepare_design import digest

STEP = 1e-6  # In the physical engineering-strain coordinates.
TEST_INDICES = np.linspace(0, 511, 16, dtype=int)
PATH_ENDPOINT_INDICES = np.arange(39, 400, 40, dtype=int)
DIRECTIONS = np.asarray([[1., 0.], [0., 1.],
                         [2**-.5, 2**-.5], [2**-.5, -2**-.5]])


def energy_stress(model, e, scales):
    x = torch.as_tensor(e/scales["strain_scale"],
                        dtype=torch.float64).clone().detach().requires_grad_(True)
    w, s = model.energy_and_stress(x, create_graph=False)
    return ((w[:, 0]*scales["energy_scale"]).detach().numpy(),
            (s*scales["energy_scale"]/scales["strain_scale"]).detach().numpy())


def finite_difference(model, e, scales):
    _, stress = energy_stress(model, e, scales)
    energy_gradient = np.empty_like(stress)
    tangent = np.empty((len(e), 3, 3))
    for j in range(3):
        positive, negative = e.copy(), e.copy()
        positive[:, j] += STEP
        negative[:, j] -= STEP
        wp, sp = energy_stress(model, positive, scales)
        wm, sm = energy_stress(model, negative, scales)
        energy_gradient[:, j] = (wp-wm)/(2*STEP)
        tangent[:, :, j] = (sp-sm)/(2*STEP)
    return stress, energy_gradient, tangent


def analytic_tangent(model, e, scales):
    x = torch.as_tensor(e/scales["strain_scale"],
                        dtype=torch.float64).clone().detach().requires_grad_(True)
    _, s = model.energy_and_stress(x, create_graph=True)
    tangent = torch.stack([
        torch.autograd.grad(s[:, k].sum(), x, retain_graph=k < 2)[0]
        for k in range(3)], dim=1)
    return (tangent*scales["energy_scale"]/scales["strain_scale"]**2).detach().numpy()


def sampled_rank_one_curvature(model, e, scales):
    """D²_F W(F)[a⊗b,a⊗b] at F=sqrt(I+2E) for fixed unit a,b."""
    green = np.zeros((len(e), 2, 2))
    green[:, 0, 0] = e[:, 0]
    green[:, 1, 1] = e[:, 1]
    green[:, 0, 1] = green[:, 1, 0] = e[:, 2]/2
    c = np.eye(2)[None] + 2*green
    eigenvalues, eigenvectors = np.linalg.eigh(c)
    if np.min(eigenvalues) <= 0:
        raise ValueError("Sampled Green strain has no positive square root")
    f0 = (eigenvectors*np.sqrt(eigenvalues)[:, None, :]) @ eigenvectors.transpose(0, 2, 1)
    h = np.stack([np.outer(a, b) for a in DIRECTIONS for b in DIRECTIONS])
    f0 = np.repeat(f0, len(h), axis=0)
    h = np.tile(h, (len(e), 1, 1))
    f0 = torch.as_tensor(f0, dtype=torch.float64)
    h = torch.as_tensor(h, dtype=torch.float64)
    t = torch.zeros(len(f0), dtype=torch.float64, requires_grad=True)
    f = f0+t[:, None, None]*h
    c = f.transpose(1, 2) @ f
    x = torch.stack(((c[:, 0, 0]-1)/2, (c[:, 1, 1]-1)/2,
                     c[:, 0, 1]), dim=1)/scales["strain_scale"]
    energy = model.energy(x).sum()*scales["energy_scale"]
    first = torch.autograd.grad(energy, t, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), t)[0].detach().numpy()
    if not np.isfinite(second).all():
        raise FloatingPointError("Nonfinite sampled rank-one curvature")
    tolerance = 1e-8*max(1., float(np.max(np.abs(second))))
    return dict(minimum_Pa=float(np.min(second)),
                p01_Pa=float(np.percentile(second, 1)),
                negative_below_relative_tolerance=int(np.sum(second < -tolerance)),
                relative_tolerance_Pa=tolerance,
                sample_count=len(second))


def main():
    target = RESULTS / "mechanical_audit.json"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {target}")
    lock, scales = verify_gate()
    summary_path = RESULTS / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["status"] != "complete":
        raise ValueError("Final evaluation must be complete before mechanical audit")
    # Coordinate access only: never read S/W/D test or path target arrays here.
    with np.load(LABELS, allow_pickle=False) as data:
        e_test = np.asarray(data["E_test"])[TEST_INDICES]
        e_paths = np.asarray(data["E_paths"])[PATH_ENDPOINT_INDICES]
    e = np.vstack((e_test, e_paths))
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    rows = []
    for row in lock["entries"]:
        model = load_model(row)
        stress, gradient, numerical_tangent = finite_difference(model, e, scales)
        tangent = analytic_tangent(model, e, scales)
        stress_scale = max(1., float(np.max(np.abs(stress))))
        tangent_scale = max(1., float(np.max(np.abs(tangent))))
        rows.append(dict(slug=row["slug"], model=row["model"], seed=row["seed"],
            max_energy_gradient_stress_difference_Pa=float(np.max(np.abs(gradient-stress))),
            max_energy_gradient_stress_relative_to_max_stress=float(
                np.max(np.abs(gradient-stress))/stress_scale),
            max_tangent_fd_difference_Pa=float(np.max(np.abs(numerical_tangent-tangent))),
            max_tangent_fd_relative_to_max_tangent=float(
                np.max(np.abs(numerical_tangent-tangent))/tangent_scale),
            max_tangent_asymmetry_Pa=float(np.max(np.abs(tangent-
                tangent.transpose(0, 2, 1)))),
            rank_one=sampled_rank_one_curvature(model, e, scales)))
        print(json.dumps(dict(slug=row["slug"],
            tangent_fd_relative=rows[-1]["max_tangent_fd_relative_to_max_tangent"],
            rank_one_minimum_Pa=rows[-1]["rank_one"]["minimum_Pa"])), flush=True)
    result = dict(status="complete", scope="Post-evaluation numerical QA only; not a proof",
        model_lock_sha256=digest(LOCK), final_summary_sha256=digest(summary_path),
        script_sha256=digest(Path(__file__)),
        test_input_indices=TEST_INDICES.tolist(),
        path_endpoint_indices=PATH_ENDPOINT_INDICES.tolist(),
        rank_one_a_and_b=DIRECTIONS.tolist(),
        strain_fd_step=STEP, target_response_arrays_read=False, rows=rows)
    base._atomic_json(target, result)
    print(target)


if __name__ == "__main__":
    main()
