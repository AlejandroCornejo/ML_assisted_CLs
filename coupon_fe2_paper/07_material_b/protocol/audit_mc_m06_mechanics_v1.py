"""Mechanical audit of the twelve MC-RVE m=6 fits and the same-rule Unconstrained energy.

Implementation checks after the one-time independent evaluation; no fitting,
selection, or target-response access.  Only strain coordinates (16 test inputs,
the ten path endpoints) and the reference tangent D_reference, which is also a
training quantity, are read from the labels.

Every model receives the same checks unless stated otherwise:
  reference state  W(0), S(0), and the relative misfit of D(0) against D_reference;
  derivatives      centred differences of W against S and of S against D, the
                   symmetry of D, and torch gradcheck/gradgradcheck;
  closed cycles    Gauss--Legendre stress work around a small and a large rectangle
                   in (E11, E22) at fixed shear, with a quadrature-order table and
                   the edgewise comparison of stress work with energy differences;
  nonnegativity    the saved-parameter sufficient bound of Appendix C and a
                   logarithmic scan of its lower-bound function g(J), constrained
                   models only, and the broad admissible cloud of the SC-RVE audit;
  rank-one sample  exact F-space curvature at the audit states.
The derivative and rank-one routines are those of audit_final_mechanics.py; the
certificate and the broad cloud are those of 06_pann/audit_flexible.py.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from protocol import train_material_b as base
from protocol.audit_final_mechanics import (PATH_ENDPOINT_INDICES, STEP, TEST_INDICES,
                                            analytic_tangent, energy_stress,
                                            finite_difference, sampled_rank_one_curvature)
from protocol.evaluate_feature_count_m06_v1 import (FEATURES, LABELS, OUTPUT as EVALUATION,
                                                    TRAINING as M06_TRAINING, _load_model)
from protocol.prepare_design import digest
from protocol.select_features import BASE
from protocol.training_setup import AnisotropicFreeEnergy
from audit_flexible import energy_lower_bound_certificate


CAPACITY = BASE / "results/capacity_2x2_v1"
OUTPUT = BASE / "results/feature_count_analysis_v1/m06_mechanics_audit_v1"
FREE_SEEDS = (16, 29, 47)
# Rectangles in (E11, E22) at fixed engineering shear, inside the MC-RVE box
# E11, E22 in [-0.04, 0.20], 2E12 in [-0.08, 0.08].
CYCLES = {"small": dict(centre=[0.08, 0.08, 0.0], half_width=1.0e-3),
          "large": dict(centre=[0.08, 0.08, 0.04], half_width=0.11)}
CYCLE_ORDER = 64
CONVERGENCE_ORDERS = (8, 16, 32, 64, 128)


def rectangle(centre, half_width, order):
    """Gauss points, weighted increments, and edge labels of one counter-clockwise loop."""
    centre = np.asarray(centre, dtype=np.float64)
    corners = [centre + [sx*half_width, sy*half_width, 0.]
               for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))]
    xi, wi = np.polynomial.legendre.leggauss(order)
    points, increments = [], []
    for start, end in zip(corners[:-1], corners[1:]):
        points.append(.5*(start+end)+.5*xi[:, None]*(end-start))
        increments.append(.5*wi[:, None]*(end-start))
    return np.concatenate(points), np.concatenate(increments), np.asarray(corners)


def cycle_record(model, scales, centre, half_width):
    convergence = {}
    for order in CONVERGENCE_ORDERS:
        points, increments, _ = rectangle(centre, half_width, order)
        convergence[str(order)] = float(np.einsum("ni,ni->", energy_stress(model, points, scales)[1],
                                                  increments))
    points, increments, corners = rectangle(centre, half_width, CYCLE_ORDER)
    stress = energy_stress(model, points, scales)[1]
    pointwise = np.einsum("ni,ni->n", stress, increments)
    scale = float(np.abs(pointwise).sum())
    corner_energy = energy_stress(model, corners, scales)[0]
    edge_work = pointwise.reshape(4, CYCLE_ORDER).sum(axis=1)
    edge_mismatch = np.abs(edge_work-np.diff(corner_energy))
    return dict(work_J_per_m3=float(pointwise.sum()), absolute_work_scale_J_per_m3=scale,
                work_relative_to_scale=float(abs(pointwise.sum())/scale),
                max_edge_work_minus_energy_difference_J_per_m3=float(edge_mismatch.max()),
                max_edge_mismatch_relative_to_scale=float(edge_mismatch.max()/scale),
                work_by_quadrature_order_J_per_m3=convergence)


def broad_energy_cloud(model, scales):
    """The 6000-state admissible cloud of 06_pann/audit_flexible.py, same generator seed."""
    rng = np.random.default_rng(101)
    angle = rng.uniform(-np.pi, np.pi, 6000)
    stretch = np.exp(rng.uniform(np.log(.25), np.log(3.), (6000, 2)))
    co, si = np.cos(angle), np.sin(angle)
    c11 = stretch[:, 0]**2*co**2+stretch[:, 1]**2*si**2
    c22 = stretch[:, 0]**2*si**2+stretch[:, 1]**2*co**2
    c12 = (stretch[:, 0]**2-stretch[:, 1]**2)*co*si
    e = np.column_stack(((c11-1)/2, (c22-1)/2, c12))
    sample = model.energy(torch.as_tensor(e/scales["strain_scale"],
                                          dtype=torch.float64)).detach().numpy()[:, 0]
    return dict(count=len(e), stretch_range=[.25, 3.], finite=bool(np.isfinite(sample).all()),
                minimum_J_per_m3=float(sample.min())*scales["energy_scale"],
                negative_count=int((sample < -1e-8).sum()))


def lower_bound_scan(model):
    """Diagnostic of Eq. (C.2) on a grid, not a certificate; terms as in audit_flexible.py."""
    x = torch.zeros((1, 3), dtype=torch.float64)
    z, _ = model.structural_features(x)
    z = (z/model.feature_scale).detach().requires_grad_(True)
    h = (torch.autograd.grad(model.base_icnn(z).sum(), z)[0][0]/model.feature_scale).detach().numpy()
    _, p, q, b, c = model.effective_specs().detach().numpy().T
    k = 1/p+1/q
    eta = (2-b/p-c/q)/k
    alpha = float(model.barrier_coefficient.detach())+float(model.volumetric_floor)
    beta = float(model.quadratic_coefficient.detach())
    j = np.geomspace(1e-6, 1e8, 20001)[:, None]
    g = ((h*k*(j**eta-1-eta*(j-1))).sum(axis=1)+alpha*(j[:, 0]-1-np.log(j[:, 0]))
         + .5*beta*(j[:, 0]-1)**2)
    curvature = alpha+beta*j[:, 0]**2+(h*k*eta*(eta-1)*j**eta).sum(axis=1)
    concave = j[curvature < 0, 0]
    return dict(J_range=[1e-6, 1e8], grid_points=len(j), feature_eta=eta.tolist(),
                feature_hk=(h*k).tolist(), minimum_g=float(g.min()),
                argmin_J=float(j[np.argmin(g), 0]), negative_g_points=int((g < -1e-12).sum()),
                large_J_linear_coefficient=float(alpha-(h*k*eta).sum()),
                negative_curvature_J_interval=([float(concave.min()), float(concave.max())]
                                               if len(concave) else None))


def audit(model, scales, e, d_reference, constrained):
    stress, gradient, numerical_tangent = finite_difference(model, e, scales)
    tangent = analytic_tangent(model, e, scales)
    stress_scale = max(1., float(np.max(np.abs(stress))))
    tangent_scale = max(1., float(np.max(np.abs(tangent))))
    w0, s0 = energy_stress(model, np.zeros((1, 3)), scales)
    d0 = analytic_tangent(model, np.zeros((1, 3)), scales)[0]
    x = torch.as_tensor(np.array([[.02, -.01, .03], [.1, -.04, -.08]])/scales["strain_scale"],
                        dtype=torch.float64).requires_grad_(True)
    record = dict(
        reference=dict(energy_J_per_m3=float(w0[0]), max_abs_stress_Pa=float(np.abs(s0).max()),
                       tangent_relative_misfit=float(np.linalg.norm(d0-d_reference)
                                                     / np.linalg.norm(d_reference)),
                       tangent_min_eigenvalue_Pa=float(np.linalg.eigvalsh(.5*(d0+d0.T)).min())),
        derivatives=dict(
            max_energy_gradient_stress_relative_to_max_stress=float(
                np.max(np.abs(gradient-stress))/stress_scale),
            max_tangent_fd_relative_to_max_tangent=float(
                np.max(np.abs(numerical_tangent-tangent))/tangent_scale),
            max_tangent_asymmetry_Pa=float(np.max(np.abs(tangent-tangent.transpose(0, 2, 1)))),
            gradcheck=bool(torch.autograd.gradcheck(model.energy, (x,), atol=1e-5, rtol=1e-4)),
            gradgradcheck=bool(torch.autograd.gradgradcheck(model.energy, (x,), atol=1e-5,
                                                            rtol=1e-4))),
        cycles={name: cycle_record(model, scales, **spec) for name, spec in CYCLES.items()},
        broad_energy_cloud=broad_energy_cloud(model, scales),
        rank_one=sampled_rank_one_curvature(model, e, scales))
    if constrained:
        record["nonnegative_energy_certificate"] = energy_lower_bound_certificate(model)
        record["lower_bound_scan"] = lower_bound_scan(model)
    return record


def load_free(seed, scales):
    folder = CAPACITY / "training" / f"free_large_seed{seed}"
    report = json.loads((folder / "run_report.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(folder / "model.pt", map_location="cpu", weights_only=False)
    if (report["status"] != "complete" or report["model"] != "Free" or report["seed"] != seed
            or report["model_sha256"] != digest(folder / "model.pt")
            or report["capacity_rule_sha256"] != digest(CAPACITY / "training_rule.json")
            or report["test_labels_loaded"] is not False or report["path_labels_loaded"] is not False
            or checkpoint["strain_scale"] != scales["strain_scale"]
            or checkpoint["energy_scale"] != scales["energy_scale"]):
        raise ValueError(f"Invalid same-rule Unconstrained checkpoint: {folder.name}")
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        model = AnisotropicFreeEnergy(**checkpoint["configuration"]).double()
    finally:
        torch.set_default_dtype(previous)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return dict(model="Unconstrained", seed=seed, slug=f"free_large_seed{seed}",
                model_sha256=report["model_sha256"]), model


def main() -> int:
    target = OUTPUT / "audit.json"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {target}")
    gate = json.loads((EVALUATION / "gate_decision.json").read_text(encoding="utf-8"))
    evaluation = json.loads((EVALUATION / "per_run.json").read_text(encoding="utf-8"))
    if gate.get("status") != "opened_once" or len(evaluation["rows"]) != 12:
        raise ValueError("The m=6 independent evaluation is not complete")
    if not (CAPACITY / "test_path_gateA.json").is_file():
        raise ValueError("The same-rule Unconstrained gate has not been opened")
    scales = json.loads((FEATURES / "manifest.json").read_text(encoding="utf-8"))["scales"]
    # Coordinates and the training reference tangent only; no S/W/D test or path targets.
    with np.load(LABELS, allow_pickle=False) as data:
        e = np.vstack((np.asarray(data["E_test"])[TEST_INDICES],
                       np.asarray(data["E_paths"])[PATH_ENDPOINT_INDICES]))
        d_reference = np.asarray(data["D_reference"], dtype=np.float64)[0]
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    entries = []
    for row in evaluation["rows"]:
        model = _load_model(row)
        if digest(M06_TRAINING / row["slug"] / "model.pt") != row["model_sha256"]:
            raise ValueError(f"Checkpoint differs from the evaluated model: {row['slug']}")
        entries.append((dict(model=row["model"], seed=row["seed"], slug=row["slug"],
                             model_sha256=row["model_sha256"]), model, True))
    for seed in FREE_SEEDS:
        identity, model = load_free(seed, scales)
        entries.append((identity, model, False))
    rows = []
    for identity, model, constrained in entries:
        rows.append(dict(**identity, **audit(model, scales, e, d_reference, constrained)))
        record = rows[-1]
        print(json.dumps(dict(slug=record["slug"],
            reference_stress_Pa=record["reference"]["max_abs_stress_Pa"],
            tangent_fd=record["derivatives"]["max_tangent_fd_relative_to_max_tangent"],
            large_cycle=record["cycles"]["large"]["work_relative_to_scale"],
            certified=record.get("nonnegative_energy_certificate", {}).get("certified"),
            cloud_negative=record["broad_energy_cloud"]["negative_count"],
            rank_one_min_Pa=record["rank_one"]["minimum_Pa"])), flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = dict(status="complete", created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        scope="Post-evaluation implementation checks; the guarantees follow from the construction",
        script_sha256=digest(Path(__file__)),
        certificate_source_sha256=digest(BASE.parent / "06_pann/audit_flexible.py"),
        m06_gate_sha256=digest(EVALUATION / "gate_decision.json"),
        m06_per_run_sha256=digest(EVALUATION / "per_run.json"),
        capacity_rule_sha256=digest(CAPACITY / "training_rule.json"),
        labels_sha256=digest(LABELS), arrays_read=["E_test", "E_paths", "D_reference"],
        target_response_arrays_read=False, test_input_indices=TEST_INDICES.tolist(),
        path_endpoint_indices=PATH_ENDPOINT_INDICES.tolist(), strain_fd_step=STEP,
        cycles=dict(definitions=CYCLES, order_per_edge=CYCLE_ORDER,
                    orientation="counter-clockwise in (E11, E22) at fixed 2E12"),
        rows=rows)
    base._atomic_json(target, result)
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
