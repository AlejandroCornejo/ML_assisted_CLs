"""Fit/reference-only preparation. No torch import, neural fitting or FOM solve.

The explicit recipe is an addendum: preserve the already frozen data protocol.
This NumPy port of the old affine feature design avoids importing its A trainer.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import qr
from scipy.optimize import nnls

from protocol.prepare_design import deterministic_npz, digest

BASE = Path(__file__).resolve().parents[1]
RECIPE = Path(__file__).with_name("training_recipe_v1.json")


def load_fit_reference(store, allowed):
    """NPZ access is lazy: request exactly the seven allowed arrays, not all keys."""
    expected = ("E_fit", "S_fit", "W_fit", "E_reference", "S_reference",
                "W_reference", "D_reference")
    if tuple(allowed) != expected:
        raise ValueError("Unexpected selection access policy")
    arrays = {key: np.asarray(store[key], dtype=np.float64) for key in expected}
    n = len(arrays["E_fit"])
    shapes = {"E_fit": (n, 3), "S_fit": (n, 3), "W_fit": (n,),
              "E_reference": (1, 3), "S_reference": (1, 3),
              "W_reference": (1,), "D_reference": (1, 3, 3)}
    if any(arrays[key].shape != shape or not np.isfinite(arrays[key]).all()
           for key, shape in shapes.items()):
        raise ValueError("Invalid fit/reference arrays")
    if not np.array_equal(arrays["E_reference"], np.zeros((1, 3))):
        raise ValueError("Reference must be undeformed")
    return arrays


def candidate_bank(setting):
    rows = []
    projection = setting["common_start_projection"]
    for angle in np.arange(12)*np.pi/12:
        for old_p in setting["candidate_powers"]:
            for old_q in setting["candidate_powers"]:
                for old_b in sorted(set((0., old_p-.5, 2*old_p-1))):
                    for old_c in sorted(set((0., old_q-.5, 2*old_q-1))):
                        p, q = max(old_p, projection["minimum_power"]), max(old_q, projection["minimum_power"])
                        fb = np.clip(old_b/(2*p-1), projection["minimum_ratio"], projection["maximum_ratio"])
                        fc = np.clip(old_c/(2*q-1), projection["minimum_ratio"], projection["maximum_ratio"])
                        rows.append((angle, p, q, (2*p-1)*fb, (2*q-1)*fc))
    return np.asarray(rows, dtype=np.float64)


def kinematics(e):
    e = np.asarray(e, dtype=np.float64)
    det = (1+2*e[:, 0])*(1+2*e[:, 1])-e[:, 2]**2
    if np.any(1+2*e[:, 0] <= 0) or np.any(det <= 0):
        raise ValueError("C is not positive definite")
    j = np.sqrt(det)
    gj = np.column_stack((1+2*e[:, 1], 1+2*e[:, 0], -e[:, 2]))/j[:, None]
    return j, gj


def raw_features(e, specs):
    angle, p, q, b, c = np.asarray(specs).T
    co, si = np.cos(angle), np.sin(angle)
    dt = np.stack((2*co**2, 2*si**2, 2*co*si))
    du = np.stack((2*si**2, 2*co**2, -2*co*si))
    t, u = 1+e@dt, 1+e@du
    j, _ = kinematics(e)
    return t**p*j[:, None]**(-b)/p+u**q*j[:, None]**(-c)/q


def affine_design(e, specs):
    """Columns: z-z0-rho*(J-1), (J-1)^2/2, J-1-log(J).

    Derivatives are with respect to physical e=(E11,E22,2E12), conjugate
    to (S11,S22,S12); the fixed epsilon term is handled separately.
    """
    angle, p, q, b, c = np.asarray(specs).T
    co, si = np.cos(angle), np.sin(angle)
    dt = np.stack((2*co**2, 2*si**2, 2*co*si))
    du = np.stack((2*si**2, 2*co**2, -2*co*si))
    t, u = 1+e@dt, 1+e@du
    j, gj = kinematics(e)
    aa, bb = t**p*j[:, None]**(-b), u**q*j[:, None]**(-c)
    rho = 2-b/p-c/q
    a = (aa-1)/p+(bb-1)/q-rho*(j-1)[:, None]
    jac = ((aa/t)[:, None, :]*dt+(bb/u)[:, None, :]*du
           -((b*aa/p+c*bb/q)/j[:, None]+rho)[..., None, :]*gj[:, :, None])
    return (np.column_stack((a, .5*(j-1)**2, j-1-np.log(j))),
            np.concatenate((jac, ((j-1)[:, None]*gj)[:, :, None],
                            ((1-1/j)[:, None]*gj)[:, :, None]), axis=2))


def tangent_dictionary(specs):
    """Exact strain Hessians of affine_design at e=0 (not sampled tangents).

    DJ=v=(1,1,0); D2(log J)=L=diag(-2,-2,-1); D2J=vv^T+L.
    Applying the product rule to each stretch power and subtracting rho*D2J
    gives the expression below. Both analytic volume columns have vv^T.
    """
    v = np.array([1., 1., 0.])
    vv, ll = np.outer(v, v), np.diag([-2., -2., -1.])
    result = []
    for angle, p, q, b, c in specs:
        co, si = np.cos(angle), np.sin(angle)
        dt = np.array([2*co*co, 2*si*si, 2*co*si])
        du = np.array([2*si*si, 2*co*co, -2*co*si])
        rho = 2-b/p-c/q
        result.append((p-1)*np.outer(dt, dt)+(q-1)*np.outer(du, du)
                      -b*(np.outer(dt, v)+np.outer(v, dt))
                      -c*(np.outer(du, v)+np.outer(v, du))
                      +(b*b/p+c*c/q-rho)*vv-2*ll)
    return np.stack(result+[vv, vv], axis=-1)


def fit_scales(e, s, w, d0, fraction):
    ss, es = float(np.abs(e).max()), float(np.abs(w).max())
    if ss <= 0 or es <= 0:
        raise ValueError("Degenerate fit scales")
    wn, sn, dn = w/es, s*ss/es, d0*ss**2/es
    wd, sd, dd = float(np.mean(wn**2)), float(np.mean(sn**2)), float(np.mean(dn**2))
    if min(wd, sd, dd) <= 0:
        raise ValueError("Degenerate objective denominators")
    component = np.sqrt(np.mean(s**2, axis=0))
    global_component = float(np.sqrt(np.mean(s**2)))
    stress_floor = fraction*float(np.sqrt(np.mean(np.sum(s**2, axis=1))))
    j, _ = kinematics(e)
    free = np.column_stack((2*e[:, 0], 2*e[:, 1], e[:, 2], j-1))
    return dict(strain_scale=ss, energy_scale=es, stress_denominator=sd,
                energy_denominator=wd, tangent_denominator=dd,
                free_feature_scale=np.maximum(np.abs(free).max(axis=0), 1e-4).tolist(),
                stress_component_rms=component.tolist(),
                stress_component_metric_scale=np.maximum(component, fraction*global_component).tolist(),
                energy_metric_floor=fraction*float(np.sqrt(np.mean(w**2))),
                stress_metric_floor=stress_floor, tangent_metric_floor=stress_floor/ss)


def weighted_system(a, jac, w, s, scales, energy_weight):
    n, columns = a.shape
    ss, es = scales["strain_scale"], scales["energy_scale"]
    ew = np.sqrt(energy_weight/(scales["energy_denominator"]*n))
    sw = 1/np.sqrt(scales["stress_denominator"]*3*n)
    return (np.concatenate((a*ew, (jac*ss).reshape(-1, columns)*sw)),
            np.r_[w/es*ew, (s*ss/es).ravel()*sw])


def select_indices(matrix, target, reference, reference_target, coef, ccoef, setting):
    support = np.flatnonzero((coef[:-2] > setting["support_coefficient_threshold"])
                             | (ccoef[:-2] > setting["support_coefficient_threshold"]))
    score = (coef[:-2]*np.linalg.norm(matrix[:, :-2], axis=0)/np.linalg.norm(target)
             + ccoef[:-2]*np.linalg.norm(reference[:, :-2], axis=0)/np.linalg.norm(reference_target))
    count = setting["count"]
    if len(support) > count:
        selected = sorted(map(int, support), key=lambda k: (-score[k], k))[:count]
    else:
        selected = list(map(int, support))
    rows = np.linspace(0, len(matrix)-1, min(setting["qr_rows"], len(matrix)), dtype=int)
    small = matrix[rows, :-2]
    small = small/np.maximum(np.linalg.norm(small, axis=0), setting["qr_column_norm_floor"])
    _, _, pivot = qr(small, mode="economic", pivoting=True)
    for k in pivot:
        if len(selected) == count:
            break
        if int(k) not in selected:
            selected.append(int(k))
    if len(selected) != count or len(set(selected)) != count:
        raise ValueError("Cannot select the declared feature count")
    return np.asarray(selected), support, score, pivot, rows


def prepare(arrays, recipe):
    e, s, w = (arrays[key] for key in ("E_fit", "S_fit", "W_fit"))
    d0 = arrays["D_reference"][0]
    scales = fit_scales(e, s, w, d0, recipe["scaling"]["metric_floor_fraction"])
    setting = recipe["feature_selection"]
    bank = candidate_bank(setting)
    print(json.dumps(dict(stage="candidate_design", fit_count=len(e), candidates=len(bank))), flush=True)
    a, jac = affine_design(e, bank)
    matrix, target = weighted_system(a, jac, w, s, scales, recipe["objective"]["energy_weight"])
    del a, jac
    print(json.dumps(dict(stage="fit_nnls")), flush=True)
    coef, fit_residual = nnls(matrix, target, maxiter=setting["nnls_maxiter"])
    reference = tangent_dictionary(bank).reshape(9, -1)
    ref_target = d0.ravel()/scales["energy_scale"]
    print(json.dumps(dict(stage="reference_nnls_and_qr")), flush=True)
    ccoef, ref_residual = nnls(reference, ref_target, maxiter=setting["nnls_maxiter"])
    selected, support, score, pivot, rows = select_indices(matrix, target, reference, ref_target, coef, ccoef, setting)
    del matrix
    specs = bank[selected]
    a, jac = affine_design(e, specs)
    # Fixed growth coefficient in normalized energy units. D2(phi_eps)|0=-L.
    eps = recipe["models"]["constrained"]["epsilon_and_volume_floor"]
    j, gj = kinematics(e)
    phi = e[:, 0]+e[:, 1]-np.log(j)
    gphi = np.array([1., 1., 0.])-gj/j[:, None]
    matrix, target = weighted_system(a, jac, w-scales["energy_scale"]*eps*phi,
        s-scales["energy_scale"]*eps*gphi, scales, recipe["objective"]["energy_weight"])
    td = tangent_dictionary(specs)
    dd = scales["strain_scale"]**2/scales["energy_scale"]
    tw = np.sqrt(recipe["objective"]["reference_tangent_weight"]/(9*scales["tangent_denominator"]))
    augmented = np.vstack((matrix, td.reshape(9, -1)*scales["strain_scale"]**2*tw))
    augmented_target = np.r_[target, (d0*dd-eps*np.diag([2., 2., 1.])*scales["strain_scale"]**2).ravel()*tw]
    subset_coef, subset_residual = nnls(augmented, augmented_target, maxiter=setting["nnls_maxiter"])
    z0 = (1/specs[:, 1:3]).sum(axis=1)
    fscale = np.maximum(np.abs(raw_features(e, specs)-z0).max(axis=0), 1e-4)
    diagnostics = dict(candidate_fit_weighted_residual=float(fit_residual),
        candidate_reference_relative_residual=float(ref_residual/np.linalg.norm(ref_target)),
        subset_training_objective=float(subset_residual**2),
        subset_fit_stress_relative=float(np.linalg.norm(jac@subset_coef+eps*gphi-s/scales["energy_scale"])/np.linalg.norm(s/scales["energy_scale"])),
        subset_fit_energy_relative=float(np.linalg.norm(a@subset_coef+eps*phi-w/scales["energy_scale"])/np.linalg.norm(w/scales["energy_scale"])),
        subset_reference_relative=float(np.linalg.norm(td@subset_coef+eps*np.diag([2., 2., 1.])-d0/scales["energy_scale"])/np.linalg.norm(d0/scales["energy_scale"])),
        support_count=len(support), overflow_used=len(support)>setting["count"], candidate_count=len(bank))
    payload = dict(specs=specs, candidate_bank=bank, selected_candidate_indices=selected,
        support_candidate_indices=support, support_scores=score, qr_pivot_order=pivot,
        qr_fit_matrix_rows=rows, candidate_fit_coefficients=coef,
        candidate_reference_coefficients=ccoef, initialization_coefficients=subset_coef,
        feature_center=z0, feature_scale=fscale, reference_tangent=d0)
    return payload, scales, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=RECIPE)
    parser.add_argument("--labels", type=Path, default=BASE/"results/data_labels_v1.npz")
    parser.add_argument("--out", type=Path, required=True, help="New output directory; never overwrites")
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    if digest(args.recipe.with_name("data_protocol_v1.json")) != recipe["parent_data_protocol_sha256"]:
        raise ValueError("Parent data protocol changed")
    if digest(args.labels) != recipe["labels_sha256"]:
        raise ValueError("Unapproved dataset hash")
    report_path = args.labels.with_suffix(".json")
    report = json.loads(report_path.read_text())
    if not report.get("passed") or report.get("status") != "complete":
        raise ValueError("FOM assembly gate does not pass")
    parent = json.loads(args.recipe.with_name("data_protocol_v1.json").read_text())
    if (recipe["feature_selection"]["count"] != parent["models"]["paired_features"]
            or recipe["models"]["names"] != parent["models"]["primary"]
            or recipe["models"]["seeds"] != parent["models"]["initialization_seeds"]
            or recipe["objective"]["energy_weight"] != parent["training_objective"]["energy_weight"]):
        raise ValueError("Recipe disagrees with parent comparison")
    args.out.mkdir(parents=True, exist_ok=False)
    with np.load(args.labels, allow_pickle=False) as store:
        arrays = load_fit_reference(store, recipe["selection_allowed_arrays"])
    if len(arrays["E_fit"]) != parent["sampling"]["fit_total"]:
        raise ValueError("Incorrect fit count")
    payload, scales, diagnostics = prepare(arrays, recipe)
    output = args.out/"feature_table.npz"
    deterministic_npz(output, payload)
    sources = [Path(__file__), Path(__file__).with_name("prepare_design.py")]
    manifest = dict(status="complete", neural_training_started=False,
        accessed_label_arrays=recipe["selection_allowed_arrays"],
        validation_used=False, test_used=False, paths_used=False,
        recipe_sha256=digest(args.recipe), data_protocol_sha256=recipe["parent_data_protocol_sha256"],
        labels_sha256=digest(args.labels), assembly_report_sha256=digest(report_path),
        feature_table_sha256=digest(output), scales=scales, diagnostics=diagnostics,
        sources_sha256={str(p.relative_to(BASE)): digest(p) for p in sources},
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                         thread_environment={key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}),
        limitations=recipe["feature_selection"]["limitations"])
    (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    (args.out/"training_recipe_used.json").write_bytes(args.recipe.read_bytes())
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
