#!/usr/bin/env python3
"""Reproducible mechanical witnesses for the current periodic coupon RVE.

This script has two deliberately *different* purposes.  It does not re-train
or select a model.

``cycle``
    A small closed rectangle in the two independent normal Green strains.
    A hyperelastic law has zero stress work around it.  Therefore it is a
    direct falsification test for a stress-regression law without a potential.

``rank-one``
    A sampled Legendre--Hadamard test in F-space.  A negative result is a
    *candidate* instability of the unconstrained free-energy model.  It is
    reported with a finite-difference energy check, but it is not attributed
    to the real RVE until ``--verify-fom-candidate`` has also checked the FOM.

The current FOM, Regression, Free, ICNN and ICKAN checkpoints are frozen and
their SHA256 hashes are written to the report.  The test and probe labels are
used for audit only, after the model choices made in ``selected_models.json``.

Examples
--------
Fast PANN-only audit::

  PYTHONPATH=coupon_fe2_paper/.pydeps:/home/sares/Kratos_Eigen_Check/bin/Release \\
  python3 -B coupon_fe2_paper/06_pann/mechanics_witnesses.py

Include the independent periodic FOM control for the closed cycle::

  PYTHONPATH=coupon_fe2_paper/.pydeps:/home/sares/Kratos_Eigen_Check/bin/Release \\
  python3 -B coupon_fe2_paper/06_pann/mechanics_witnesses.py --with-fom-cycle

If, and only if, a negative Free candidate has been found, the final command
also checks the FOM energy at the candidate and its two rank-one neighbours::

  ... mechanics_witnesses.py --with-fom-cycle --verify-fom-candidate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent
RVE = ROOT / "00_rve"
FE2 = ROOT / "06_fe2"
RVE_EXTENSION = REPO / "RVE_NeoHookean_Homogenization" / "fe2_extension"
RVE_CORE = REPO / "RVE_NeoHookean_Homogenization" / "core"
RESULTS = HERE / "mechanics_witness_results"

for directory in (str(HERE), str(FE2), str(RVE), str(RVE_EXTENSION), str(RVE_CORE)):
    if directory not in sys.path:
        sys.path.insert(0, directory)
for kratos_path in (Path("/home/sares/Kratos_Eigen_Check/bin/Release"),
                    Path("/home/kratos/Kratos_Eigen_Check/bin/Release")):
    if kratos_path.is_dir() and str(kratos_path) not in sys.path:
        sys.path.append(str(kratos_path))
        break


# This centre is the held-out admissible state at which the current Regression
# checkpoint has its largest test-set tangent antisymmetry.  It was identified
# mechanically (not by the outcome of a loop), is inside the training box, and
# the rectangle below remains inside it.
CYCLE_CENTRE = np.array([0.005126003282957469,
                         -0.09455076131169517,
                         -0.1283720601618937])
CYCLE_HALF_WIDTH = 1.0e-3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_json(value: Any) -> Any:
    """Convert NumPy scalars and arrays without silently allowing NaNs."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): as_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_json(v) for v in value]
    return value


def selected_paths() -> dict[str, Path]:
    """The validation-selected constrained models, plus fixed baselines."""
    chosen = json.loads((HERE / "enrichment_results" / "selected_models.json").read_text())
    # Paths in the historical JSON mention the old machine.  The run name is
    # the reproducibility identifier; resolve it inside this checkout.
    icnn_name = chosen["candidates"]["icnn"][0]["run"]
    ickan_candidates = chosen["candidates"]["ickan"]
    # The selected-model file lists the candidates in validation order, but we
    # explicitly take the smallest validation loss instead of relying on order.
    ickan_name = min(ickan_candidates, key=lambda r: r["validation_stress"])["run"]
    paths = {
        "Regression": HERE / "pann_regression.pt",
        "Free": HERE / "pann_free.pt",
        "ICNN": HERE / "enrichment_results" / icnn_name / "model.pt",
        "ICKAN": HERE / "enrichment_results" / ickan_name / "model.pt",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen checkpoints: " + ", ".join(missing))
    return paths


def load_laws(paths: dict[str, Path]):
    from regression_pann_law import CouponRegressionLaw
    from free_pann_law import CouponFreePANNLaw
    from flexible_pann_law import FlexiblePANNLaw

    return {
        "Regression": CouponRegressionLaw(paths["Regression"]),
        "Free": CouponFreePANNLaw(paths["Free"]),
        "ICNN": FlexiblePANNLaw(paths["ICNN"]),
        "ICKAN": FlexiblePANNLaw(paths["ICKAN"]),
    }


def rectangle_points(centre: np.ndarray, half_width: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss points and vector increments along one counter-clockwise cycle."""
    corners = np.array([
        centre + [-half_width, -half_width, 0.0],
        centre + [+half_width, -half_width, 0.0],
        centre + [+half_width, +half_width, 0.0],
        centre + [-half_width, +half_width, 0.0],
        centre + [-half_width, -half_width, 0.0],
    ])
    xi, wi = np.polynomial.legendre.leggauss(order)
    points, increments = [], []
    for start, end in zip(corners[:-1], corners[1:]):
        points.append(0.5 * (start + end) + 0.5 * xi[:, None] * (end - start))
        increments.append(0.5 * wi[:, None] * (end - start))
    return np.concatenate(points), np.concatenate(increments)


def potential_cycle(law, points: np.ndarray, increments: np.ndarray) -> dict[str, float]:
    ans = law.response(points, tangent=False)
    return {
        "stress_work_J_per_m3": float(np.einsum("ni,ni->", ans["stress"], increments)),
        "max_stress_Pa": float(np.linalg.norm(ans["stress"], axis=1).max()),
    }


def cycle_audit(laws: dict[str, Any], order: int) -> dict[str, Any]:
    points, increments = rectangle_points(CYCLE_CENTRE, CYCLE_HALF_WIDTH, order)
    result: dict[str, Any] = {
        "centre_E": CYCLE_CENTRE,
        "half_width_in_E11_and_E22": CYCLE_HALF_WIDTH,
        "quadrature_order_per_edge": order,
        "cycle_orientation": "counter-clockwise in (E11, E22), with gamma12 fixed",
        "models": {},
    }
    for name, law in laws.items():
        record = potential_cycle(law, points, increments)
        if name != "Regression":
            # The energy is a state function: this is an exact endpoint check
            # independent of the stress quadrature used for the loop work.
            endpoint_energy = law.response(CYCLE_CENTRE[None, :], tangent=False)["energy"][0]
            record["endpoint_energy_difference_J_per_m3"] = float(endpoint_energy - endpoint_energy)
        result["models"][name] = record
    # The spline ICKAN is C2 but not a low-degree polynomial along a generic
    # strain path.  Recording this short convergence table prevents its tiny
    # quadrature residual from being mistaken for non-conservative physics.
    convergence = {}
    for q_order in (8, 16, 32, 64, 128):
        q_points, q_increments = rectangle_points(CYCLE_CENTRE, CYCLE_HALF_WIDTH, q_order)
        convergence[str(q_order)] = {
            name: potential_cycle(law, q_points, q_increments)["stress_work_J_per_m3"]
            for name, law in laws.items()
        }
    result["quadrature_work_convergence_J_per_m3"] = convergence
    return result


def F_from_E(E: np.ndarray) -> np.ndarray:
    """Positive symmetric square root of C=I+2E, with engineering shear."""
    E = np.asarray(E, dtype=float).reshape(-1, 3)
    C = np.empty((len(E), 2, 2))
    C[:, 0, 0] = 1.0 + 2.0 * E[:, 0]
    C[:, 1, 1] = 1.0 + 2.0 * E[:, 1]
    C[:, 0, 1] = C[:, 1, 0] = E[:, 2]
    eig, vec = np.linalg.eigh(C)
    if np.min(eig) <= 0.0:
        raise ValueError("E does not define a positive-definite C")
    return np.einsum("nik,nk,njk->nij", vec, np.sqrt(eig), vec)


def E_from_F(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=float).reshape(-1, 2, 2)
    C = np.matmul(np.swapaxes(F, 1, 2), F)
    return np.column_stack(((C[:, 0, 0] - 1.0) / 2.0,
                            (C[:, 1, 1] - 1.0) / 2.0,
                            C[:, 0, 1]))


def min_rank_one_curvature(law, E: np.ndarray, n_b: int, *,
                           tags: np.ndarray | None = None,
                           tag_name: str | None = None) -> dict[str, Any]:
    """Sample the exact F-space Legendre--Hadamard expression.

    With H=a outer b and |a|=|b|=1,

      D2 W(F)[H,H] = a.T [B.T (dS/dE) B + (b.T S b) I] a,

    where B maps a to dE under the rank-one increment.  This includes the
    geometric stress term; testing eigenvalues of dS/dE alone would be wrong.
    """
    E = np.asarray(E, dtype=float).reshape(-1, 3)
    F = F_from_E(E)
    out = law.response(E, tangent=True)
    D, stress = out["tangent"], out["stress"]
    S = np.zeros((len(E), 2, 2))
    S[:, 0, 0], S[:, 1, 1] = stress[:, 0], stress[:, 1]
    S[:, 0, 1] = S[:, 1, 0] = stress[:, 2]

    if (tags is None) != (tag_name is None):
        raise ValueError("tags and tag_name must be supplied together")
    if tags is not None and len(tags) != len(E):
        raise ValueError("A tag is required for every state")
    best = {"curvature_Pa": float("inf")}
    per_state_minimum = np.full(len(E), np.inf)
    per_state_a = np.empty((len(E), 2))
    per_state_b = np.empty((len(E), 2))
    for theta in np.linspace(0.0, np.pi, n_b, endpoint=False):
        b = np.array([np.cos(theta), np.sin(theta)])
        B = np.stack((b[0] * F[:, :, 0],
                      b[1] * F[:, :, 1],
                      b[1] * F[:, :, 0] + b[0] * F[:, :, 1]), axis=1)
        Q = np.einsum("nki,nkl,nlj->nij", B, D, B)
        Q += np.einsum("i,nij,j->n", b, S, b)[:, None, None] * np.eye(2)
        values, vectors = np.linalg.eigh(Q)
        update = values[:, 0] < per_state_minimum
        per_state_minimum[update] = values[update, 0]
        per_state_a[update] = vectors[update, :, 0]
        per_state_b[update] = b
        index = int(np.argmin(values[:, 0]))
        value = float(values[index, 0])
        if value < best["curvature_Pa"]:
            best = {
                "curvature_Pa": value,
                "state_index": index,
                "E": E[index],
                "F": F[index],
                "a": vectors[index, :, 0],
                "b": b,
                "det_F": float(np.linalg.det(F[index])),
            }
    best["states_with_a_negative_sampled_direction"] = int(np.count_nonzero(per_state_minimum < 0.0))
    best["n_states"] = int(len(E))
    best["n_b_directions"] = int(n_b)
    if tags is not None:
        negative = np.flatnonzero(per_state_minimum < 0.0)
        if len(negative):
            first_tag = float(np.min(tags[negative]))
            # Among the first sampled ring choose the most negative direction;
            # this is deterministic and maximizes the finite-difference signal.
            first = negative[np.isclose(tags[negative], first_tag)]
            index = int(first[np.argmin(per_state_minimum[first])])
            best[f"first_negative_by_{tag_name}"] = {
                tag_name: first_tag,
                "curvature_Pa": float(per_state_minimum[index]),
                "state_index": index,
                "E": E[index],
                "F": F[index],
                "a": per_state_a[index],
                "b": per_state_b[index],
                "det_F": float(np.linalg.det(F[index])),
            }
        else:
            best[f"first_negative_by_{tag_name}"] = None
    return best


def uniform_box_states(rng: np.random.Generator, n: int) -> np.ndarray:
    data = np.load(ROOT / "03_data" / "data.npz")
    lower, upper = data["E_train"].min(axis=0), data["E_train"].max(axis=0)
    raw = rng.uniform(lower, upper, size=(n, 3))
    determinant_C = (1.0 + 2.0 * raw[:, 0]) * (1.0 + 2.0 * raw[:, 1]) - raw[:, 2] ** 2
    return raw[determinant_C > 0.0]


def broad_F_states(rng: np.random.Generator, n: int) -> np.ndarray:
    """Objective broad scan: principal stretches and material angle, no rotations."""
    principal = np.exp(rng.uniform(np.log(0.25), np.log(3.0), size=(n, 2)))
    angle = rng.uniform(0.0, np.pi, size=n)
    co, si = np.cos(angle), np.sin(angle)
    C11 = principal[:, 0] ** 2 * co ** 2 + principal[:, 1] ** 2 * si ** 2
    C22 = principal[:, 0] ** 2 * si ** 2 + principal[:, 1] ** 2 * co ** 2
    C12 = (principal[:, 0] ** 2 - principal[:, 1] ** 2) * co * si
    return np.column_stack(((C11 - 1.0) / 2.0, (C22 - 1.0) / 2.0, C12))


def radial_box_states(rng: np.random.Generator, n_directions: int,
                      rings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Outward rays from the sampled-box centre, labelled by ring factor.

    The direction is Chebyshev-normalized, so a ring factor of one reaches a
    face of the original rectangular data box.  This is a controlled way to
    locate the *first* sampled extrapolative violation rather than presenting
    only the most extreme point in a broad cloud.
    """
    data = np.load(ROOT / "03_data" / "data.npz")
    lower, upper = data["E_train"].min(axis=0), data["E_train"].max(axis=0)
    centre, half_range = 0.5 * (lower + upper), 0.5 * (upper - lower)
    directions = rng.normal(size=(n_directions, 3))
    directions /= np.max(np.abs(directions), axis=1)[:, None]
    states = centre + rings[:, None, None] * directions[None, :, :] * half_range
    tags = np.repeat(rings, n_directions)
    states = states.reshape(-1, 3)
    determinant_C = ((1.0 + 2.0 * states[:, 0]) * (1.0 + 2.0 * states[:, 1])
                     - states[:, 2] ** 2)
    return states[determinant_C > 0.0], tags[determinant_C > 0.0]


def finite_difference_candidate(law, candidate: dict[str, Any]) -> list[dict[str, float]]:
    """An independent finite-difference check of the predicted negative mode."""
    F = np.asarray(candidate["F"], dtype=float)
    H = np.outer(np.asarray(candidate["a"], dtype=float), np.asarray(candidate["b"], dtype=float))
    E0 = E_from_F(F[None, :, :])
    W0 = float(law.response(E0, tangent=False)["energy"][0])
    records = []
    for step in (1.0e-4, 3.0e-4, 1.0e-3):
        fp, fm = F + step * H, F - step * H
        if min(np.linalg.det(fp), np.linalg.det(fm)) <= 0.0:
            continue
        Wp = float(law.response(E_from_F(fp[None, :, :]), tangent=False)["energy"][0])
        Wm = float(law.response(E_from_F(fm[None, :, :]), tangent=False)["energy"][0])
        records.append({"step": step, "central_second_difference_Pa": (Wp + Wm - 2.0 * W0) / step ** 2})
    return records


def rank_one_audit(laws: dict[str, Any], n_box: int, n_broad: int, n_b: int,
                   n_radial_directions: int, radial_rings: np.ndarray) -> dict[str, Any]:
    data = np.load(ROOT / "03_data" / "data.npz")
    valid_probe = np.isfinite(data["S_probe"]).all(axis=1) & np.isfinite(data["W_probe"])
    rng = np.random.default_rng(20260906)
    radial_states, radial_tags = radial_box_states(rng, n_radial_directions, radial_rings)
    clouds = {
        "held_out_test": (data["E_test"], None),
        "converged_probe": (data["E_probe"][valid_probe], None),
        "uniform_training_box": (uniform_box_states(rng, n_box), None),
        "radial_box_expansion": (radial_states, radial_tags),
        "broad_unvalidated_principal_stretch_scan": (broad_F_states(rng, n_broad), None),
    }
    report: dict[str, Any] = {
        "method": "sampled finite-strain Legendre--Hadamard expression including the geometric stress term",
        "random_seed": 20260906,
        "broad_principal_stretch_range": [0.25, 3.0],
        "radial_box_expansion_rings": radial_rings,
        "clouds": {},
    }
    for cloud_name, (states, tags) in clouds.items():
        report["clouds"][cloud_name] = {"models": {}}
        for model_name in ("Free", "ICNN", "ICKAN"):
            record = min_rank_one_curvature(
                laws[model_name], states, n_b,
                tags=tags, tag_name="ring_factor" if tags is not None else None,
            )
            if model_name == "Free":
                record["finite_difference_energy_check"] = finite_difference_candidate(laws[model_name], record)
            report["clouds"][cloud_name]["models"][model_name] = record
    return report


def fom_cycle(order: int) -> dict[str, Any]:
    """Independent FOM control, used only for the cycle test."""
    from _material_law_guard_claude import true_neo_hookean_active
    from periodic_fom import PeriodicRVE

    points, increments = rectangle_points(CYCLE_CENTRE, CYCLE_HALF_WIDTH, order)
    # Reuse the exact 1546-element mesh that generated the frozen data rather
    # than regenerating a nominally identical mesh through Gmsh.  That keeps
    # the witness independent of optional meshing software and removes a
    # needless source of discretization variation.  The reference square is
    # [-1,1]^2, hence its void-inclusive area is 4.
    mesh_base, cell_area, n_elements = ROOT / "03_data" / "rve_mesh", 4.0, 1546
    with true_neo_hookean_active():
        rve = PeriodicRVE(mesh_base, cell_area=cell_area)
        # Traverse in order.  Warm starts are only a Newton aid: this RVE has
        # no history, hence cannot generate a physical loop work.
        last_E = CYCLE_CENTRE
        _s, q = rve.solve(last_E)
        W_initial = rve.homogenized_energy()
        stresses = []
        for point in points:
            stress, q = rve.solve(point, u_ind_init=q, E_start=last_E)
            stresses.append(stress)
            last_E = point
        rve.solve(CYCLE_CENTRE, u_ind_init=q, E_start=last_E)
        W_final = rve.homogenized_energy()
    stresses = np.asarray(stresses)
    return {
        "mesh_elements": n_elements,
        "quadrature_order_per_edge": order,
        "stress_work_J_per_m3": float(np.einsum("ni,ni->", stresses, increments)),
        "endpoint_energy_difference_J_per_m3": float(W_final - W_initial),
        "max_stress_Pa": float(np.linalg.norm(stresses, axis=1).max()),
    }


def fom_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """FOM energy curvature only after the Free scan supplied a candidate."""
    from _material_law_guard_claude import true_neo_hookean_active
    from periodic_fom import PeriodicRVE

    F = np.asarray(candidate["F"], dtype=float)
    H = np.outer(np.asarray(candidate["a"], dtype=float), np.asarray(candidate["b"], dtype=float))
    records: list[dict[str, float]] = []
    with true_neo_hookean_active():
        rve = PeriodicRVE(ROOT / "03_data" / "rve_mesh", cell_area=4.0)
        E0 = E_from_F(F[None, :, :])[0]
        try:
            _s, q = rve.solve(E0)
            W0 = rve.homogenized_energy()
        except RuntimeError as error:
            return {"converged": False, "failure": str(error), "curvatures": records}
        for step in (1.0e-4, 3.0e-4, 1.0e-3):
            fp, fm = F + step * H, F - step * H
            if min(np.linalg.det(fp), np.linalg.det(fm)) <= 0.0:
                continue
            ep, em = E_from_F(fp[None, :, :])[0], E_from_F(fm[None, :, :])[0]
            try:
                rve.solve(ep, u_ind_init=q, E_start=E0)
                Wp = rve.homogenized_energy()
                rve.solve(em, u_ind_init=q, E_start=E0)
                Wm = rve.homogenized_energy()
            except RuntimeError as error:
                return {"converged": False, "failure": str(error), "curvatures": records}
            records.append({"step": step, "central_second_difference_Pa": (Wp + Wm - 2.0 * W0) / step ** 2})
    return {"converged": True, "curvatures": records}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-fom-cycle", action="store_true", help="also solve the periodic FOM along the cycle")
    parser.add_argument("--verify-fom-candidate", action="store_true", help="check a negative Free candidate with the periodic FOM")
    parser.add_argument("--cycle-only", action="store_true", help="run only the inexpensive closed-cycle witness")
    parser.add_argument("--cycle-order", type=int, default=64, help="Gauss points per cycle edge for PANNs; FOM uses min(order, 8)")
    parser.add_argument("--box-states", type=int, default=12000)
    parser.add_argument("--broad-states", type=int, default=24000)
    parser.add_argument("--radial-directions", type=int, default=1800,
                        help="Chebyshev-normalized directions for the outward-box scan")
    parser.add_argument("--radial-max-ring", type=float, default=4.0)
    parser.add_argument("--radial-ring-step", type=float, default=0.05)
    parser.add_argument("--rank-one-directions", type=int, default=180)
    parser.add_argument("--fom-candidate-from", type=Path,
                        help="read the first radial Free candidate from a prior audit and only verify it with the FOM")
    parser.add_argument("--free-radial-only", action="store_true",
                        help="fast, standalone Free-only radial scan; use before an FOM candidate verification")
    parser.add_argument("--output", type=Path, default=RESULTS / "current_rve_mechanics_witnesses.json")
    args = parser.parse_args()
    if args.fom_candidate_from is not None:
        source = json.loads(args.fom_candidate_from.read_text())
        try:
            if "free_radial" in source:
                candidate = source["free_radial"]["first_negative_by_ring_factor"]
            else:
                candidate = source["rank_one"]["clouds"]["radial_box_expansion"]["models"]["Free"][
                    "first_negative_by_ring_factor"]
        except KeyError as error:
            raise ValueError(f"{args.fom_candidate_from} has no radial Free candidate") from error
        if candidate is None:
            report = {
                "source_audit": str(args.fom_candidate_from),
                "status": "not_run",
                "reason": "The source radial scan found no negative Free candidate.",
            }
        else:
            print("Verifying the stored first negative Free candidate with the independent FOM...", flush=True)
            report = {
                "source_audit": str(args.fom_candidate_from),
                "candidate": candidate,
                "result": fom_candidate(candidate),
            }
        output = (RESULTS / "current_rve_first_negative_fom_verification.json"
                  if args.output == RESULTS / "current_rve_mechanics_witnesses.json" else args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(as_json(report), indent=2, allow_nan=False) + "\n")
        print(json.dumps({"output": str(output), "status": report["status"] if "status" in report else report["result"]["converged"]}, indent=2))
        return 0
    if (min(args.cycle_order, args.box_states, args.broad_states,
            args.rank_one_directions, args.radial_directions) < 1
            or args.radial_max_ring < 1.0 or args.radial_ring_step <= 0.0):
        raise ValueError("All sample counts and quadrature orders must be positive")
    radial_rings = np.arange(1.0, args.radial_max_ring + 0.5 * args.radial_ring_step,
                             args.radial_ring_step)

    if args.cycle_only:
        paths = selected_paths()
        report = {
            "scope": "frozen current periodic RVE; closed-cycle witness only, no retraining or selection",
            "checkpoints": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
            "cycle": cycle_audit(load_laws(paths), args.cycle_order),
        }
        if args.with_fom_cycle:
            print("Running independent FOM closed-cycle control...", flush=True)
            report["cycle"]["FOM"] = fom_cycle(min(args.cycle_order, 8))
        output = (RESULTS / "current_rve_cycle_audit.json"
                  if args.output == RESULTS / "current_rve_mechanics_witnesses.json" else args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(as_json(report), indent=2, allow_nan=False) + "\n")
        print(json.dumps({"output": str(output),
                          "Regression_cycle_work_J_per_m3": report["cycle"]["models"]["Regression"]["stress_work_J_per_m3"]}, indent=2))
        return 0

    if args.free_radial_only:
        paths = selected_paths()
        law = load_laws(paths)["Free"]
        rng = np.random.default_rng(20260906)
        states, tags = radial_box_states(rng, args.radial_directions, radial_rings)
        free_radial = min_rank_one_curvature(law, states, args.rank_one_directions,
                                             tags=tags, tag_name="ring_factor")
        candidate = free_radial["first_negative_by_ring_factor"]
        if candidate is not None:
            free_radial["finite_difference_energy_check"] = finite_difference_candidate(law, candidate)
        report = {
            "scope": "frozen current periodic RVE; Free-only outward-box audit, no retraining or selection",
            "checkpoint": {"path": str(paths["Free"]), "sha256": sha256(paths["Free"])},
            "method": "sampled finite-strain Legendre--Hadamard expression including the geometric stress term",
            "random_seed": 20260906,
            "radial_box_expansion_rings": radial_rings,
            "free_radial": free_radial,
        }
        output = (RESULTS / "current_rve_free_radial_scan.json"
                  if args.output == RESULTS / "current_rve_mechanics_witnesses.json" else args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(as_json(report), indent=2, allow_nan=False) + "\n")
        print(json.dumps(as_json({"output": str(output), "first_negative": candidate}), indent=2))
        return 0

    paths = selected_paths()
    laws = load_laws(paths)
    report: dict[str, Any] = {
        "scope": "frozen current periodic elliptical-void RVE; no retraining and no model selection",
        "checkpoints": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
        "cycle": cycle_audit(laws, args.cycle_order),
        "rank_one": rank_one_audit(laws, args.box_states, args.broad_states,
                                    args.rank_one_directions, args.radial_directions,
                                    radial_rings),
    }
    if args.with_fom_cycle:
        print("Running independent FOM closed-cycle control...", flush=True)
        report["cycle"]["FOM"] = fom_cycle(min(args.cycle_order, 8))

    if args.verify_fom_candidate:
        radial = report["rank_one"]["clouds"]["radial_box_expansion"]["models"]["Free"]
        candidate = radial["first_negative_by_ring_factor"]
        if candidate is None:
            report["fom_candidate"] = {"status": "not_run", "reason": "No negative Free candidate in this predeclared scan."}
        else:
            print("Verifying the negative Free candidate with the independent FOM...", flush=True)
            report["fom_candidate"] = {
                "status": "run",
                "source": "first negative state in the radial box expansion",
                "candidate": candidate,
                "finite_difference_energy_check": fom_candidate(candidate),
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    clean = as_json(report)
    args.output.write_text(json.dumps(clean, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "Regression_cycle_work_J_per_m3": clean["cycle"]["models"]["Regression"]["stress_work_J_per_m3"],
        "Free_min_broad_rank_one_curvature_Pa": clean["rank_one"]["clouds"]["broad_unvalidated_principal_stretch_scan"]["models"]["Free"]["curvature_Pa"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
