#!/usr/bin/env python3
"""Staged B-only reference preflight: validate the predictor, then check meshes/domain."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import json
import time
from pathlib import Path
import numpy as np
from run_pilot import HERE, ROOT, derivative_check, digest, field_stats, pf, relative
from adaptive_continuation import ContinuationFailure, solve
from geometry import build, cavity_parameters
from audit_pilot import polygon_distance, self_intersection, rank_one_screen


def save(report, output):
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def old_states(spec):
    folder = HERE / spec["parent_refinement"]
    parent = json.loads((folder / "report.json").read_text())
    rows = {(s["path"], s["fraction"]): s for s in parent["states"] if s["ok"]}
    retry = HERE / spec["parent_y_compression"]
    replacement = json.loads((retry / "report.json").read_text())
    if replacement["mesh_sha256"] != parent["mesh_sha256"]:
        raise ValueError("Targeted retry does not use the original finer mesh")
    rows.update({(s["path"], s["fraction"]): s for s in replacement["states"] if s["ok"]})
    if len(rows) != 32:
        raise ValueError("Original comparison does not contain all 32 targets")
    return rows, parent, folder, retry


def rim_indices(geometry, xy):
    result = []
    for hole in cavity_parameters(geometry):
        t = np.radians(hole["angle"])
        rotation = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        local = (xy-hole["center"]) @ rotation
        ids = np.where(np.abs((local[:, 0]/hole["a"])**2+(local[:, 1]/hole["b"])**2-1) < 1e-7)[0]
        if len(ids) < 6:
            raise RuntimeError("Cannot identify cavity boundary")
        order = np.argsort(np.arctan2(local[ids, 1]/hole["b"], local[ids, 0]/hole["a"]))
        result.append(ids[order])
    return result


def boundary_screen(xy, displacement, rims, e, L):
    from fom_solver_rve import DeformationGradientFromGreenLagrange2D
    F = DeformationGradientFromGreenLagrange2D(e)
    polygons = [(xy+displacement)[ids] for ids in rims]
    minimum = np.inf
    import itertools
    for i, first in enumerate(polygons):
        for j, second in enumerate(polygons):
            for shift in itertools.product((-1, 0, 1), repeat=2):
                if i == j and shift == (0, 0):
                    continue
                minimum = min(minimum, polygon_distance(first, second+F@(L*np.array(shift))))
    errors = []
    for axis in (0, 1):
        faces = [np.where(np.isclose(xy[:, axis], s*L/2, atol=1e-8, rtol=0))[0] for s in (-1, 1)]
        first, last = [ids[np.argsort(xy[ids, 1-axis])] for ids in faces]
        error = displacement[last]-displacement[first]-(xy[last]-xy[first])@(F-np.eye(2)).T
        errors.append(float(np.linalg.norm(error, axis=1).max()))
    return dict(min_polygon_gap=float(minimum), self_intersection=any(self_intersection(p) for p in polygons),
                max_periodic_jump_error=max(errors))


def evaluate(rve, e, controls, start=None, q_start=None):
    _stress, q, attempts = solve(rve, e, controls["initial_guess_max_increment"],
                                controls["min_increment"], start, q_start)
    stress, tangent, q = rve.stress_and_tangent_consistent(
        e, u_ind_init=q, E_start=e, return_state=True)
    fields, u, pk1 = field_stats(rve, e, q)
    row = dict(ok=True, strain=np.asarray(e).tolist(), stress=stress.tolist(), tangent=tangent.tolist(),
               energy=rve.homogenized_energy(), fields=fields, attempts=attempts)
    return row, q, u, pk1


def differences(row, ref):
    return dict(stress=relative(row["stress"], ref["stress"]),
                tangent=relative(row["tangent"], ref["tangent"]),
                energy=relative(row["energy"], ref["energy"]),
                pk1_l2=relative(row["fields"]["pk1_l2"], ref["fields"]["pk1_l2"]),
                pk1_max=relative(row["fields"]["pk1_max"], ref["fields"]["pk1_max"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=HERE / "preflight_spec.json")
    parser.add_argument("--stage", choices=("validation", "reference", "check"), required=True)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    controls = json.loads(args.spec.read_text())
    rows, parent, folder, retry = old_states(controls)
    geometry = parent["parent_spec"]
    if args.stage != "validation":
        if not args.validation:
            raise ValueError("First validate the continuation helper against existing results")
        validation = json.loads((args.validation / "report.json").read_text())
        if not validation.get("passed") or validation["spec_sha256"] != digest(args.spec):
            raise ValueError("Validation failed or uses a different predeclared specification")
        if validation["helper_sha256"] != digest(HERE / "adaptive_continuation.py"):
            raise ValueError("Continuation helper changed since validation")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "spec.json").write_bytes(args.spec.read_bytes())
    report = dict(stage=args.stage, status="running", spec=controls, spec_sha256=digest(args.spec),
                  helper_sha256=digest(HERE / "adaptive_continuation.py"),
                  driver_sha256=digest(Path(__file__)), states=[], failures=[],
                  parent_report_sha256=digest(folder / "report.json"),
                  parent_mesh_sha256=parent["mesh_sha256"])
    save(report, output)
    start_time = time.perf_counter()
    role = "check" if args.stage == "check" else "reference"
    if role == "check":
        report["geometry"], xy, triangles = build(geometry, controls["mesh"], output / role)
    else:
        for suffix in (".mdpa", ".npz"):
            (output / (role+suffix)).write_bytes((folder / ("finer"+suffix)).read_bytes())
        saved = np.load(output / (role+".npz"))
        xy, triangles = saved["xy"], saved["triangles"]
        report["geometry"] = parent["geometry"]
    report["mesh_sha256"] = digest(output / (role+".mdpa"))
    print(args.stage, "mesh:", len(triangles), "elements", flush=True)
    from _material_law_guard_claude import true_neo_hookean_active
    with true_neo_hookean_active():
        rve = pf.PeriodicRVE(output / role, cell_area=geometry["cell_side"]**2)
        if args.stage == "validation":
            for path in controls["baseline_paths"]:
                ref = rows[path, 1.0]
                source = retry if path == "axial_y_compression" else folder
                data = np.load(source / (path+"_endpoint.npz"))
                for increment in controls["validation_max_increments"]:
                    settings = dict(controls, initial_guess_max_increment=increment)
                    record = dict(path=path, max_increment=increment, ok=False)
                    report["states"].append(record)
                    try:
                        row, q, u, _pk1 = evaluate(rve, ref["strain"], settings)
                        error = differences(row, ref)
                        error["node_displacement"] = relative(u[rve._eq_map], data["node_displacement"], 1e-12)
                        record.update(row, differences=error,
                            passed=all(error[k] <= controls["baseline_agreement_tolerance"]
                                       for k in ("stress", "tangent", "energy", "node_displacement")))
                    except Exception as exc:
                        record["error"] = repr(exc)
                        record["attempts"] = getattr(exc, "attempts", [])
                        report["failures"].append(dict(path=path, error=repr(exc)))
                    save(report, output)
                print("validated", path, flush=True)
            expected = len(controls["baseline_paths"])*len(controls["validation_max_increments"])
            report["passed"] = len(report["states"]) == expected and all(s.get("passed") for s in report["states"])
        else:
            cases = []
            if role == "check":
                for (path, fraction), ref in rows.items():
                    cases.append((f"pilot__{path}__{fraction:g}", ref["strain"], path, ref))
                # Dictionary replacement of y compression changes insertion order.
                cases.sort(key=lambda c: (c[2], np.linalg.norm(c[1])))
            cases += [("domain__"+name, e, None, None) for name, e in controls["domain_states"].items()]
            rims = rim_indices(geometry, xy)
            last_group, previous, q_previous = None, np.zeros(3), None
            for name, values, group, ref in cases:
                if group is None or group != last_group:
                    previous, q_previous = np.zeros(3), None
                e = np.array(values)
                record = dict(name=name, ok=False)
                report["states"].append(record)
                try:
                    row, q, u, pk1 = evaluate(rve, e, controls, previous, q_previous)
                    record.update(row, boundary=boundary_screen(xy, u[rve._eq_map], rims, e, geometry["cell_side"]),
                                  sampled_min_rank_one_curvature=rank_one_screen(e, row["stress"], row["tangent"]))
                    if ref is not None:
                        record["reference_differences"] = differences(ref, row)
                    np.savez_compressed(output / (name+".npz"), strain=e, stress=row["stress"],
                        tangent=row["tangent"], node_displacement=u[rve._eq_map],
                        F_micro=rve.assembler._F, P_micro=pk1, weights=rve.assembler.w_detJ)
                    if name.removeprefix("domain__") in controls["derivative_states"]:
                        record["derivatives"] = [derivative_check(rve, e, q, np.array(row["stress"]),
                                                   np.array(row["tangent"]), step) for step in controls["fd_steps"]]
                    previous, q_previous, last_group = e, q, group
                except Exception as exc:
                    record["error"] = repr(exc)
                    record["attempts"] = getattr(exc, "attempts", record.get("attempts", []))
                    record["ok"] = False
                    report["failures"].append(dict(name=name, error=repr(exc)))
                    last_group = None
                save(report, output)
                print(role, name, record["ok"], flush=True)
            report["numerical_states_complete"] = all(s["ok"] for s in report["states"]) and len(report["states"]) == len(cases)
    report["status"] = "complete"
    report["elapsed_seconds"] = time.perf_counter()-start_time
    report["scope"] = "B-only numerical preflight, not full-domain stability, contact, uniqueness or "
    report["scope"] += "asymptotic mesh-convergence proof. All failed increments remain in attempts."
    save(report, output)
    print("Output:", output, flush=True)
    return 0 if report.get("passed", report.get("numerical_states_complete")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
