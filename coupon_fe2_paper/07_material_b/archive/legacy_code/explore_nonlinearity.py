#!/usr/bin/env python3
"""Bounded B-only exploration; reuse unchanged solver, geometry and cached meshes."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import json
import time
from pathlib import Path
import numpy as np
from run_preflight import evaluate, rim_indices, boundary_screen
from run_pilot import HERE, ROOT, digest, derivative_check, pf
from audit_pilot import rank_one_screen


def nonlinearity(row, D0):
    e, s, D = [np.array(row[k]) for k in ("strain", "stress", "tangent")]
    linear_s, linear_w = D0@e, float(e@D0@e/2)
    unit = e/np.linalg.norm(e)
    return dict(linear_stress=linear_s.tolist(), linear_energy=linear_w,
        stress_linear_deviation=float(np.linalg.norm(s-linear_s)/max(np.linalg.norm(s), 1.)),
        tangent_change=float(np.linalg.norm(D-D0)/np.linalg.norm(D0)),
        energy_linear_deviation=float(abs(row["energy"]-linear_w)/max(abs(row["energy"]), 1.)),
        directional_tangent_ratio=float(unit@D@unit/(unit@D0@unit)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", choices=("reference", "check"), required=True)
    parser.add_argument("--spec", type=Path, default=HERE / "nonlinear_exploration_spec.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    validation_path = HERE / spec["validation"] / "report.json"
    validation = json.loads(validation_path.read_text())
    if not validation["passed"] or validation["helper_sha256"] != digest(HERE / "adaptive_continuation.py"):
        raise ValueError("Require the previously validated, unchanged continuation helper")
    parent_path = HERE / spec["parent_reference"] / "report.json"
    parent = json.loads(parent_path.read_text())
    geometry = parent["parent_spec"]
    source = HERE / spec["parent_reference" if args.mesh == "reference" else "parent_check"]
    source_report = json.loads((source / "report.json").read_text())
    mesh = source / ("finer" if args.mesh == "reference" else "check")
    if digest(mesh.with_suffix(".mdpa")) != source_report["mesh_sha256"]:
        raise ValueError("Cached mesh changed")
    original = {}
    for row in source_report["states"]:
        if not row["ok"]:
            continue
        if args.mesh == "reference":
            original[row["path"], row["fraction"]] = row
        elif row["name"].startswith("pilot__"):
            _prefix, path, fraction = row["name"].split("__")
            original[path, float(fraction)] = row
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "spec.json").write_bytes(args.spec.read_bytes())
    sources = [Path(__file__), HERE / "adaptive_continuation.py", HERE / "run_preflight.py",
               HERE / "run_pilot.py", HERE / "audit_pilot.py", ROOT / "00_rve/periodic_fom.py",
               ROOT / "config.py", ROOT.parent / "RVE_NeoHookean_Homogenization/core/fom_solver_rve.py",
               parent_path, source / "report.json", mesh.with_suffix(".mdpa"), validation_path]
    report = dict(status="running", mesh_role=args.mesh, spec=spec, spec_sha256=digest(args.spec),
        mesh_sha256=source_report["mesh_sha256"], n_elements=source_report["geometry"]["n_elements"],
        geometry=geometry, sources_sha256={str(p):digest(p) for p in sources},
        baseline=[], states=[], failures=[])
    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    save()
    timer = time.perf_counter()
    mesh_data = np.load(mesh.with_suffix(".npz"))
    xy = mesh_data["xy"]
    rims = rim_indices(geometry, xy)
    from _material_law_guard_claude import true_neo_hookean_active
    with true_neo_hookean_active():
        rve = pf.PeriodicRVE(mesh, cell_area=geometry["cell_side"]**2)
        s0, D0, _q0 = rve.stress_and_tangent_consistent(
            np.zeros(3), u_ind_init=np.zeros(rve.n_ind), E_start=np.zeros(3), return_state=True)
        report["reference"] = dict(stress=s0.tolist(), tangent=D0.tolist(), energy=rve.homogenized_energy())
        if np.linalg.norm(s0) > 1e-3 or abs(report["reference"]["energy"]) > 1e-3:
            raise ValueError("Unexpected nonzero reference stress or energy")
        for path, targets in spec["paths"].items():
            for fraction in geometry["path_fractions"]:
                row = original[path, fraction]
                report["baseline"].append(dict(row, path=path, nonlinearity=nonlinearity(row, D0)))
            seed = original[path, 1.]
            seed_path = source / ((path+"_endpoint") if args.mesh == "reference" else ("pilot__"+path+"__1"))
            with np.load(seed_path.with_suffix(".npz")) as fields:
                previous = np.array(fields["strain"])
                if not np.array_equal(previous, seed["strain"]):
                    raise ValueError("Seed strain differs from report")
                u = np.zeros(rve.n_dof)
                u[rve._eq_map] = fields["node_displacement"]
                q = (u-rve._g(previous))[rve.ind_dof]
                if np.max(abs(rve.T@q+rve._g(previous)-u)) > 1e-10:
                    raise ValueError("Saved seed does not satisfy periodic constraints")
            report["sources_sha256"][str(seed_path.with_suffix(".npz"))] = digest(seed_path.with_suffix(".npz"))
            for index, target in enumerate(targets):
                record = dict(path=path, level=index+1, strain=target, ok=False)
                report["states"].append(record)
                try:
                    row, trial_q, trial_u, pk1 = evaluate(rve, np.array(target), spec, previous, q)
                    boundary = boundary_screen(xy, trial_u[rve._eq_map], rims, np.array(target), geometry["cell_side"])
                    lh = rank_one_screen(target, row["stress"], row["tangent"])
                    record.update(row, boundary=boundary, sampled_min_rank_one_curvature=lh,
                                  nonlinearity=nonlinearity(row, D0))
                    screen = (boundary["min_polygon_gap"] > spec["stop_polygon_gap"]
                              and not boundary["self_intersection"] and boundary["max_periodic_jump_error"] < 1e-8
                              and lh >= spec["stop_sampled_rank_one_curvature"]
                              and row["fields"]["relative_reduced_residual"] <= spec["stop_relative_residual"])
                    record["physical_screen_passed"] = bool(screen)
                    np.savez_compressed(output / f"{path}__{index+1}.npz", strain=target,
                        stress=row["stress"], tangent=row["tangent"], node_displacement=trial_u[rve._eq_map],
                        F_micro=rve.assembler._F, P_micro=pk1, weights=rve.assembler.w_detJ)
                    if not screen:
                        record["stop_reason"] = "Failed sampled determinant/residual/boundary/rank-one screen"
                        break
                    if path == spec["derivative_path"] and index == len(targets)-1:
                        record["derivatives"] = [derivative_check(rve, np.array(target), trial_q,
                            np.array(row["stress"]), np.array(row["tangent"]), h) for h in spec["fd_steps"]]
                    previous, q = np.array(target), trial_q
                except Exception as exc:
                    record["ok"] = False
                    record["error"] = repr(exc)
                    record["attempts"] = getattr(exc, "attempts", record.get("attempts", []))
                    report["failures"].append(dict(path=path, level=index+1, error=repr(exc)))
                    break
                finally:
                    save()
                    print(args.mesh, path, index+1, record["ok"], record.get("nonlinearity"), flush=True)
        report["status"] = "complete"
        report["all_targets_reached_and_screened"] = (len(report["states"]) == sum(map(len, spec["paths"].values()))
            and all(s["ok"] and s.get("physical_screen_passed", False) for s in report["states"]))
    report["elapsed_seconds"] = time.perf_counter()-timer
    report["scope"] = spec["scope"]
    save()
    return 0 if report["all_targets_reached_and_screened"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
