#!/usr/bin/env python3
"""B-only combined preflight on cached meshes with explicit saved-state provenance."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import json
import time
from pathlib import Path
import numpy as np
from run_pilot import HERE, ROOT, digest, derivative_check, relative, pf
from run_preflight import evaluate, rim_indices, boundary_screen
from audit_pilot import rank_one_screen


def find_seed(name, role, directories, reports):
    if name == "positive_green_shear":
        kind = "expanded_rays"
        rows = [s for s in reports[kind]["states"] if s["path"] == name and s["level"] == 2]
        filename = name+"__2.npz"
    elif name == "negative_green_shear":
        kind = "original_rays"
        if role == "reference":
            rows = [s for s in reports[kind]["states"] if s["path"] == name and s["fraction"] == 1.]
            filename = name+"_endpoint.npz"
        else:
            rows = [s for s in reports[kind]["states"] if s["name"] == "pilot__"+name+"__1"]
            filename = "pilot__"+name+"__1.npz"
    else:
        kind = "primary"
        rows = [s for s in reports[kind]["states"] if s["name"] == "domain__"+name]
        filename = "domain__"+name+".npz"
    if len(rows) != 1 or not rows[0]["ok"]:
        raise ValueError("Missing, duplicate or failed seed: "+name)
    return rows[0], HERE / directories[kind] / filename


def screen(row, spec):
    b, f = row["boundary"], row["fields"]
    return bool(f["min_micro_J"] > 0 and f["relative_reduced_residual"] <= spec["relative_residual_tolerance"]
        and b["min_polygon_gap"] > spec["stop_polygon_gap"] and not b["self_intersection"]
        and b["max_periodic_jump_error"] < spec["periodic_jump_tolerance"]
        and row["sampled_min_rank_one_curvature"] >= spec["minimum_sampled_rank_one_curvature"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", choices=("reference", "check"), required=True)
    parser.add_argument("--spec", type=Path, default=HERE / "expanded_box_spec.json")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    original_spec = HERE / spec["original_spec"]
    validation_path = HERE / spec["validation"] / "report.json"
    validation = json.loads(validation_path.read_text())
    if not validation["passed"] or validation["spec_sha256"] != digest(original_spec):
        raise ValueError("Original continuation-validation record/spec changed")
    if validation["helper_sha256"] != digest(HERE / "adaptive_continuation.py"):
        raise ValueError("Continuation helper changed since validation")
    directories = spec["source_directories"][args.mesh]
    reports = {kind:json.loads((HERE / directories[kind] / "report.json").read_text())
               for kind in ("primary", "original_rays", "expanded_rays")}
    primary = reports["primary"]
    if primary["status"] != "complete" or primary["spec_sha256"] != digest(original_spec):
        raise ValueError("Require completed original-box preflight on the same specification")
    mesh = HERE / directories["primary"] / directories["mesh_name"]
    mesh_hash = digest(mesh.with_suffix(".mdpa"))
    if any(r["mesh_sha256"] != mesh_hash for r in reports.values()):
        raise ValueError("Seed reports use different meshes")
    original = json.loads(original_spec.read_text())
    parent_path = HERE / original["parent_refinement"] / "report.json"
    if digest(parent_path) != primary["parent_report_sha256"]:
        raise ValueError("Historical geometry parent changed")
    geometry = json.loads(parent_path.read_text())["parent_spec"]
    bounds = np.array([spec["candidate_box"][k] for k in ("E11", "E22", "2E12")])
    seeds = {}
    for name, target in spec["states"].items():
        if not np.all((bounds[:, 0] <= target) & (target <= bounds[:, 1])):
            raise ValueError("Target outside declared candidate box")
        row, path = find_seed(name, args.mesh, directories, reports)
        with np.load(path) as fields:
            if not np.array_equal(fields["strain"], row["strain"]) or not np.array_equal(fields["stress"], row["stress"]):
                raise ValueError("Seed report/field mismatch: "+name)
        if (name in spec["reuse_states"]) != np.array_equal(target, row["strain"]):
            raise ValueError("Reuse flag/strain mismatch: "+name)
        seeds[name] = row, path
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "spec.json").write_bytes(args.spec.read_bytes())
    (output / "run_box_extension_used.py").write_bytes(Path(__file__).read_bytes())
    sources = [Path(__file__), HERE / "adaptive_continuation.py", HERE / "run_preflight.py", HERE / "run_pilot.py",
        HERE / "audit_pilot.py", ROOT / "00_rve/periodic_fom.py", ROOT / "config.py",
        ROOT.parent / "RVE_NeoHookean_Homogenization/core/fom_solver_rve.py",
        ROOT.parent / "RVE_NeoHookean_Homogenization/core/StructuralMaterials.json",
        original_spec, parent_path, validation_path, mesh.with_suffix(".mdpa")]
    sources += [HERE / directories[k] / "report.json" for k in reports]
    sources += [path for _row, path in seeds.values()]
    report = dict(status="plan_only" if args.plan_only else "running", mesh_role=args.mesh,
        spec=spec, spec_sha256=digest(args.spec), mesh_sha256=mesh_hash,
        helper_sha256=digest(HERE / "adaptive_continuation.py"), n_elements=primary["geometry"]["n_elements"],
        sources_sha256={str(p):digest(p) for p in sources}, states=[], cold_checks=[], failures=[], screen_failures=[],
        started_utc=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        plan=[dict(name=name, strain=target, seed_strain=seeds[name][0]["strain"],
                   seed_file=str(seeds[name][1]), reused=name in spec["reuse_states"]) for name, target in spec["states"].items()])
    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    save()
    if args.plan_only:
        print("Validated plan:", args.mesh, len(seeds), "targets;", len(spec["reuse_states"]), "reused", flush=True)
        return 0
    timer = time.perf_counter()
    data = np.load(mesh.with_suffix(".npz"))
    xy = data["xy"]
    rims = rim_indices(geometry, xy)
    from _material_law_guard_claude import true_neo_hookean_active
    with true_neo_hookean_active():
        rve = pf.PeriodicRVE(mesh, cell_area=geometry["cell_side"]**2)
        for name, target in spec["states"].items():
            seed, seed_file = seeds[name]
            record = dict(name=name, strain=target, ok=False,
                          origin="reused" if name in spec["reuse_states"] else "new",
                          seed_strain=seed["strain"], seed_file=str(seed_file))
            report["states"].append(record)
            try:
                if record["origin"] == "reused":
                    for k in ("stress", "tangent", "energy", "fields", "boundary", "sampled_min_rank_one_curvature"):
                        record[k] = seed[k]
                    record.update(ok=True, attempts=[], fields_file=str(seed_file))
                else:
                    with np.load(seed_file) as fields:
                        previous = np.array(fields["strain"])
                        u = np.zeros(rve.n_dof)
                        u[rve._eq_map] = fields["node_displacement"]
                    q = (u-rve._g(previous))[rve.ind_dof]
                    if np.max(abs(rve.T@q+rve._g(previous)-u)) > 1e-10:
                        raise ValueError("Saved seed violates periodic displacement constraints")
                    row, q, u, pk1 = evaluate(rve, np.array(target), spec, previous, q)
                    record.update(row, boundary=boundary_screen(xy, u[rve._eq_map], rims, np.array(target), geometry["cell_side"]),
                                  sampled_min_rank_one_curvature=rank_one_screen(target, row["stress"], row["tangent"]))
                    fields_file = output / (name+".npz")
                    np.savez_compressed(fields_file, strain=target, stress=row["stress"], tangent=row["tangent"],
                        node_displacement=u[rve._eq_map], F_micro=rve.assembler._F,
                        P_micro=pk1, weights=rve.assembler.w_detJ)
                    record["fields_file"] = str(fields_file)
                record["physical_screen_passed"] = screen(record, spec)
                if not record["physical_screen_passed"]:
                    report["screen_failures"].append(name)
                elif name in spec["derivative_states"]:
                    record["derivatives"] = [derivative_check(rve, np.array(target), q,
                        np.array(record["stress"]), np.array(record["tangent"]), h) for h in spec["fd_steps"]]
            except Exception as exc:
                record.update(ok=False, error=repr(exc), attempts=getattr(exc, "attempts", record.get("attempts", [])))
                report["failures"].append(dict(name=name, error=repr(exc)))
            save()
            print(args.mesh, name, record["ok"], record.get("physical_screen_passed"), flush=True)
        for name in spec["cold_check_states"]:
            warm = next(s for s in report["states"] if s["name"] == name)
            cold = dict(name=name, ok=False)
            report["cold_checks"].append(cold)
            if not warm["ok"] or not warm.get("physical_screen_passed"):
                cold["error"] = "Warm target failed; independent cold check not attempted"
                save()
                continue
            try:
                row, _q, u, _pk1 = evaluate(rve, np.array(warm["strain"]), spec)
                with np.load(warm["fields_file"]) as fields:
                    displacement_error = relative(u[rve._eq_map], fields["node_displacement"], 1e-12)
                differences = {k:relative(row[k], warm[k]) for k in ("stress", "tangent", "energy")}
                differences["node_displacement"] = displacement_error
                cold.update(row, differences=differences, passed=all(v <= spec["cold_agreement_tolerance"] for v in differences.values()))
            except Exception as exc:
                cold.update(error=repr(exc), attempts=getattr(exc, "attempts", []))
            save()
            print(args.mesh, "independent zero-start", name, cold["ok"], cold.get("passed"), flush=True)
    report["status"] = "complete"
    report["all_targets_reached"] = len(report["states"]) == len(spec["states"]) and all(s["ok"] for s in report["states"])
    report["elapsed_seconds"] = time.perf_counter()-timer
    report["scope"] = spec["scope"]
    save()
    return 0 if report["all_targets_reached"] and not report["screen_failures"] and all(c.get("passed") for c in report["cold_checks"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
