#!/usr/bin/env python3
"""Additional fine/finer sensitivity check after the original coarse peak failed.

Keep original coarse/fine screening results. Repeat all 32 strain targets on a
third mesh, without training, and compare against the completed fine-mesh pilot.
"""
import argparse
import json
import time
from pathlib import Path
from run_pilot import cfg, digest, field_stats, pf, relative
from geometry import build
import numpy as np

REFINEMENT = dict(name="finer", size_far=0.095, size_hole=0.035)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mesh-from", type=Path, help="Reuse the unchanged finer mesh from an earlier refinement")
    parser.add_argument("--paths", nargs="+", help="Explicit subset of the predeclared paths for a targeted retry")
    parser.add_argument("--density", type=float, help="Explicit continuation refinement; does not change target strains")
    args = parser.parse_args()
    parent = json.loads((args.pilot / "report.json").read_text())
    if parent["status"] != "pilot complete":
        raise RuntimeError("Wait for the full pilot before running this refinement")
    spec = parent["spec"]
    fine = next(m for m in parent["meshes"] if m["name"] == "fine")
    paths = args.paths or list(spec["paths"])
    if not set(paths).issubset(spec["paths"]):
        raise ValueError("Retry paths must belong to the original specification")
    density = args.density or spec["continuation_densities"][0]
    if density <= 0:
        raise ValueError("Continuation density must be positive")
    lookup = {(s["path"], s["fraction"]): s for s in fine["states"] if s["ok"] and s["path"] in paths}
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(parent_report_sha256=digest(args.pilot / "report.json"), mesh=REFINEMENT,
                  parent_spec=spec, status="running", states=[], failures=[],
                  source_sha256=digest(Path(__file__)), paths=paths, continuation_density=density)
    if args.mesh_from:
        previous = json.loads((args.mesh_from / "report.json").read_text())
        if previous["parent_spec"] != spec:
            raise ValueError("Cannot reuse a mesh from a different specification")
        if digest(args.mesh_from / "finer.mdpa") != previous["mesh_sha256"]:
            raise ValueError("Reused mesh hash changed")
        for suffix in (".mdpa", ".npz"):
            (out / ("finer" + suffix)).write_bytes((args.mesh_from / ("finer" + suffix)).read_bytes())
        mesh = np.load(out / "finer.npz")
        xy, triangles = mesh["xy"], mesh["triangles"]
        report["geometry"] = previous["geometry"]
        report["reused_mesh_from"] = str(args.mesh_from)
    else:
        report["geometry"], xy, triangles = build(spec, REFINEMENT, out / "finer")
    report["mesh_sha256"] = digest(out / "finer.mdpa")
    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save()
    start = time.perf_counter()
    from _material_law_guard_claude import true_neo_hookean_active
    with true_neo_hookean_active():
        rve = pf.PeriodicRVE(out / "finer", cell_area=spec["cell_side"]**2)
        pf.SUBSTEPS_PER_UNIT_STRAIN = density
        report["n_independent_dofs"] = int(rve.n_ind)
        report["quadrature_porosity"] = float(1-np.sum(rve.assembler.w_detJ)/rve.denom)
        for path, endpoint in spec["paths"].items():
            if path not in paths:
                continue
            q, prev_e = np.zeros(rve.n_ind), np.zeros(3)
            for fraction in spec["path_fractions"]:
                e = fraction*np.array(endpoint)
                row = dict(path=path, fraction=fraction, strain=e.tolist(), ok=False)
                report["states"].append(row)
                try:
                    stress, tangent, q = rve.stress_and_tangent_consistent(
                        e, u_ind_init=q, E_start=prev_e, return_state=True)
                    stats, u, pk1 = field_stats(rve, e, q)
                    row.update(ok=True, stress=stress.tolist(), tangent=tangent.tolist(),
                               energy=rve.homogenized_energy(), fields=stats)
                    ref = lookup[path, fraction]
                    row["fine_relative_differences"] = dict(
                        stress=relative(ref["stress"], stress), tangent=relative(ref["tangent"], tangent),
                        pk1_l2=relative(ref["fields"]["pk1_l2"], stats["pk1_l2"]),
                        pk1_max=relative(ref["fields"]["pk1_max"], stats["pk1_max"]))
                    prev_e = e
                    if fraction == 1:
                        np.savez_compressed(out / (path + "_endpoint.npz"), strain=e, stress=stress,
                            tangent=tangent, node_displacement=u[rve._eq_map],
                            F_micro=rve.assembler._F, P_micro=pk1, weights=rve.assembler.w_detJ)
                except Exception as exc:
                    row["error"] = repr(exc)
                    report["failures"].append(dict(path=path, fraction=fraction, error=repr(exc)))
                    save()
                    break
                save()
            print("finer", path, "finished; failures:", len(report["failures"]), flush=True)
    report["elapsed_seconds"] = time.perf_counter()-start
    tol = spec["screening_tolerances"]
    good = [s for s in report["states"] if s["ok"]]
    checks = []
    for key in ("stress", "tangent", "pk1_l2", "pk1_max"):
        value = max((s["fine_relative_differences"][key] for s in good), default=None)
        threshold = tol["mesh_" + key + "_relative_difference"]
        checks.append(dict(check="fine/finer: worst " + key, value=value, threshold=threshold,
                           passed=value is not None and value <= threshold))
    residual = max((s["fields"]["relative_reduced_residual"] for s in good), default=None)
    checks.append(dict(check="finer: equilibrium residual", value=residual,
                       threshold=tol["relative_reduced_residual"],
                       passed=residual is not None and residual <= tol["relative_reduced_residual"]))
    report["checks"] = checks
    expected = len(paths)*len(spec["path_fractions"])
    report["passed"] = (len(good) == len(lookup) == expected and not report["failures"]
                        and all(c["passed"] for c in checks))
    report["status"] = "complete"
    report["scope"] = "Additional numerical mesh sensitivity at all pilot targets. Peak/L2 stress "
    report["scope"] += "statistics are sampled functionals, not pointwise field error estimates; "
    report["scope"] += "no exact solution, stability certificate or full-domain validation."
    save()
    print("Fine/finer screening passed:", report["passed"], flush=True)
    print(checks, flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
