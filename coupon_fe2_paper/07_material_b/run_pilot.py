#!/usr/bin/env python3
"""Bounded material-B FOM pilot. No training or modification of material A.

Require a new output directory for each invocation; retain failures.
Loads and screening tolerances are fixed in the specification before solving.
"""
from __future__ import annotations

import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/coupon-b-mpl")

import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "00_rve"))
from geometry import build, cavity_parameters, geometry_checks
import periodic_fom as pf
import config as cfg


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_report(report, output):
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def relative(a, b, floor=1.0):
    return float(np.linalg.norm(np.asarray(a) - b) / max(np.linalg.norm(b), floor))


def field_stats(rve, strain, q):
    a = rve.assembler
    u = rve.T @ q + rve._g(strain)
    _K, rhs = a.Assemble(u)
    determinants = np.linalg.det(a._F)
    stress = np.empty(a._S_voigt.shape[:2] + (2, 2))
    stress[..., 0, 0] = a._S_voigt[..., 0]
    stress[..., 1, 1] = a._S_voigt[..., 1]
    stress[..., 0, 1] = stress[..., 1, 0] = a._S_voigt[..., 2]
    pk1 = a._F @ stress
    norms = np.linalg.norm(pk1, axis=(-2, -1))
    w = a.w_detJ
    force_floor = cfg.MATRIX_YOUNG * rve.thickness * np.sqrt(rve.A0) * 1e-12
    stats = dict(min_micro_J=float(determinants.min()),
                 max_micro_J=float(determinants.max()),
                 pk1_max=float(norms.max()),
                 pk1_l2=float(np.sqrt(np.sum(w * norms**2) / np.sum(w))),
                 relative_reduced_residual=float(np.linalg.norm(rve.T.T @ rhs) /
                                                 max(np.linalg.norm(rhs), force_floor)))
    if not all(np.isfinite(v) for v in stats.values()) or stats["min_micro_J"] <= 0:
        raise RuntimeError("Nonfinite fields or nonpositive microscopic J at quadrature points")
    return stats, u, pk1


def derivative_check(rve, strain, q, stress, tangent, step):
    grad, fd = np.zeros(3), np.zeros((3, 3))
    for j in range(3):
        ep, em = strain.copy(), strain.copy()
        ep[j] += step
        em[j] -= step
        sp, _ = rve.solve(ep, u_ind_init=q, E_start=strain)
        wp = rve.homogenized_energy()
        sm, _ = rve.solve(em, u_ind_init=q, E_start=strain)
        wm = rve.homogenized_energy()
        grad[j] = (wp - wm) / (2 * step)
        fd[:, j] = (sp - sm) / (2 * step)
    return dict(step=step, energy_gradient_relative_error=relative(grad, stress),
                tangent_relative_error=relative(fd, tangent),
                fd_tangent_relative_asymmetry=relative(fd, fd.T),
                energy_gradient=grad.tolist(), fd_tangent=fd.tolist())


def summarize(report):
    spec, checks = report["spec"], []
    tol = spec["screening_tolerances"]
    def add(name, value, key):
        checks.append(dict(check=name, value=float(value), threshold=tol[key],
                           passed=bool(value <= tol[key])))
    for mesh in report["meshes"]:
        name = mesh["name"]
        if not mesh.get("reference"):
            checks.append(dict(check=name + ": initialization/reference", passed=False))
            continue
        ref = mesh["reference"]
        add(name + ": reference stress/E_matrix", np.linalg.norm(ref["stress"])/cfg.MATRIX_YOUNG,
            "reference_stress_over_young")
        add(name + ": reference energy/E_matrix", abs(ref["energy"])/cfg.MATRIX_YOUNG,
            "reference_energy_over_young")
        states = mesh.get("states", [])
        expected = len(spec["paths"]) * len(spec["path_fractions"])
        checks.append(dict(check=name + ": all prescribed states",
                           passed=len(states) == expected and all(s["ok"] for s in states)))
        good = [s for s in states if s["ok"]]
        if good:
            add(name + ": worst equilibrium residual",
                max(s["fields"]["relative_reduced_residual"] for s in good), "relative_reduced_residual")
        derivatives = mesh.get("derivatives", [])
        expected_fd = len(spec["derivative_paths"]) * len(spec["fd_steps"])
        checks.append(dict(check=name + ": all prescribed derivative checks",
                           passed=len(derivatives) == expected_fd and all(d["ok"] for d in derivatives)))
        for key in ("energy_gradient_relative_error", "tangent_relative_error", "fd_tangent_relative_asymmetry"):
            values = [d[key] for d in derivatives if d["ok"]]
            if values:
                add(name + ": worst " + key, max(values), key)
        continuation = mesh.get("continuation", [])
        checks.append(dict(check=name + ": all continuation checks",
                           passed=len(continuation) == len(spec["paths"]) and all(c["ok"] for c in continuation)))
        values = [c["stress_relative_difference"] for c in continuation if c["ok"]]
        if values:
            add(name + ": worst continuation difference", max(values),
                "continuation_stress_relative_difference")
    comparisons = []
    if len(report["meshes"]) == 2:
        coarse, fine = report["meshes"]
        lookup = {(s["path"], s["fraction"]): s for s in fine.get("states", []) if s["ok"]}
        for row in coarse.get("states", []):
            key = row["path"], row["fraction"]
            if not row["ok"] or key not in lookup:
                continue
            ref = lookup[key]
            comparisons.append(dict(path=row["path"], fraction=row["fraction"],
                stress=relative(row["stress"], ref["stress"]),
                tangent=relative(row["tangent"], ref["tangent"]),
                pk1_l2=relative(row["fields"]["pk1_l2"], ref["fields"]["pk1_l2"]),
                pk1_max=relative(row["fields"]["pk1_max"], ref["fields"]["pk1_max"])))
        for key in ("stress", "tangent", "pk1_l2", "pk1_max"):
            if comparisons:
                add("coarse/fine: worst " + key, max(c[key] for c in comparisons),
                    "mesh_" + key + "_relative_difference")
    report["mesh_comparisons"] = comparisons
    report["screening_checks"] = checks
    report["screening_passed"] = bool(checks and all(c["passed"] for c in checks))
    report["scope"] = (
        "Numerical screening at specified states, not proof of stability, uniqueness, "
        "contact absence, or accuracy throughout a strain box. Two meshes show sensitivity, "
        "not asymptotic convergence. Tangent symmetry is checked using unsymmetrized stress "
        "finite differences; the consistent solver tangent is symmetrized internally.")


def plots(report, fields, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Ellipse
    # The shared FOM imports a large-font LaTeX style; isolate diagnostic styling.
    plt.rcParams.update({"text.usetex": False, "font.family": "DejaVu Serif",
        "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "figure.titlesize": 12})
    spec, L = report["spec"], report["spec"]["cell_side"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.add_patch(plt.Rectangle((-L/2, -L/2), L, L, facecolor="#e9eff6", edgecolor="#263648"))
    for i, hole in enumerate(cavity_parameters(spec), 1):
        ax.add_patch(Ellipse(hole["center"], 2*hole["a"], 2*hole["b"], angle=hole["angle"],
                             facecolor="white", edgecolor="#067d8f", lw=1.5))
        ax.text(*hole["center"], str(i), ha="center", va="center", color="#067d8f")
    ax.set(xlim=(-.55*L, .55*L), ylim=(-.55*L, .55*L), xlabel="$X_1$", ylabel="$X_2$",
           title="Material B: fixed pilot candidate (20% void)")
    ax.set_aspect("equal")
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(output / ("geometry." + ext), dpi=180)
    plt.close(fig)
    if not any(m.get("states") for m in report["meshes"]):
        return
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True)
    colors = ("#176ab4", "#dc7b24", "#067d8f")
    for ax, path in zip(axes.ravel(), spec["paths"]):
        for mindex, mesh in enumerate(report["meshes"]):
            rows = [s for s in mesh.get("states", []) if s["path"] == path and s["ok"]]
            for j in range(3):
                ax.plot([0] + [s["fraction"] for s in rows],
                        [0] + [s["stress"][j]/1e6 for s in rows],
                        color=colors[j], ls="--" if mindex == 0 else "-",
                        label=("$S_{11}$", "$S_{22}$", "$S_{12}$")[j] if mindex == 1 else None)
        ax.set_title(path.replace("_", " "), fontsize=9)
        ax.set_xlabel("Ray fraction $t$")
        ax.set_ylabel("Stress [MPa]")
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Material B FOM pilot: dashed coarse / solid fine; prescribed Green-strain rays")
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(output / ("pilot_response." + ext), dpi=160)
    plt.close(fig)
    if fields:
        xy, triangles, displacement, pk1 = fields
        fig, ax = plt.subplots(figsize=(5, 5))
        collection = PolyCollection((xy + displacement)[triangles[:, :3]],
                                    array=np.linalg.norm(pk1, axis=(-2, -1)).mean(axis=1)/1e6,
                                    cmap="viridis", edgecolors="none")
        ax.add_collection(collection)
        ax.autoscale()
        ax.set_aspect("equal")
        ax.set(xlabel="$x_1$", ylabel="$x_2$",
               title="Combined-x endpoint: deformed mesh\nMean Gauss-point $\\|P\\|_F$ [MPa]")
        fig.colorbar(collection, ax=ax)
        fig.tight_layout()
        fig.savefig(output / "fine_combined_field.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=HERE / "pilot_spec.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mesh-only", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "spec.json").write_bytes(args.spec.read_bytes())
    report = dict(spec=spec, spec_sha256=digest(args.spec), status="running",
                  python=platform.python_version(), numpy=np.__version__,
                  started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  geometry=geometry_checks(spec), meshes=[], failures=[])
    sources = [HERE / "geometry.py", Path(__file__), ROOT / "00_rve/periodic_fom.py",
               ROOT / "config.py", ROOT.parent / "RVE_NeoHookean_Homogenization/core/fom_solver_rve.py",
               ROOT.parent / "RVE_NeoHookean_Homogenization/core/StructuralMaterials.json"]
    report["source_sha256"] = {str(p.relative_to(ROOT.parent)): digest(p) for p in sources}
    save_report(report, output)
    start, fields = time.perf_counter(), None
    for mesh_spec in spec["meshes"]:
        name = mesh_spec["name"]
        mesh = dict(name=name, specification=mesh_spec, states=[], derivatives=[], continuation=[])
        report["meshes"].append(mesh)
        base = output / name
        try:
            mesh["geometry"], xy, triangles = build(spec, mesh_spec, base)
            mesh["mesh_sha256"] = digest(base.with_suffix(".mdpa"))
            print(name, "mesh:", mesh["geometry"]["n_elements"], "elements", flush=True)
            save_report(report, output)
            if args.mesh_only:
                continue
            from _material_law_guard_claude import true_neo_hookean_active
            with true_neo_hookean_active():
                rve = pf.PeriodicRVE(base, cell_area=spec["cell_side"]**2)
                a = rve.assembler
                for value, expected in ((a.young, cfg.MATRIX_YOUNG), (a.poisson, cfg.MATRIX_POISSON),
                                        (a.thickness, cfg.MATRIX_THICKNESS)):
                    if not np.allclose(value, expected, atol=0, rtol=0):
                        raise RuntimeError("Material B does not use the declared A matrix properties")
                mesh["n_independent_dofs"] = int(rve.n_ind)
                mesh["n_gauss"] = int(a.n_elems * a.n_gauss)
                mesh["quadrature_porosity"] = float(1 - np.sum(a.w_detJ)/rve.denom)
                stress, tangent, q0 = rve.stress_and_tangent_consistent(np.zeros(3), return_state=True)
                mesh["reference"] = dict(stress=stress.tolist(), tangent=tangent.tolist(),
                    energy=rve.homogenized_energy(), tangent_eigenvalues=np.linalg.eigvalsh(tangent).tolist())
                pf.SUBSTEPS_PER_UNIT_STRAIN = spec["continuation_densities"][0]
                for path, endpoint in spec["paths"].items():
                    prev_e, q = np.zeros(3), q0.copy()
                    endpoint_state = None
                    for fraction in spec["path_fractions"]:
                        strain = fraction * np.array(endpoint)
                        row = dict(path=path, fraction=fraction, strain=strain.tolist(), ok=False)
                        mesh["states"].append(row)
                        try:
                            st = time.perf_counter()
                            stress, tangent, q = rve.stress_and_tangent_consistent(
                                strain, u_ind_init=q, E_start=prev_e, return_state=True)
                            stats, u, pk1 = field_stats(rve, strain, q)
                            row.update(ok=True, stress=stress.tolist(), tangent=tangent.tolist(),
                                energy=rve.homogenized_energy(), fields=stats,
                                elapsed_seconds=time.perf_counter()-st,
                                tangent_eigenvalues=np.linalg.eigvalsh(tangent).tolist())
                            prev_e = strain
                            if fraction == 1:
                                endpoint_state = strain.copy(), q.copy(), stress.copy(), tangent.copy()
                                np.savez_compressed(output / (name + "_" + path + "_endpoint.npz"),
                                    strain=strain, stress=stress, tangent=tangent, q=q, u=u,
                                    F_micro=a._F, P_micro=pk1, weights=a.w_detJ,
                                    node_displacement=u[rve._eq_map])
                                if name == "fine" and path == "combined_x":
                                    fields = xy, triangles, u[rve._eq_map], pk1.copy()
                        except Exception as exc:
                            row["error"] = repr(exc)
                            report["failures"].append(dict(mesh=name, path=path, fraction=fraction,
                                                          error=repr(exc), traceback=traceback.format_exc()))
                            print(name, path, fraction, "FAILED:", exc, flush=True)
                            save_report(report, output)
                            break
                        save_report(report, output)
                    if endpoint_state is None:
                        continue
                    strain, q, stress, tangent = endpoint_state
                    print(name, path, "endpoint S [MPa]:", stress/1e6, flush=True)
                    if path in spec["derivative_paths"]:
                        for step in spec["fd_steps"]:
                            row = dict(path=path, step=step, ok=False)
                            mesh["derivatives"].append(row)
                            try:
                                row.update(derivative_check(rve, strain, q, stress, tangent, step), ok=True)
                            except Exception as exc:
                                row["error"] = repr(exc)
                                report["failures"].append(dict(mesh=name, path=path, stage="derivatives",
                                                              error=repr(exc)))
                    row = dict(path=path, ok=False)
                    mesh["continuation"].append(row)
                    try:
                        # Independent zero-start ramp with twice the continuation density.
                        pf.SUBSTEPS_PER_UNIT_STRAIN = spec["continuation_densities"][1]
                        cold_s, _ = rve.solve(strain, u_ind_init=q0, E_start=np.zeros(3))
                        row.update(ok=True, stress_relative_difference=relative(cold_s, stress),
                                   energy_relative_difference=relative(rve.homogenized_energy(),
                                                                       mesh["states"][-1]["energy"]))
                    except Exception as exc:
                        row["error"] = repr(exc)
                        report["failures"].append(dict(mesh=name, path=path, stage="continuation",
                                                      error=repr(exc)))
                    finally:
                        pf.SUBSTEPS_PER_UNIT_STRAIN = spec["continuation_densities"][0]
                    save_report(report, output)
        except Exception as exc:
            mesh["error"] = repr(exc)
            report["failures"].append(dict(mesh=name, stage="mesh/initialization", error=repr(exc),
                                          traceback=traceback.format_exc()))
            print(name, "FAILED:", exc, flush=True)
            save_report(report, output)
    report["elapsed_seconds"] = time.perf_counter()-start
    report["status"] = "mesh-only complete" if args.mesh_only else "pilot complete"
    if not args.mesh_only:
        summarize(report)
    save_report(report, output)
    plots(report, fields, output)
    print("Output:", output, flush=True)
    print("Failures:", len(report["failures"]), "screening passed:", report.get("screening_passed"), flush=True)
    return 1 if report["failures"] or report.get("screening_passed") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
