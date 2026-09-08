#!/usr/bin/env python3
"""Clean direct PANN--FE2 coupon runs with an energy-consistent tangent.

The default is the validation-selected, flexible 32-feature ICNN.  It is the
corrected polyconvex coupon model, not the historical 15-feature checkpoint.
``--tier free`` instead runs the separately saved, four-feature unconstrained
energy PANN under exactly the same macro conditions.  ``--tier regression``
runs the three-feature direct-stress baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJ = ROOT.parent / "RVE_NeoHookean_Homogenization"
PANN = ROOT / "06_pann"
for candidate in (ROOT, ROOT / "00_rve", ROOT / "01_macro_prepass", PANN,
                  PROJ / "fe2_extension", PROJ / "core"):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)
_kratos_candidates = (
    Path("/home/sares/Kratos_Eigen_Check/bin/Release"),
    Path("/home/kratos/Kratos_Eigen_Check/bin/Release"),
)
KRATOS_PATH = next((p for p in _kratos_candidates if p.is_dir()), _kratos_candidates[0])
if str(KRATOS_PATH) not in sys.path:
    sys.path.append(str(KRATOS_PATH))

from flexible_pann_law import FlexiblePANNLaw, finite_difference_tangent_check, sha256
from free_pann_law import CouponFreePANNLaw, finite_difference_tangent_check as free_tangent_check
from regression_pann_law import CouponRegressionLaw, finite_difference_tangent_check as regression_tangent_check
from run_fom_fe2 import _canonical_meshes, _coverage, _force_vector, _minimal_rbm_constraints


DEFAULT_ICNN = PANN / "enrichment_results" / "flex_icnn_dynamic32_seed16" / "model.pt"
DEFAULT_FREE = PANN / "pann_free.pt"
DEFAULT_REGRESSION = PANN / "pann_regression.pt"


def _configure_torch(threads: int) -> None:
    # Leave Kratos/BLAS at one thread in the launch environment.  PyTorch owns
    # exactly this one direct surrogate evaluation and may use the CPU threads.
    import torch
    torch.set_num_threads(int(threads))
    torch.set_num_interop_threads(1)


def run(args) -> int:
    _configure_torch(args.threads)
    defaults = dict(flexible=DEFAULT_ICNN, free=DEFAULT_FREE, regression=DEFAULT_REGRESSION)
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else defaults[args.tier]
    if args.tier == "free":
        law = CouponFreePANNLaw(checkpoint)
        tangent_check = free_tangent_check
    elif args.tier == "regression":
        law = CouponRegressionLaw(checkpoint)
        tangent_check = regression_tangent_check
    else:
        law = FlexiblePANNLaw(checkpoint)
        tangent_check = finite_difference_tangent_check
    if args.tier == "flexible" and law.metadata["core"] != "icnn" and not args.allow_non_icnn:
        raise RuntimeError("this ICNN run requires an ICNN checkpoint; use --allow-non-icnn explicitly")

    # Tangent preflight uses three relevant in-box states.  This validation is
    # outside the timed macro solve and does not tune or alter the checkpoint.
    tangent_audit = tangent_check(
        law, np.array(((0.0, 0.0, 0.0), (0.05, -0.025, -0.04),
                       (0.14, -0.055, 0.04))), h=1.0e-6,
    )
    if tangent_audit["tangent_fd_relative"] > 3.0e-5:
        raise RuntimeError(f"PANN tangent preflight failed: {tangent_audit}")
    if args.tier != "regression" and tangent_audit["tangent_asymmetry_relative"] > 1.0e-10:
        raise RuntimeError(f"PANN tangent lost energy symmetry: {tangent_audit}")

    macro_base, rve_base = _canonical_meshes(args.macro_divisor)
    import fom_solver_rve as fom
    from macro_prepass import MacroCoupon

    macro = MacroCoupon(macro_base, verbose=True)
    f_final = _force_vector(macro, args.force)
    fixed = _minimal_rbm_constraints(macro)
    free_mask = np.ones(macro.n_dof, dtype=bool)
    free_mask[fixed] = False
    free = np.flatnonzero(free_mask)
    grid = np.load(ROOT / "02_sampling" / "train_grid.npz")
    blo, bhi = np.asarray(grid["blo"]), np.asarray(grid["bhi"])
    u = np.zeros(macro.n_dof)
    records, cloud = [], []
    cached_assembly = None
    original_law = fom._neo_hookean_pk2_2d_vectorized
    t0 = time.perf_counter()  # deployed checkpoint and macro setup excluded

    try:
        fom._neo_hookean_pk2_2d_vectorized = law
        for step in range(1, args.max_steps + 1):
            f_ext = f_final * (step / args.n_steps)
            res0, converged = None, False
            best = (np.inf, u.copy())
            for it in range(1, args.max_newton + 1):
                tic = time.perf_counter()
                if cached_assembly is None:
                    K, rhs = macro.assembler.Assemble(u)
                else:
                    K, rhs = cached_assembly
                    cached_assembly = None
                residual = rhs + f_ext
                res = float(np.linalg.norm(residual[free]))
                res0 = max(res, 1.0e-30) if res0 is None else res0
                rel = res / res0
                best = min(best, (res, u.copy()), key=lambda item: item[0])
                print(f"  step {step:2d}/{args.n_steps} iter {it:2d}: |R|={res:.5e}, "
                      f"rel={rel:.3e}, wall={time.perf_counter() - tic:.2f}s", flush=True)
                if res < args.abs_tol or rel < args.rel_tol:
                    converged = True
                    break
                du = spsolve(K[free, :][:, free].tocsc(), residual[free])
                if not np.all(np.isfinite(du)):
                    raise RuntimeError(f"non-finite macro update at step {step}, iteration {it}")
                u[free] += du
            if not converged:
                u = best[1]
                raise RuntimeError(f"macro Newton did not converge at step {step}; "
                                   f"best relative residual={best[0] / res0:.3e}")

            E = macro.assembler._E_voigt.reshape(-1, 3).copy()
            cov = _coverage(E, blo, bhi)
            records.append(dict(step=step, iterations=it, residual=res,
                                relative_residual=rel, coverage=cov))
            cloud.append(E)
            cached_assembly = (K, rhs)
            print(f"    converged: in-box {cov['inside']}/{cov['n']}; "
                  f"E11=[{cov['minimum'][0]:+.4f},{cov['maximum'][0]:+.4f}], "
                  f"E22=[{cov['minimum'][1]:+.4f},{cov['maximum'][1]:+.4f}], "
                  f"g12=[{cov['minimum'][2]:+.4f},{cov['maximum'][2]:+.4f}]", flush=True)

        wall = time.perf_counter() - t0
        E_final = macro.assembler._E_voigt.reshape(-1, 3).copy()
        S_final = macro.assembler._S_voigt.reshape(-1, 3).copy()
        final_response = law.response(E_final, tangent=False)
        u_nodes = np.stack((u[macro.eq_map[:, 0]], u[macro.eq_map[:, 1]]), axis=1)
        tag = args.tag or f"{args.tier}_w{args.macro_divisor:g}_f{args.force / 1e3:g}kn"
        out = HERE / f"pann_fe2_{tag}.npz"
        summary = dict(
            status="converged",
            model=dict(
                flexible="flexible_polyconvex_pann",
                free="free_energy_pann",
                regression="direct_stress_regression",
            )[args.tier],
            tier=args.tier, tag=tag,
            output=str(out), wall_seconds=wall, torch_threads=args.threads,
            force_per_end=args.force, n_steps_requested=args.n_steps,
            n_steps_completed=len(records), macro_elements=macro.assembler.n_elems,
            macro_gauss_points=macro.assembler.n_elems * macro.assembler.n_gauss,
            rve_elements_reference=int(np.load(ROOT / "03_data" / "data.npz")["n_elements"]),
            checkpoint=law.metadata, tangent_preflight=tangent_audit,
            macro_newton=records, material_calls=law.calls,
            material_points=law.points, summed_pann_seconds=law.seconds,
            macro_mesh_sha256=sha256(str(macro_base) + ".mdpa"),
            rve_mesh_sha256=sha256(str(rve_base) + ".mdpa"),
        )
        if not args.no_output:
            output_data = dict(
                coords=macro.xy, connectivity=macro.assembler.connectivity,
                u_nodal=u_nodes, E_final=E_final, S_final=S_final,
                E_path=np.concatenate(cloud, axis=0), force_per_end=args.force,
                n_steps_requested=args.n_steps, n_steps_completed=len(records),
                torch_threads=args.threads,
            )
            if "energy" in final_response:
                output_data["W_final"] = final_response["energy"]
            np.savez_compressed(out, **output_data)
            out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        return 0
    finally:
        fom._neo_hookean_pk2_2d_vectorized = original_law


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("flexible", "free", "regression"), default="flexible")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--allow-non-icnn", action="store_true")
    # The ICNN's 32-by-24-by-24 batches are small.  A local timing sweep on
    # the coupon's 2070-point batch found the optimum at four CPU threads;
    # using all hardware threads adds OpenMP scheduling overhead instead.
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--macro-divisor", type=float, default=4.0)
    parser.add_argument("--force", type=float, default=1.0e5)
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-newton", type=int, default=12)
    parser.add_argument("--rel-tol", type=float, default=1.0e-7)
    parser.add_argument("--abs-tol", type=float, default=1.0e-5)
    parser.add_argument("--tag", default="")
    parser.add_argument("--no-output", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.n_steps < 1 or not 1 <= args.max_steps <= args.n_steps:
        parser.error("require 1 <= --max-steps <= --n-steps")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
