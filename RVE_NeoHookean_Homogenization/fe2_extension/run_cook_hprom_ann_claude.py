#!/usr/bin/env python3
"""Stage 3: Cook's membrane driven by GENUINE FE^2 -- at every macro
Gauss point, query the extracted D-HPROM-ANN (Stage 1) or iterative
HPROM-ANN (Stage 2) law instead of the trained PANN. Same
Total-Lagrangian Newton-Raphson driver as Cook.gid/run_cook_pann_claude.py
(mesh construction, load application, line search all imported
read-only from there -- they are material-law-agnostic, nothing there
is modified). Results are written only under fe2_extension/, never into
Cook.gid/, so the already-reported PANN results stay untouched.

Cost warning (see the staged plan): each D-HPROM-ANN/HPROM-ANN
evaluate() call rebuilds nothing (the law is built once, per
get_law()), but still costs far more than a PANN forward pass. Start
small (nx=4 or nx=6) and confirm wall-clock time before scaling up.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COOK_DIR = ROOT / "Cook.gid"
for p in (str(ROOT / "core"), str(COOK_DIR), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import fom_solver_rve as fom  # noqa: E402
import KratosMultiphysics as KM  # noqa: E402

from build_cook_mesh_claude import build_mesh  # noqa: E402
from run_cook_pann_claude import (  # noqa: E402  (read-only reuse, material-law-agnostic)
    build_model_part,
    consistent_edge_force,
    _line_search_alpha,
)

import dhprom_ann_direct_law_claude as dhprom_module  # noqa: E402
import hprom_ann_iterative_law_claude as hprom_iter_module  # noqa: E402
import dhprom_ann_direct_law_float64_claude as dhprom_f64_module  # noqa: E402
import hprom_ann_iterative_law_float64_claude as hprom_iter_f64_module  # noqa: E402
import pann_constitutive_law_claude as pann_law  # noqa: E402  (read-only reuse)


def _make_pann_material_func(which):
    """Wraps pann_law.pann_pk2_2d_vectorized(e_voigt, which) to match the
    (E_flat, young, poisson) -> (S, CC) contract MATERIAL_FUNCS expects
    (young/poisson accepted but unused, signature compatibility only) --
    same pattern run_cook_pann_claude.py's own run_newton() uses via its
    _pann_material closure."""
    def _f(e_voigt, young=None, poisson=None):
        return pann_law.pann_pk2_2d_vectorized(e_voigt, which=which)
    return _f

EDGE_LENGTH = 16.0
LINE_LOAD_MODULUS_FINAL = 13_000_000.0
TOTAL_FORCE_FINAL = LINE_LOAD_MODULUS_FINAL * EDGE_LENGTH
N_STEPS = 20
RESIDUAL_REL_TOL = 1.0e-4
RESIDUAL_ABS_TOL = 1.0e-9

# A step whose BEST iterate reduced the relative residual below this (but
# never reached RESIDUAL_REL_TOL) is "stalled" -- accepted and advanced
# past, not aborted. Above this, "diverged" -- the Newton iteration never
# got anywhere close, so continuing to build on it would be meaningless
# even though the code still technically can (see run_newton_fe2).
STALL_ACCEPT_REL_TOL = 1.0e-2
# Early-exit stall detector: once at least 2*STALL_WINDOW iterations have
# run, compare the best residual in the most recent STALL_WINDOW iterations
# against the best residual in the STALL_WINDOW before that; if the
# improvement is under STALL_PATIENCE_REL, the residual has plateaued and
# there is no point grinding on to max_newton_iter. STALL_WINDOW=2 (not a
# larger window) is deliberate: this residual oscillates on its plateau
# rather than settling to a fixed point (unlike a monotone-then-flat
# signal, where comparing single consecutive iterations -- as
# burgers2d-rom-workbench/burgers/gauss_newton.py's _relative_drop does --
# would work directly), so a single-point comparison is too noise-prone,
# but there is no reason to wait for a full 6-iteration window either. A
# window of 2 is the smallest choice that is still a min-over-2 (not a
# single raw point) while triggering as early as iteration 4.
STALL_WINDOW = 2
STALL_PATIENCE_REL = 0.10

MATERIAL_FUNCS = {
    "dhprom": dhprom_module.dhprom_ann_pk2_2d_vectorized,
    "dhprom_consistent": dhprom_module.dhprom_ann_pk2_2d_vectorized_consistent,
    "hprom_iterative": hprom_iter_module.hprom_ann_iterative_pk2_2d_vectorized,
    "hprom_iterative_consistent": hprom_iter_module.hprom_ann_iterative_pk2_2d_vectorized_consistent,
    "pann_certified": _make_pann_material_func("certified"),
    "pann_free": _make_pann_material_func("free"),
    "pann_ickan": _make_pann_material_func("ickan"),
    "pann_regression": _make_pann_material_func("regression"),
    # float64 decoder precision (see dhprom_ann_direct_law_float64_claude.py /
    # hprom_ann_iterative_law_float64_claude.py for why): resolves the small-strain
    # cancellation in q_s_final = q_s_final_map - N0_const - J0_const@q_p that the
    # float32 originals above carry at Cook's actual (tiny) operating strains.
    "dhprom_f64": dhprom_f64_module.dhprom_ann_pk2_2d_vectorized_float64,
    "dhprom_f64_consistent": dhprom_f64_module.dhprom_ann_pk2_2d_vectorized_consistent_float64,
    "hprom_iterative_f64": hprom_iter_f64_module.hprom_ann_iterative_pk2_2d_vectorized_float64,
    "hprom_iterative_f64_consistent": hprom_iter_f64_module.hprom_ann_iterative_pk2_2d_vectorized_consistent_float64,
}


def run_newton_fe2(
    which: str, nx: int = 4, ny: int = 4, verbose: bool = True, diagnose: bool = False,
    use_line_search: bool = True, save_npz: bool = True, max_newton_iter: int = 30,
    n_steps: int = N_STEPS, total_force_final: float = TOTAL_FORCE_FINAL,
    stall_accept_rel_tol: float = STALL_ACCEPT_REL_TOL,
    stall_window: int = STALL_WINDOW, stall_patience_rel: float = STALL_PATIENCE_REL,
):
    if which not in MATERIAL_FUNCS:
        raise ValueError(f"which must be one of {list(MATERIAL_FUNCS)}, got {which!r}")
    mp, coords, tris, left_nodes, right_nodes = build_model_part(nx, ny)
    n_dof, eq_map, ta = fom.SetUpDofEquationIdsAndDisplacementAdaptor(mp)

    fom._neo_hookean_pk2_2d_vectorized = MATERIAL_FUNCS[which]
    assembler = fom.VectorizedAssembler(mp, n_dof, eq_map, log_label=f"CookFE2[{which}]")

    f_unit = consistent_edge_force(coords, right_nodes)
    f_unit_eq = np.zeros(n_dof)
    np.add.at(f_unit_eq, eq_map[:, 0], f_unit[:, 0])
    np.add.at(f_unit_eq, eq_map[:, 1], f_unit[:, 1])

    free_dofs = np.array([d for d in range(n_dof)
                           if d not in set(eq_map[left_nodes, 0]) | set(eq_map[left_nodes, 1])])

    u = np.zeros(n_dof)
    step_log = []
    residual_histories = {}
    u_step1 = None
    t_wall_start = time.perf_counter()
    n_material_calls = 0

    for step in range(1, n_steps + 1):
        load_factor = total_force_final * step / n_steps
        f_ext = f_unit_eq * load_factor

        converged = False
        stall_detected = False
        n_iter = 0
        res_norm0 = None
        res_history = []
        best_u, best_res = u.copy(), np.inf
        for it in range(1, max_newton_iter + 1):
            n_iter = it
            t0 = time.perf_counter()
            K, rhs_int = assembler.Assemble(u)
            n_material_calls += 1
            residual = rhs_int + f_ext
            res_free = residual[free_dofs]
            res_norm = np.linalg.norm(res_free)
            res_history.append(res_norm)
            if res_norm0 is None:
                res_norm0 = max(res_norm, 1e-12)
            if np.isfinite(res_norm) and res_norm < best_res:
                best_res, best_u = float(res_norm), u.copy()
            if verbose:
                print(f"      iter {it:2d}  |res|={res_norm:.6e}  |res|/|res0|={res_norm / res_norm0:.6e}"
                      f"  ({time.perf_counter() - t0:.2f}s)")
            if res_norm < RESIDUAL_ABS_TOL or res_norm / res_norm0 < RESIDUAL_REL_TOL:
                converged = True
                break
            if it >= 2 * stall_window:
                recent_best = min(res_history[-stall_window:])
                prior_best = min(res_history[-2 * stall_window:-stall_window])
                if recent_best >= (1.0 - stall_patience_rel) * prior_best:
                    stall_detected = True
                    if verbose:
                        print(f"      [{which}] step {step}: stall detected at iter {it} "
                              f"(recent_best={recent_best:.6e}, prior_best={prior_best:.6e}); "
                              f"accepting best iterate and moving on")
                    break

            K_ff = K[free_dofs, :][:, free_dofs]
            try:
                du_free = spsolve(K_ff.tocsc(), res_free)
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"  [{which}] step {step}: linear solve failed at iter {it}: {exc}")
                break
            if not np.all(np.isfinite(du_free)):
                if verbose:
                    print(f"  [{which}] step {step}: non-finite update at iter {it}")
                break

            alpha = 1.0
            if use_line_search:
                alpha = _line_search_alpha(assembler, u, du_free, free_dofs, f_ext)
                n_material_calls += 2  # _line_search_alpha's own two eval_r() Assemble() calls (typical)
            u[free_dofs] += alpha * du_free

        best_rel = best_res / res_norm0 if np.isfinite(best_res) else np.inf
        if converged:
            status = "converged"
        elif np.isfinite(best_rel) and best_rel < stall_accept_rel_tol:
            status = "stalled"
        else:
            status = "diverged"
        if not converged:
            if not np.all(np.isfinite(best_u)):
                raise RuntimeError(
                    f"[{which}] step {step}: no finite iterate was ever produced -- cannot continue."
                )
            u = best_u.copy()

        residual_histories[step] = np.array(res_history)
        if step == 1:
            u_step1 = u.copy()
        step_tip_uy = u[eq_map[right_nodes, 1]]
        step_log.append({
            "step": step, "load": load_factor, "iters": n_iter, "converged": converged,
            "status": status, "best_rel": float(best_rel), "stall_detected_early": stall_detected,
            "tip_uy_min": float(np.min(step_tip_uy)), "tip_uy_max": float(np.max(step_tip_uy)),
        })
        if verbose:
            tag = {"converged": "OK", "stalled": "STALLED (accepted best iterate, advancing)",
                   "diverged": "DIVERGED (accepted best iterate, advancing)"}[status]
            elapsed = time.perf_counter() - t_wall_start
            print(f"  [{which}] step {step:2d}  load={load_factor:.3e}  iters={n_iter:2d}  {tag}  "
                  f"best_rel={best_rel:.3e}  (elapsed={elapsed:.1f}s, material calls so far~{n_material_calls})")
        # Never abort the ramp: always advance with the best available iterate. A
        # step's own imperfection is exactly what this exploratory driver is meant
        # to characterize (does it stay bounded across later, larger-load steps, or
        # compound/worsen?), not something to hide by stopping early.

    fom.SetDisplacementFromEquationVector(u, eq_map, ta)
    e_voigt, s_voigt = assembler.ComputeStrainStressOnly(u)
    e_flat = e_voigt.reshape(-1, 3).copy()
    s_flat = s_voigt.reshape(-1, 3).copy()

    tip_uy = [mp.GetNode(nid + 1).GetSolutionStepValue(KM.DISPLACEMENT_Y) for nid in right_nodes]
    u_nodal = np.stack([u[eq_map[:, 0]], u[eq_map[:, 1]]], axis=1)
    fully_converged = all(s["converged"] for s in step_log) and len(step_log) == n_steps
    ever_diverged = any(s["status"] == "diverged" for s in step_log)
    t_wall_total = time.perf_counter() - t_wall_start

    if verbose:
        print(f"\n  [{which}] per-step status: "
              + ", ".join(f"{s['step']}:{s['status'][0].upper()}" for s in step_log))

    if save_npz:
        np.savez(
            HERE / f"cook_results_{which}_claude.npz",
            coords=coords, tris=tris, u_nodal=u_nodal, e_gp=e_flat, s_gp=s_flat,
            iters_per_step=np.array([s["iters"] for s in step_log]),
            converged_per_step=np.array([s["converged"] for s in step_log]),
            status_per_step=np.array([s["status"] for s in step_log]),
            best_rel_per_step=np.array([s["best_rel"] for s in step_log]),
            stall_detected_early_per_step=np.array([s["stall_detected_early"] for s in step_log]),
            tip_uy_min_per_step=np.array([s["tip_uy_min"] for s in step_log]),
            tip_uy_max_per_step=np.array([s["tip_uy_max"] for s in step_log]),
            load_per_step=np.array([s["load"] for s in step_log]),
            residual_history_step1=residual_histories.get(1, np.zeros(0)),
            fully_converged=fully_converged,
            ever_diverged=ever_diverged,
        )

    print(f"  [{which}] TOTAL wall time={t_wall_total:.1f}s, material calls~{n_material_calls}, "
          f"avg {t_wall_total / max(n_material_calls, 1):.3f}s/call")

    return {
        "which": which,
        "step_log": step_log,
        "e11_range": (float(e_flat[:, 0].min()), float(e_flat[:, 0].max())),
        "gamma12_range": (float(e_flat[:, 2].min()), float(e_flat[:, 2].max())),
        "tip_uy_range": (float(np.min(tip_uy)), float(np.max(tip_uy))),
        "fully_converged": fully_converged,
        "ever_diverged": ever_diverged,
        "wall_time": t_wall_total,
        "n_material_calls": n_material_calls,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--which", default="dhprom", choices=list(MATERIAL_FUNCS))
    p.add_argument("--nx", type=int, default=4)
    p.add_argument("--ny", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=N_STEPS)
    p.add_argument("--max-newton-iter", type=int, default=30)
    a = p.parse_args()

    res = run_newton_fe2(
        a.which, nx=a.nx, ny=a.ny, n_steps=a.n_steps, max_newton_iter=a.max_newton_iter,
        verbose=True, use_line_search=True, save_npz=True,
    )
    print("\n=== summary ===")
    print(f"{res['which']:14s}: fully_converged={res['fully_converged']}, ever_diverged={res['ever_diverged']}, "
          f"E11={res['e11_range']}, gamma12={res['gamma12_range']}, tip_uy={res['tip_uy_range']}")
    for s in res["step_log"]:
        print(f"    step {s['step']:2d}  load={s['load']:.3e}  status={s['status']:9s}  "
              f"iters={s['iters']:2d}  best_rel={s['best_rel']:.3e}  "
              f"tip_uy=({s['tip_uy_min']:.4f},{s['tip_uy_max']:.4f})  "
              f"early_stall_detect={s['stall_detected_early']}")
