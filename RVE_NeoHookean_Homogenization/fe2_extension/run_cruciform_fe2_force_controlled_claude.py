#!/usr/bin/env python3
"""Force-controlled counterpart to run_cruciform_fe2_claude.py's
run_newton_fe2_cruciform, built to test a specific hypothesis: the
uncertified PANN tiers (Regression, Free hyperelastic) show no
convergence trouble anywhere across a wide displacement-controlled
sweep (magnitude up to 4x the standard protocol, full asymmetric/
uniaxial/mixed tension-compression directions) NOR any local rank-one
convexity violation across those same states -- but Cook's membrane
(force-controlled) makes both of them stall every step. A limit point
(the global force-displacement curve turning over, tangent stiffness
losing positive-definiteness at the SYSTEM level at some load) is
invisible to both of those checks by construction: it's a property of
the force-controlled equilibrium path, not of the material's pointwise
response, and doesn't require any local material defect. This driver
lets that mechanism actually manifest, if present, by prescribing a
ramped total force at each arm tip instead of a ramped displacement.

Loading: the same 4 tips as the displacement-controlled protocol, each
now force- rather than displacement-driven, in the SAME outward
direction (px:+x, mx:-x, py:+y, my:-y) via a consistent (Simpson-
weighted) nodal load exactly like Cook's own consistent_edge_force,
applied along each tip's own (corner-mid-corner-mid-corner) edge --
verified this session to already have the right node ordering for that
formula. Only the center node is Dirichlet-fixed (both components);
every tip dof is free, in equilibrium with the applied force instead of
prescribed.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve

HERE = Path("/home/kratos/ML_assisted_CLs_clean/RVE_NeoHookean_Homogenization/fe2_extension")
ROOT = HERE.parent
COOK_DIR = ROOT / "Cook.gid"
for p in (str(ROOT / "core"), str(COOK_DIR), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import fom_solver_rve as fom  # noqa: E402
from run_cook_pann_claude import consistent_edge_force, _line_search_alpha  # noqa: E402  (read-only reuse)
from run_cruciform_fe2_claude import build_cruciform_model_part, MATERIAL_FUNCS  # noqa: E402

N_STEPS = 20
RESIDUAL_REL_TOL = 1.0e-4
RESIDUAL_ABS_TOL = 1.0e-9
STALL_ACCEPT_REL_TOL = 1.0e-2
STALL_WINDOW = 2
STALL_PATIENCE_REL = 0.10


def tip_force_unit(coords, tip_nodes, n_dof, eq_map):
    """Sum of 4 consistent nodal-load contributions, one per tip, each
    pulling outward along its own single free direction -- the force-
    controlled analog of the displacement protocol's 4 prescribed tip
    pulls. Total magnitude 1.0 per tip (scaled by load_factor later)."""
    f = np.zeros((coords.shape[0], 2))
    f += consistent_edge_force(coords, tip_nodes["px"], direction=(1.0, 0.0))
    f += consistent_edge_force(coords, tip_nodes["mx"], direction=(-1.0, 0.0))
    f += consistent_edge_force(coords, tip_nodes["py"], direction=(0.0, 1.0))
    f += consistent_edge_force(coords, tip_nodes["my"], direction=(0.0, -1.0))
    f_eq = np.zeros(n_dof)
    np.add.at(f_eq, eq_map[:, 0], f[:, 0])
    np.add.at(f_eq, eq_map[:, 1], f[:, 1])
    return f_eq


def run_newton_fe2_cruciform_force(
    which, n_body=6, n_arm_len=4, verbose=True, use_line_search=True, save_npz=True,
    max_newton_iter=30, n_steps=N_STEPS, total_force_final=1.0,
    stall_accept_rel_tol=STALL_ACCEPT_REL_TOL, stall_window=STALL_WINDOW, stall_patience_rel=STALL_PATIENCE_REL,
):
    if which not in MATERIAL_FUNCS:
        raise ValueError(f"which must be one of {list(MATERIAL_FUNCS)}, got {which!r}")
    mp, coords, tris, tip_nodes, center_node = build_cruciform_model_part(n_body, n_arm_len, fix_tips=False)
    n_dof, eq_map, ta = fom.SetUpDofEquationIdsAndDisplacementAdaptor(mp)

    fom._neo_hookean_pk2_2d_vectorized = MATERIAL_FUNCS[which]
    assembler = fom.VectorizedAssembler(mp, n_dof, eq_map, log_label=f"CruciformFE2Force[{which}]")

    f_unit_eq = tip_force_unit(coords, tip_nodes, n_dof, eq_map)
    dirichlet_dofs = set(eq_map[center_node].tolist())
    free_dofs = np.array([d for d in range(n_dof) if d not in dirichlet_dofs])

    u = np.zeros(n_dof)
    step_log = []
    residual_histories = {}
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
                print(f"      iter {it:2d}  |res|={res_norm:.6e}  |res|/|res0|={res_norm / res_norm0:.6e}")
            if res_norm < RESIDUAL_ABS_TOL or res_norm / res_norm0 < RESIDUAL_REL_TOL:
                converged = True
                break
            if it >= 2 * stall_window:
                recent_best = min(res_history[-stall_window:])
                prior_best = min(res_history[-2 * stall_window:-stall_window])
                if recent_best >= (1.0 - stall_patience_rel) * prior_best:
                    stall_detected = True
                    if verbose:
                        print(f"      [{which}] step {step}: stall detected at iter {it}; accepting best iterate")
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
                n_material_calls += 2
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
                raise RuntimeError(f"[{which}] step {step}: no finite iterate was ever produced.")
            u = best_u.copy()

        residual_histories[step] = np.array(res_history)
        tip_ux_px = u[eq_map[tip_nodes["px"], 0]]
        tip_uy_py = u[eq_map[tip_nodes["py"], 1]]
        step_log.append({
            "step": step, "load": load_factor, "iters": n_iter, "converged": converged, "status": status,
            "best_rel": float(best_rel), "stall_detected_early": stall_detected,
            "tip_ux_px_mean": float(np.mean(tip_ux_px)), "tip_uy_py_mean": float(np.mean(tip_uy_py)),
        })
        if verbose:
            tag = {"converged": "OK", "stalled": "STALLED (accepted)", "diverged": "DIVERGED (accepted)"}[status]
            elapsed = time.perf_counter() - t_wall_start
            print(f"  [{which}] step {step:2d}  load={load_factor:.4e}  iters={n_iter:2d}  {tag}  "
                  f"best_rel={best_rel:.3e}  tip_ux(px)={tip_ux_px.mean():.4f}  tip_uy(py)={tip_uy_py.mean():.4f}  "
                  f"(elapsed={elapsed:.1f}s)", flush=True)

    fom.SetDisplacementFromEquationVector(u, eq_map, ta)
    e_voigt, s_voigt = assembler.ComputeStrainStressOnly(u)
    e_flat = e_voigt.reshape(-1, 3).copy()
    s_flat = s_voigt.reshape(-1, 3).copy()
    fully_converged = all(s["converged"] for s in step_log) and len(step_log) == n_steps
    ever_diverged = any(s["status"] == "diverged" for s in step_log)
    t_wall_total = time.perf_counter() - t_wall_start

    if verbose:
        print(f"\n  [{which}] per-step status: " + ", ".join(f"{s['step']}:{s['status'][0].upper()}" for s in step_log))

    if save_npz:
        np.savez(
            HERE / f"cruciform_force_results_{which}_claude.npz",
            coords=coords, tris=tris, u_nodal=np.stack([u[eq_map[:, 0]], u[eq_map[:, 1]]], axis=1),
            e_gp=e_flat, s_gp=s_flat,
            iters_per_step=np.array([s["iters"] for s in step_log]),
            converged_per_step=np.array([s["converged"] for s in step_log]),
            status_per_step=np.array([s["status"] for s in step_log]),
            best_rel_per_step=np.array([s["best_rel"] for s in step_log]),
            tip_ux_px_mean_per_step=np.array([s["tip_ux_px_mean"] for s in step_log]),
            tip_uy_py_mean_per_step=np.array([s["tip_uy_py_mean"] for s in step_log]),
            load_per_step=np.array([s["load"] for s in step_log]),
            fully_converged=fully_converged, ever_diverged=ever_diverged, total_force_final=total_force_final,
        )

    print(f"  [{which}] TOTAL wall time={t_wall_total:.1f}s, material calls~{n_material_calls}, "
          f"avg {t_wall_total / max(n_material_calls, 1):.3f}s/call", flush=True)

    return {
        "which": which, "step_log": step_log,
        "fully_converged": fully_converged, "ever_diverged": ever_diverged,
        "wall_time": t_wall_total, "n_material_calls": n_material_calls,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--which", default="pann_certified", choices=list(MATERIAL_FUNCS))
    p.add_argument("--n-body", type=int, default=6)
    p.add_argument("--n-arm-len", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=N_STEPS)
    p.add_argument("--total-force-final", type=float, default=1.0)
    a = p.parse_args()

    res = run_newton_fe2_cruciform_force(
        a.which, n_body=a.n_body, n_arm_len=a.n_arm_len, n_steps=a.n_steps,
        total_force_final=a.total_force_final, verbose=True, use_line_search=True, save_npz=True,
    )
    print(f"\n=== summary === fully_converged={res['fully_converged']}, ever_diverged={res['ever_diverged']}")
