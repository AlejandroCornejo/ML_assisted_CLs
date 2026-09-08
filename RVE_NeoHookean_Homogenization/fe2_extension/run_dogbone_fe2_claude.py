#!/usr/bin/env python3
"""Dogbone (notched tension) specimen driven by genuine FE^2 -- the third
macroscopic problem in this project's FE^2 comparison, after Cook's membrane
(shear/bending-dominated, mostly outside this RVE's own comfortable range)
and the cruciform (biaxial, but with sharp reentrant corners). This one is
built specifically to be axial-tension-dominated, per Joaquin Hernandez's
own recommendation (a classic uniaxial traction coupon, "estiramiento de una
barra"), with a genuine but bounded (non-singular, convex-corner) stress
concentration at the S-curve shoulders -- so it is a real structural FE^2
demonstration, not a trivial uniform-affine block, while staying as close as
this project's own training range allows to pure tension.

Same material-law-swapping pattern, Newton driver, line search, and
stall/diverge bookkeeping as run_cruciform_fe2_claude.py -- reuses that
module's own MATERIAL_FUNCS dict directly rather than redefining any of the
six survivor models' constructors."""
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

from gen_dogbone_mesh_claude import generate_dogbone_mesh  # noqa: E402
from run_cook_pann_claude import _line_search_alpha  # noqa: E402
from run_cruciform_fe2_claude import MATERIAL_FUNCS  # noqa: E402  (reused, not redefined)

N_STEPS = 20
RESIDUAL_REL_TOL = 1.0e-4
RESIDUAL_ABS_TOL = 1.0e-9
STALL_ACCEPT_REL_TOL = 1.0e-2
STALL_WINDOW = 2
STALL_PATIENCE_REL = 0.10


def build_dogbone_model_part(L_gauge=8.0, W_gauge=4.0, W_grip=8.0, R=2.0, L_grip=6.0):
    """Left end fully clamped (both DOFs), right end displacement-driven in
    X and held to Y=0 (a rigid grip face, no rotation) -- matching Joaquin's
    own MAW-ECM-paper convention for treating a prescribed-boundary-
    displacement problem as homogenization with a physical interpretation.
    No separate RBM-fixing node needed: the fully clamped left end already
    removes all three planar rigid-body modes, exactly as Cook's own
    single-clamped-edge convention does."""
    coords, tris, left_nodes, right_nodes, geom = generate_dogbone_mesh(
        L_gauge=L_gauge, W_gauge=W_gauge, W_grip=W_grip, R=R, L_grip=L_grip)
    model = KM.Model()
    mp = model.CreateModelPart("Structure")
    mp.SetBufferSize(1)
    mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
    mp.AddNodalSolutionStepVariable(KM.REACTION)

    for i, (x, y) in enumerate(coords):
        mp.CreateNewNode(i + 1, float(x), float(y), 0.0)

    prop = mp.GetProperties()[1]
    prop.SetValue(KM.YOUNG_MODULUS, 1.0)
    prop.SetValue(KM.POISSON_RATIO, 0.3)
    prop.SetValue(KM.THICKNESS, 1.0)

    for e, conn in enumerate(tris):
        node_ids = [int(c) + 1 for c in conn]
        mp.CreateNewElement("TotalLagrangianElement2D6N", e + 1, node_ids, prop)

    KM.VariableUtils().AddDof(KM.DISPLACEMENT_X, mp)
    KM.VariableUtils().AddDof(KM.DISPLACEMENT_Y, mp)

    for nid in left_nodes:
        mp.GetNode(nid + 1).Fix(KM.DISPLACEMENT_X)
        mp.GetNode(nid + 1).Fix(KM.DISPLACEMENT_Y)
    for nid in right_nodes:
        mp.GetNode(nid + 1).Fix(KM.DISPLACEMENT_X)
        mp.GetNode(nid + 1).Fix(KM.DISPLACEMENT_Y)

    return mp, coords, tris, left_nodes, right_nodes, geom


def run_newton_fe2_dogbone(
    which, L_gauge=8.0, W_gauge=4.0, W_grip=8.0, R=2.0, L_grip=6.0,
    verbose=True, use_line_search=True, save_npz=True,
    max_newton_iter=30, n_steps=N_STEPS, delta_x_final=2.0,
    stall_accept_rel_tol=STALL_ACCEPT_REL_TOL, stall_window=STALL_WINDOW, stall_patience_rel=STALL_PATIENCE_REL,
    max_steps_to_run=None,
):
    """delta_x_final: total prescribed axial displacement at the right end
    (left end fixed at 0), ramped linearly over n_steps -- same ramp
    convention as Cook/cruciform. Not yet calibrated to any particular
    target macro-strain range; call once with a cheap law (pann_certified)
    and inspect e11_range/e22_range/g12_range before trusting any delta as
    "the" final value, exactly as this project's own established discipline
    (Cook's load-ceiling sweep, the cruciform's own ICNN-only mesh-
    convergence pass) requires."""
    if which not in MATERIAL_FUNCS:
        raise ValueError(f"which must be one of {list(MATERIAL_FUNCS)}, got {which!r}")
    mp, coords, tris, left_nodes, right_nodes, geom = build_dogbone_model_part(
        L_gauge, W_gauge, W_grip, R, L_grip)
    n_dof, eq_map, ta = fom.SetUpDofEquationIdsAndDisplacementAdaptor(mp)

    fom._neo_hookean_pk2_2d_vectorized = MATERIAL_FUNCS[which]
    assembler = fom.VectorizedAssembler(mp, n_dof, eq_map, log_label=f"DogboneFE2[{which}]")

    left_x_dofs = eq_map[left_nodes, 0].tolist()
    left_y_dofs = eq_map[left_nodes, 1].tolist()
    right_x_dofs = eq_map[right_nodes, 0].tolist()
    right_y_dofs = eq_map[right_nodes, 1].tolist()
    dirichlet_dofs = set(left_x_dofs + left_y_dofs + right_x_dofs + right_y_dofs)
    free_dofs = np.array([d for d in range(n_dof) if d not in dirichlet_dofs])
    f_ext = np.zeros(n_dof)

    u = np.zeros(n_dof)
    step_log = []
    residual_histories = {}
    t_wall_start = time.perf_counter()
    n_material_calls = 0

    for step in range(1, n_steps + 1):
        frac = step / n_steps
        u[right_x_dofs] = delta_x_final * frac
        # left_y/right_y/left_x stay at 0 throughout (already zero-initialized)

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
        step_log.append({
            "step": step, "iters": n_iter, "converged": converged, "status": status,
            "best_rel": float(best_rel), "stall_detected_early": stall_detected,
        })
        if verbose:
            tag = {"converged": "OK", "stalled": "STALLED (accepted)", "diverged": "DIVERGED (accepted)"}[status]
            elapsed = time.perf_counter() - t_wall_start
            print(f"  [{which}] step {step:2d}  frac={frac:.3f}  iters={n_iter:2d}  {tag}  "
                  f"best_rel={best_rel:.3e}  (elapsed={elapsed:.1f}s)")

        if max_steps_to_run is not None and step >= max_steps_to_run:
            if verbose:
                print(f"  [{which}] stopping early after {step}/{n_steps} steps (max_steps_to_run="
                      f"{max_steps_to_run})")
            break

    fom.SetDisplacementFromEquationVector(u, eq_map, ta)
    e_voigt, s_voigt = assembler.ComputeStrainStressOnly(u)
    e_flat = e_voigt.reshape(-1, 3).copy()
    s_flat = s_voigt.reshape(-1, 3).copy()

    _, rhs_int_final = assembler.Assemble(u)
    n_material_calls += 1
    reaction_x = float(np.sum(rhs_int_final[right_x_dofs]))

    fully_converged = all(s["converged"] for s in step_log) and len(step_log) == n_steps
    ever_diverged = any(s["status"] == "diverged" for s in step_log)
    t_wall_total = time.perf_counter() - t_wall_start

    if verbose:
        print(f"\n  [{which}] per-step status: " + ", ".join(f"{s['step']}:{s['status'][0].upper()}" for s in step_log))
        print(f"  [{which}] final macro strain range: E11=[{e_flat[:, 0].min():.4f},{e_flat[:, 0].max():.4f}] "
              f"E22=[{e_flat[:, 1].min():.4f},{e_flat[:, 1].max():.4f}] "
              f"g12=[{e_flat[:, 2].min():.4f},{e_flat[:, 2].max():.4f}]")

    if save_npz:
        np.savez(
            HERE / f"dogbone_results_{which}_claude.npz",
            coords=coords, tris=tris, u_nodal=np.stack([u[eq_map[:, 0]], u[eq_map[:, 1]]], axis=1),
            e_gp=e_flat, s_gp=s_flat,
            iters_per_step=np.array([s["iters"] for s in step_log]),
            converged_per_step=np.array([s["converged"] for s in step_log]),
            status_per_step=np.array([s["status"] for s in step_log]),
            best_rel_per_step=np.array([s["best_rel"] for s in step_log]),
            residual_history_step1=residual_histories.get(1, np.zeros(0)),
            fully_converged=fully_converged, ever_diverged=ever_diverged,
            delta_x_final=delta_x_final, reaction_x=reaction_x,
        )

    print(f"  [{which}] TOTAL wall time={t_wall_total:.1f}s, material calls~{n_material_calls}, "
          f"avg {t_wall_total / max(n_material_calls, 1):.3f}s/call")

    return {
        "which": which, "step_log": step_log,
        "e11_range": (float(e_flat[:, 0].min()), float(e_flat[:, 0].max())),
        "e22_range": (float(e_flat[:, 1].min()), float(e_flat[:, 1].max())),
        "g12_range": (float(e_flat[:, 2].min()), float(e_flat[:, 2].max())),
        "fully_converged": fully_converged, "ever_diverged": ever_diverged,
        "wall_time": t_wall_total, "n_material_calls": n_material_calls,
        "reaction_x": reaction_x,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--which", default="pann_certified", choices=list(MATERIAL_FUNCS))
    p.add_argument("--n-steps", type=int, default=N_STEPS)
    p.add_argument("--delta", type=float, default=2.0)
    a = p.parse_args()

    res = run_newton_fe2_dogbone(
        a.which, n_steps=a.n_steps, delta_x_final=a.delta,
        verbose=True, use_line_search=True, save_npz=True,
    )
    print("\n=== summary ===")
    print(f"{res['which']}: fully_converged={res['fully_converged']}, ever_diverged={res['ever_diverged']}, "
          f"E11={res['e11_range']}, E22={res['e22_range']}, g12={res['g12_range']}, wall={res['wall_time']:.1f}s")
