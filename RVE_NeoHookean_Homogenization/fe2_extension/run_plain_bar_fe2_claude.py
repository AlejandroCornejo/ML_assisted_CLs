#!/usr/bin/env python3
"""Plain rectangular bar (no notch/fillet) driven by genuine FE^2 -- the
limiting/control case for the dogbone: same total length, same width as
the dogbone's gauge, same clamped-one-end/pulled-other-end BC convention,
same material-law-swapping Newton driver as run_dogbone_fe2_claude.py.
Isolates whether the dogbone's shoulder (a load-independent strain-
concentration factor, confirmed this session) is really what caps the
achievable E11 before the shear budget is exceeded, or whether something
else (e.g. a St Venant-type end effect) would cap it first even with no
notch at all."""
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

from generate_plain_bar_mesh_claude import generate_plain_bar_mesh  # noqa: E402
from run_cook_pann_claude import _line_search_alpha  # noqa: E402
from run_cruciform_fe2_claude import MATERIAL_FUNCS  # noqa: E402

N_STEPS = 20
RESIDUAL_REL_TOL = 1.0e-4
RESIDUAL_ABS_TOL = 1.0e-9
STALL_ACCEPT_REL_TOL = 1.0e-2
STALL_WINDOW = 2
STALL_PATIENCE_REL = 0.10


def build_plain_bar_model_part(L, W, nx, ny):
    coords, tris, left_nodes, right_nodes, geom = generate_plain_bar_mesh(L=L, W=W, nx=nx, ny=ny)
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


def run_newton_fe2_plain_bar(
    which, L=None, W=None, nx=25, ny=4,
    verbose=True, use_line_search=True, save_npz=False,
    max_newton_iter=30, n_steps=N_STEPS, delta_x_final=1.5,
    stall_accept_rel_tol=STALL_ACCEPT_REL_TOL, stall_window=STALL_WINDOW, stall_patience_rel=STALL_PATIENCE_REL,
):
    from generate_plain_bar_mesh_claude import L_TOTAL_DEFAULT, W_DEFAULT
    L = L_TOTAL_DEFAULT if L is None else L
    W = W_DEFAULT if W is None else W

    if which not in MATERIAL_FUNCS:
        raise ValueError(f"which must be one of {list(MATERIAL_FUNCS)}, got {which!r}")
    mp, coords, tris, left_nodes, right_nodes, geom = build_plain_bar_model_part(L, W, nx, ny)
    n_dof, eq_map, ta = fom.SetUpDofEquationIdsAndDisplacementAdaptor(mp)

    fom._neo_hookean_pk2_2d_vectorized = MATERIAL_FUNCS[which]
    assembler = fom.VectorizedAssembler(mp, n_dof, eq_map, log_label=f"PlainBarFE2[{which}]")

    left_x_dofs = eq_map[left_nodes, 0].tolist()
    left_y_dofs = eq_map[left_nodes, 1].tolist()
    right_x_dofs = eq_map[right_nodes, 0].tolist()
    right_y_dofs = eq_map[right_nodes, 1].tolist()
    dirichlet_dofs = set(left_x_dofs + left_y_dofs + right_x_dofs + right_y_dofs)
    free_dofs = np.array([d for d in range(n_dof) if d not in dirichlet_dofs])
    f_ext = np.zeros(n_dof)

    u = np.zeros(n_dof)
    step_log = []
    t_wall_start = time.perf_counter()
    n_material_calls = 0

    for step in range(1, n_steps + 1):
        frac = step / n_steps
        u[right_x_dofs] = delta_x_final * frac

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
                    break

            K_ff = K[free_dofs, :][:, free_dofs]
            try:
                du_free = spsolve(K_ff.tocsc(), res_free)
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"  [{which}] step {step}: linear solve failed at iter {it}: {exc}")
                break
            if not np.all(np.isfinite(du_free)):
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

        step_log.append({"step": step, "iters": n_iter, "converged": converged, "status": status,
                          "best_rel": float(best_rel), "stall_detected_early": stall_detected})
        if verbose:
            tag = {"converged": "OK", "stalled": "STALLED", "diverged": "DIVERGED"}[status]
            print(f"  [{which}] step {step:2d}  frac={frac:.3f}  iters={n_iter:2d}  {tag}  best_rel={best_rel:.3e}")

    fom.SetDisplacementFromEquationVector(u, eq_map, ta)
    e_voigt, s_voigt = assembler.ComputeStrainStressOnly(u)
    e_flat = e_voigt.reshape(-1, 3).copy()
    s_flat = s_voigt.reshape(-1, 3).copy()

    fully_converged = all(s["converged"] for s in step_log) and len(step_log) == n_steps
    ever_diverged = any(s["status"] == "diverged" for s in step_log)
    t_wall_total = time.perf_counter() - t_wall_start

    if verbose:
        print(f"\n  [{which}] final macro strain range: E11=[{e_flat[:, 0].min():.4f},{e_flat[:, 0].max():.4f}] "
              f"E22=[{e_flat[:, 1].min():.4f},{e_flat[:, 1].max():.4f}] "
              f"g12=[{e_flat[:, 2].min():.4f},{e_flat[:, 2].max():.4f}]")

    if save_npz:
        np.savez(
            HERE / f"plain_bar_results_{which}_claude.npz",
            coords=coords, tris=tris, u_nodal=np.stack([u[eq_map[:, 0]], u[eq_map[:, 1]]], axis=1),
            e_gp=e_flat, s_gp=s_flat,
            iters_per_step=np.array([s["iters"] for s in step_log]),
            status_per_step=np.array([s["status"] for s in step_log]),
            fully_converged=fully_converged, ever_diverged=ever_diverged, delta_x_final=delta_x_final,
        )

    print(f"  [{which}] TOTAL wall time={t_wall_total:.1f}s, material calls~{n_material_calls}")

    return {
        "which": which, "step_log": step_log,
        "e11_range": (float(e_flat[:, 0].min()), float(e_flat[:, 0].max())),
        "e22_range": (float(e_flat[:, 1].min()), float(e_flat[:, 1].max())),
        "g12_range": (float(e_flat[:, 2].min()), float(e_flat[:, 2].max())),
        "fully_converged": fully_converged, "ever_diverged": ever_diverged, "wall_time": t_wall_total,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--which", default="pann_certified", choices=list(MATERIAL_FUNCS))
    p.add_argument("--delta", type=float, default=1.5)
    a = p.parse_args()
    res = run_newton_fe2_plain_bar(a.which, delta_x_final=a.delta, verbose=False, save_npz=False)
    print(f"\n{res['which']}: fully_converged={res['fully_converged']}  ever_diverged={res['ever_diverged']}  "
          f"E11={res['e11_range']}  E22={res['e22_range']}  g12={res['g12_range']}  wall={res['wall_time']:.1f}s")
