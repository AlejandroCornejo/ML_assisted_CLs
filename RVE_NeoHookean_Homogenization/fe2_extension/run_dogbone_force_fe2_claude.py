#!/usr/bin/env python3
"""Dogbone specimen, FORCE-controlled at BOTH ends (equal and opposite
consistent nodal tractions on the grip end faces), instead of the fully-
clamped (Dirichlet ux AND uy) ends used in run_dogbone_fe2_claude.py.

Unlike the plain bar (which became a perfectly uniform, trivial state under
force control -- confirmed this session), the dogbone still has a genuine
geometric feature (the shoulder). This isolates how much of the shoulder's
own shear concentration was a boundary-condition artifact (the rigid
clamped-face effect just confirmed on the plain bar) versus how much is a
real, geometry-driven stress concentration that force control cannot remove.

Minimal RBM removal only: the center node (x=0,y=0) fully fixed, plus one
node's y-displacement pinned to remove the remaining in-plane rotation --
not a whole clamped face."""
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
from run_cook_pann_claude import _line_search_alpha, consistent_edge_force  # noqa: E402
from run_cruciform_fe2_claude import MATERIAL_FUNCS  # noqa: E402

N_STEPS = 20
RESIDUAL_REL_TOL = 1.0e-4
RESIDUAL_ABS_TOL = 1.0e-9
STALL_ACCEPT_REL_TOL = 1.0e-2
STALL_WINDOW = 2
STALL_PATIENCE_REL = 0.10


def build_dogbone_force_model_part(L_gauge=8.0, W_gauge=4.0, W_grip=8.0, R=2.0, L_grip=6.0, nx_hint=None):
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

    center_node = int(np.argmin(np.sum(coords ** 2, axis=1)))
    mp.GetNode(center_node + 1).Fix(KM.DISPLACEMENT_X)
    mp.GetNode(center_node + 1).Fix(KM.DISPLACEMENT_Y)

    right_mid = min(right_nodes, key=lambda i: abs(coords[i, 1]))
    mp.GetNode(right_mid + 1).Fix(KM.DISPLACEMENT_Y)

    return mp, coords, tris, left_nodes, right_nodes, center_node, right_mid, geom


def run_newton_fe2_dogbone_force(
    which, L_gauge=8.0, W_gauge=4.0, W_grip=8.0, R=2.0, L_grip=6.0,
    verbose=True, use_line_search=True, save_npz=False,
    max_newton_iter=30, n_steps=N_STEPS, total_force_final=2.0e8,
    stall_accept_rel_tol=STALL_ACCEPT_REL_TOL, stall_window=STALL_WINDOW, stall_patience_rel=STALL_PATIENCE_REL,
):
    if which not in MATERIAL_FUNCS:
        raise ValueError(f"which must be one of {list(MATERIAL_FUNCS)}, got {which!r}")
    mp, coords, tris, left_nodes, right_nodes, center_node, right_mid, geom = build_dogbone_force_model_part(
        L_gauge, W_gauge, W_grip, R, L_grip)
    n_dof, eq_map, ta = fom.SetUpDofEquationIdsAndDisplacementAdaptor(mp)

    fom._neo_hookean_pk2_2d_vectorized = MATERIAL_FUNCS[which]
    assembler = fom.VectorizedAssembler(mp, n_dof, eq_map, log_label=f"DogboneForceFE2[{which}]")

    dirichlet_dofs = {int(eq_map[center_node, 0]), int(eq_map[center_node, 1]), int(eq_map[right_mid, 1])}
    free_dofs = np.array([d for d in range(n_dof) if d not in dirichlet_dofs])

    f_right = consistent_edge_force(coords, right_nodes, direction=(1.0, 0.0))
    f_left = -consistent_edge_force(coords, left_nodes, direction=(1.0, 0.0))
    f_unit = np.zeros(n_dof)
    for nid in range(coords.shape[0]):
        f_unit[eq_map[nid, 0]] += f_right[nid, 0] + f_left[nid, 0]
        f_unit[eq_map[nid, 1]] += f_right[nid, 1] + f_left[nid, 1]

    u = np.zeros(n_dof)
    step_log = []
    t_wall_start = time.perf_counter()

    for step in range(1, n_steps + 1):
        frac = step / n_steps
        f_ext = total_force_final * frac * f_unit

        converged = False
        status = "diverged"
        n_iter = 0
        res_norm0 = None
        res_history = []
        best_u, best_res = u.copy(), np.inf
        for it in range(1, max_newton_iter + 1):
            n_iter = it
            K, rhs_int = assembler.Assemble(u)
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
            u[free_dofs] += alpha * du_free

        best_rel = best_res / res_norm0 if np.isfinite(best_res) else np.inf
        if converged:
            status = "converged"
        elif np.isfinite(best_rel) and best_rel < stall_accept_rel_tol:
            status = "stalled"
        if not converged:
            if not np.all(np.isfinite(best_u)):
                raise RuntimeError(f"[{which}] step {step}: no finite iterate was ever produced.")
            u = best_u.copy()

        step_log.append({"step": step, "iters": n_iter, "converged": converged, "status": status,
                          "best_rel": float(best_rel)})
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
            HERE / f"dogbone_force_results_{which}_claude.npz",
            coords=coords, tris=tris, u_nodal=np.stack([u[eq_map[:, 0]], u[eq_map[:, 1]]], axis=1),
            e_gp=e_flat, s_gp=s_flat, fully_converged=fully_converged, ever_diverged=ever_diverged,
            total_force_final=total_force_final,
        )

    print(f"  [{which}] TOTAL wall time={t_wall_total:.1f}s")

    return {
        "which": which, "step_log": step_log,
        "e11_range": (float(e_flat[:, 0].min()), float(e_flat[:, 0].max())),
        "e22_range": (float(e_flat[:, 1].min()), float(e_flat[:, 1].max())),
        "g12_range": (float(e_flat[:, 2].min()), float(e_flat[:, 2].max())),
        "fully_converged": fully_converged, "ever_diverged": ever_diverged, "wall_time": t_wall_total,
        "coords": coords, "tris": tris, "e_gp": e_flat,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--which", default="pann_certified", choices=list(MATERIAL_FUNCS))
    p.add_argument("--force", type=float, default=2.0e8)
    a = p.parse_args()
    res = run_newton_fe2_dogbone_force(a.which, total_force_final=a.force, verbose=False, save_npz=False)
    print(f"\n{res['which']}: fully_converged={res['fully_converged']}  ever_diverged={res['ever_diverged']}  "
          f"E11={res['e11_range']}  E22={res['e22_range']}  g12={res['g12_range']}  wall={res['wall_time']:.1f}s")
