#!/usr/bin/env python3
"""Exploratory, non-paper run: genuine (non-reduced) FE^2 for JUST Cook's
membrane's first load step (5% of the final load, nx=8), to get one
independently-computed ground-truth tip-u_y number to compare against the
certified PANN, D-HPROM-ANN/HPROM-ANN (f64), free, and regression results
already saved in this directory -- all of which already have
tip_uy_min_per_step[0]/tip_uy_max_per_step[0] on file. A quick comparison
point only, not wired into the paper.

Step 1 is uniquely comparable across every one of those laws: all of them
start from the same undeformed, zero-displacement state, so there is no
warm-start confound from a previous, model-specific converged state
(unlike step 2+, whose Newton warm-start already differs law to law).

Mirrors run_cook_hprom_ann_claude.py's run_newton_fe2 Newton loop for a
SINGLE step, reusing its own constants (EDGE_LENGTH, LINE_LOAD_MODULUS_FINAL,
TOTAL_FORCE_FINAL, N_STEPS) so load_factor is exactly total_force_final*1/20,
not a different, incomparable 100%-in-one-shot load. That module itself is
imported read-only for its constants and is not modified or re-run here.
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

from run_cook_pann_claude import build_model_part, consistent_edge_force, _line_search_alpha  # noqa: E402
from run_cook_hprom_ann_claude import (  # noqa: E402  (read-only: constants only, not re-run)
    EDGE_LENGTH, LINE_LOAD_MODULUS_FINAL, TOTAL_FORCE_FINAL, N_STEPS,
)
from fom_nested_law_claude import fom_nested_pk2_2d_vectorized  # noqa: E402

NX = NY = 8
MAX_NEWTON_ITER = 12
USE_LINE_SEARCH = False  # try the cheap path first -- the tangent's exact
# accuracy cannot change the converged fixed point (only the FOM's own true
# stress law, enforced at zero residual, determines it), so plain, undamped
# Newton is worth trying before paying line search's extra material calls
# (each of which re-triggers a full 384-GP x 7-solve material evaluation).
RESIDUAL_REL_TOL = 1.0e-4
RESIDUAL_ABS_TOL = 1.0e-9


def main():
    mp, coords, tris, left_nodes, right_nodes = build_model_part(NX, NY)
    n_dof, eq_map, ta = fom.SetUpDofEquationIdsAndDisplacementAdaptor(mp)

    fom._neo_hookean_pk2_2d_vectorized = fom_nested_pk2_2d_vectorized
    assembler = fom.VectorizedAssembler(mp, n_dof, eq_map, log_label="CookFOMnested[step1]")

    f_unit = consistent_edge_force(coords, right_nodes)
    f_unit_eq = np.zeros(n_dof)
    np.add.at(f_unit_eq, eq_map[:, 0], f_unit[:, 0])
    np.add.at(f_unit_eq, eq_map[:, 1], f_unit[:, 1])

    free_dofs = np.array([d for d in range(n_dof)
                           if d not in set(eq_map[left_nodes, 0]) | set(eq_map[left_nodes, 1])])

    load_factor = TOTAL_FORCE_FINAL * 1 / N_STEPS
    f_ext = f_unit_eq * load_factor
    print(f"[fom-nested-step1] nx=ny={NX}, {tris.shape[0]} elements, {3 * tris.shape[0]} Gauss points, "
          f"load_factor={load_factor:.4e} (= 1/{N_STEPS} of TOTAL_FORCE_FINAL={TOTAL_FORCE_FINAL:.4e}), "
          f"use_line_search={USE_LINE_SEARCH}", flush=True)

    u = np.zeros(n_dof)
    res_history = []
    best_u, best_res = u.copy(), np.inf
    res_norm0 = None
    t_wall_start = time.perf_counter()
    converged = False
    n_iter_done = 0

    for it in range(1, MAX_NEWTON_ITER + 1):
        n_iter_done = it
        t0 = time.perf_counter()
        K, rhs_int = assembler.Assemble(u)
        residual = rhs_int + f_ext
        res_free = residual[free_dofs]
        res_norm = np.linalg.norm(res_free)
        res_history.append(res_norm)
        if res_norm0 is None:
            res_norm0 = max(res_norm, 1e-12)
        if np.isfinite(res_norm) and res_norm < best_res:
            best_res, best_u = float(res_norm), u.copy()
        elapsed = time.perf_counter() - t_wall_start
        print(f"[fom-nested-step1] iter {it:2d}  |res|={res_norm:.6e}  |res|/|res0|={res_norm / res_norm0:.6e}"
              f"  (iter time={time.perf_counter() - t0:.1f}s, elapsed={elapsed:.1f}s)", flush=True)
        if res_norm < RESIDUAL_ABS_TOL or res_norm / res_norm0 < RESIDUAL_REL_TOL:
            converged = True
            break
        if it >= 4:
            recent_best = min(res_history[-2:])
            prior_best = min(res_history[-4:-2])
            if recent_best >= 0.90 * prior_best:
                print(f"[fom-nested-step1] stall detected at iter {it}; accepting best iterate", flush=True)
                break

        K_ff = K[free_dofs, :][:, free_dofs]
        du_free = spsolve(K_ff.tocsc(), res_free)
        if not np.all(np.isfinite(du_free)):
            print(f"[fom-nested-step1] non-finite update at iter {it}; stopping", flush=True)
            break

        alpha = 1.0
        if USE_LINE_SEARCH:
            alpha = _line_search_alpha(assembler, u, du_free, free_dofs, f_ext)
        u[free_dofs] += alpha * du_free

    if not converged:
        u = best_u.copy()

    tip_uy = u[eq_map[right_nodes, 1]]
    t_wall_total = time.perf_counter() - t_wall_start
    best_rel = best_res / res_norm0 if np.isfinite(best_res) else np.inf
    print(f"\n[fom-nested-step1] DONE. converged={converged}  n_iter={n_iter_done}  best_rel={best_rel:.4e}",
          flush=True)
    print(f"[fom-nested-step1] tip_uy range = ({tip_uy.min():.6f}, {tip_uy.max():.6f})  "
          f"TOTAL wall time = {t_wall_total:.1f}s", flush=True)

    np.savez(
        HERE / "cook_results_fom_nested_step1_claude.npz",
        coords=coords, tris=tris, u_nodal=np.stack([u[eq_map[:, 0]], u[eq_map[:, 1]]], axis=1),
        tip_uy_min=float(tip_uy.min()), tip_uy_max=float(tip_uy.max()),
        converged=converged, n_iter=n_iter_done, best_rel=float(best_rel), wall_time=t_wall_total,
        load_factor=load_factor, use_line_search=USE_LINE_SEARCH,
        residual_history=np.array(res_history),
    )
    print(f"[fom-nested-step1] saved to cook_results_fom_nested_step1_claude.npz", flush=True)


if __name__ == "__main__":
    main()
