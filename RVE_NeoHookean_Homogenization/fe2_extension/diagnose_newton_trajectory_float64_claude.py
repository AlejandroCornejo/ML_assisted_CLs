#!/usr/bin/env python3
"""Decisive precision-floor test: exact same Cook step-1 Newton loop as
diagnose_newton_trajectory_rankone_claude.py (same mesh, same BCs, same
line search, imported read-only), but with D-HPROM-ANN's float64 copy
(dhprom_ann_direct_law_float64_claude.py) as the material law instead of
the original float32 version. If the residual plateau drops to a
meaningfully lower level (or the step actually converges), that confirms
float32 precision was the bottleneck. If the plateau is unchanged, that
refutes it, just as cleanly as the tangent and rank-one tests did.
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
from run_cook_pann_claude import build_model_part, consistent_edge_force, _line_search_alpha  # noqa: E402
import dhprom_ann_direct_law_float64_claude as dhprom_f64_module  # noqa: E402

EDGE_LENGTH = 16.0
LINE_LOAD_MODULUS_FINAL = 13_000_000.0
TOTAL_FORCE_FINAL = LINE_LOAD_MODULUS_FINAL * EDGE_LENGTH
N_STEPS = 20
RESIDUAL_REL_TOL = 1.0e-4
RESIDUAL_ABS_TOL = 1.0e-9


def run_step1_float64(nx=8, ny=8, max_it=15):
    mp, coords, tris, left_nodes, right_nodes = build_model_part(nx, ny)
    n_dof, eq_map, ta = fom.SetUpDofEquationIdsAndDisplacementAdaptor(mp)

    fom._neo_hookean_pk2_2d_vectorized = dhprom_f64_module.dhprom_ann_pk2_2d_vectorized_consistent_float64
    assembler = fom.VectorizedAssembler(mp, n_dof, eq_map, log_label="diag-float64")

    f_unit = consistent_edge_force(coords, right_nodes)
    f_unit_eq = np.zeros(n_dof)
    np.add.at(f_unit_eq, eq_map[:, 0], f_unit[:, 0])
    np.add.at(f_unit_eq, eq_map[:, 1], f_unit[:, 1])

    free_dofs = np.array([d for d in range(n_dof)
                           if d not in set(eq_map[left_nodes, 0]) | set(eq_map[left_nodes, 1])])

    u = np.zeros(n_dof)
    load_factor = TOTAL_FORCE_FINAL / N_STEPS
    f_ext = f_unit_eq * load_factor

    res_norm0 = None
    res_history = []
    for it in range(1, max_it + 1):
        t0 = time.perf_counter()
        K, rhs_int = assembler.Assemble(u)
        residual = rhs_int + f_ext
        res_free = residual[free_dofs]
        res_norm = float(np.linalg.norm(res_free))
        res_history.append(res_norm)
        if res_norm0 is None:
            res_norm0 = max(res_norm, 1e-12)
        print(f"  iter {it:2d}: |res|={res_norm:.6e}  |res|/|res0|={res_norm / res_norm0:.6e}  ({time.perf_counter() - t0:.2f}s)")
        if res_norm < RESIDUAL_ABS_TOL or res_norm / res_norm0 < RESIDUAL_REL_TOL:
            print("  CONVERGED.")
            break

        K_ff = K[free_dofs, :][:, free_dofs]
        du_free = spsolve(K_ff.tocsc(), res_free)
        if not np.all(np.isfinite(du_free)):
            print("  non-finite update, stopping.")
            break
        alpha = _line_search_alpha(assembler, u, du_free, free_dofs, f_ext)
        u[free_dofs] += alpha * du_free

    return res_history


if __name__ == "__main__":
    print("[diag-float64] running Cook step-1 Newton with D-HPROM-ANN-consistent-FLOAT64 ...")
    res_history = run_step1_float64()

    print("\n=== comparison ===")
    print("float64 residual history:", [f"{r:.4e}" for r in res_history])
    print(f"float64 plateau range (iters 5+): "
          f"[{min(res_history[4:]):.4e}, {max(res_history[4:]):.4e}]" if len(res_history) > 4 else "N/A")
    print("\nFor reference, the ORIGINAL float32 version's step-1 plateau (from earlier this session) "
          "was approximately [3.1e3, 3.8e3] over iterations 5-15.")
