#!/usr/bin/env python3
"""Static analog of As'ad, Avery & Farhat (2022, IJNME 123:2738-2759,
Section 4.3/Figure 9)'s dynamic consistency demonstration, using our own
plane-stress solid element (no shells needed -- the phenomenon is a
property of the material law, not the element formulation): zero external
force everywhere, and the MINIMAL constraint needed to remove the 3 planar
rigid-body modes (both DOFs of the center node, plus the y-DOF of one node
on the +x axis, to remove rotation) -- nothing else fixed.

If S(E=0)=0 (verified this session: ICNN, ICKAN, Free), u=0 already
satisfies equilibrium exactly (zero residual, zero Newton iterations
needed) -- the structure simply stays at rest, as it must with no applied
load. If S(E=0)!=0 (verified this session: Regression, |S(0)|~=2.79e5),
u=0 is NOT an equilibrium state despite zero applied load -- the solver
must search for whatever configuration balances the spurious self-stress
at the free DOFs, producing visible, unforced, spontaneous deformation
from nothing."""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
import time
from pathlib import Path

HERE = Path("/home/kratos/ML_assisted_CLs_clean/RVE_NeoHookean_Homogenization/fe2_extension")
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
os.chdir(str(HERE))

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.insert(0, KRATOS_PATH)

import numpy as np
from scipy.sparse.linalg import spsolve

from run_cruciform_fe2_claude import MATERIAL_FUNCS, RESIDUAL_ABS_TOL  # noqa: E402
import fom_solver_rve as fom  # noqa: E402
from run_cook_pann_claude import _line_search_alpha  # noqa: E402
from build_square_panel_mesh_claude import build_square_panel_mesh  # noqa: E402
import KratosMultiphysics as KM  # noqa: E402

N_BODY = 8
L_BODY = 12.0


def build_unforced_model_part():
    coords, tris, edge_nodes, center_node = build_square_panel_mesh(n_body=N_BODY, L_body=L_BODY)

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

    # Minimal rigid-body-mode removal: both DOFs at the center (0,0), plus
    # the y-DOF of one node strictly on the +x axis (removes rotation) --
    # nothing else. Found by searching the right-edge node list for the one
    # actually at y=0 (the structured mesh's own indexing order is bottom-
    # to-top, not guaranteed to put it first).
    px_axis_node = next(n for n in edge_nodes["px"] if abs(coords[n, 1]) < 1e-9)
    mp.GetNode(center_node + 1).Fix(KM.DISPLACEMENT_X)
    mp.GetNode(center_node + 1).Fix(KM.DISPLACEMENT_Y)
    mp.GetNode(px_axis_node + 1).Fix(KM.DISPLACEMENT_Y)

    return mp, coords, tris, center_node, px_axis_node


def run_consistency_test(which, max_newton_iter=50, use_line_search=True, verbose=True):
    if which not in MATERIAL_FUNCS:
        raise ValueError(f"which must be one of {list(MATERIAL_FUNCS)}, got {which!r}")

    mp, coords, tris, center_node, px_axis_node = build_unforced_model_part()
    n_dof, eq_map, ta = fom.SetUpDofEquationIdsAndDisplacementAdaptor(mp)

    fom._neo_hookean_pk2_2d_vectorized = MATERIAL_FUNCS[which]
    assembler = fom.VectorizedAssembler(mp, n_dof, eq_map, log_label=f"ConsistencyTest[{which}]")

    dirichlet_dofs = set(eq_map[center_node].tolist()) | {int(eq_map[px_axis_node, 1])}
    free_dofs = np.array([d for d in range(n_dof) if d not in dirichlet_dofs])
    f_ext = np.zeros(n_dof)  # zero external force, everywhere, always

    u = np.zeros(n_dof)
    res_norm0 = None
    res_history = []
    t0 = time.perf_counter()
    for it in range(1, max_newton_iter + 1):
        K, rhs_int = assembler.Assemble(u)
        residual = rhs_int + f_ext
        res_free = residual[free_dofs]
        res_norm = np.linalg.norm(res_free)
        res_history.append(res_norm)
        if res_norm0 is None:
            res_norm0 = max(res_norm, 1e-12)
        if verbose:
            print(f"  [{which}] iter {it:2d}  |res|={res_norm:.6e}  |u|_max={np.max(np.abs(u)):.6e}")
        if res_norm < RESIDUAL_ABS_TOL:
            break
        K_ff = K[free_dofs, :][:, free_dofs]
        try:
            du_free = spsolve(K_ff.tocsc(), res_free)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{which}] linear solve failed at iter {it}: {exc}")
            break
        if not np.all(np.isfinite(du_free)):
            print(f"  [{which}] non-finite update at iter {it}")
            break
        alpha = 1.0
        if use_line_search:
            alpha = _line_search_alpha(assembler, u, du_free, free_dofs, f_ext)
        u[free_dofs] += alpha * du_free

    wall = time.perf_counter() - t0
    u_max = float(np.max(np.abs(u)))
    print(f"[{which}] TOTAL: {it} iterations, final |res|={res_history[-1]:.3e}, "
          f"max |u|={u_max:.6e}, wall={wall:.2f}s")
    return {"which": which, "n_iter": it, "res_history": res_history, "u": u.copy(), "u_max": u_max}


if __name__ == "__main__":
    results = {}
    for which in ("pann_certified", "pann_free", "pann_regression"):
        print(f"\n=== {which}: zero force, minimal rigid-body constraint only ===")
        results[which] = run_consistency_test(which, verbose=True)

    print("\n=== SUMMARY: spontaneous deformation under ZERO applied load ===")
    for which, res in results.items():
        print(f"  {which:16s}: {res['n_iter']} iters, max|u|={res['u_max']:.6e} "
              f"(expect ~0 for consistent laws, nonzero for Regression)")
    print("CONSISTENCY_TEST_DONE_MARKER")
