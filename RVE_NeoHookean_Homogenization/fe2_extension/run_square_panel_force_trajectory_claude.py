#!/usr/bin/env python3
"""Force-controlled square panel driven through a multi-segment traction
trajectory (not a single ramp to one final state): tension held at a
moderate, comfortably-"within-range" 3x reference (below the 4x-clean/6x-
diverge threshold already found this session for combined tension+shear,
so this is NOT chasing a known extreme), while shear is swept through a
full cycle (0 -> +0.1 -> -0.1 -> 0 of the tension traction) at that
SUSTAINED tension level -- testing whether reversing shear DIRECTION under
sustained load causes trouble a monotonic ramp doesn't reveal.

Traction at each point is t=S.N per edge, S=[[mult,r*mult],[r*mult,mult]]
(mult=tension multiple of REF_FORCE_SQUARE, r=shear ratio) -- self-
equilibrated by construction, same as run_square_panel_force_combined_
claude.py's own pattern, reusing Cook's own consistent_edge_force unmodified.
Incremental warm-starting from the previous step's own converged state
(the standard, appropriate approach for force control -- unlike
displacement control, there is no closed-form target state to seed at
directly, since the equilibrium state is exactly what solving finds)."""
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

from run_cruciform_fe2_claude import (  # noqa: E402
    MATERIAL_FUNCS, RESIDUAL_ABS_TOL, RESIDUAL_REL_TOL,
    STALL_ACCEPT_REL_TOL, STALL_WINDOW, STALL_PATIENCE_REL,
)
import fom_solver_rve as fom  # noqa: E402
from run_cook_pann_claude import _line_search_alpha  # noqa: E402
import run_cruciform_fe2_force_controlled_claude as rcf  # noqa: E402
from build_square_panel_mesh_claude import build_square_panel_mesh  # noqa: E402

N_BODY = 8
L_BODY = 12.0
STEPS_PER_SEGMENT = 15
REF_FORCE_SQUARE = 5.150081381e9  # ICNN's own |reaction_px| at delta=1.2 on this same panel, verified this session

# (tension_multiple, shear_ratio) keyframes -- tension held fixed at a
# moderate 3x (well below the already-known 4x-clean/6x-diverge threshold),
# shear ratio swept through a full reversing cycle at that sustained level.
KEYFRAMES = [(0.0, 0.0), (3.0, 0.0), (3.0, 0.1), (3.0, -0.1), (3.0, 0.0), (0.0, 0.0)]


def interpolated_targets(keyframes, steps_per_segment):
    targets = []
    for i in range(len(keyframes) - 1):
        a = np.array(keyframes[i])
        b = np.array(keyframes[i + 1])
        for s in range(1, steps_per_segment + 1):
            frac = s / steps_per_segment
            targets.append(tuple(a + (b - a) * frac))
    return targets


def run_force_trajectory(which, verbose=True, use_line_search=True,
                          max_newton_iter=30, keyframes=KEYFRAMES, steps_per_segment=STEPS_PER_SEGMENT):
    if which not in MATERIAL_FUNCS:
        raise ValueError(f"which must be one of {list(MATERIAL_FUNCS)}, got {which!r}")

    coords, tris, edge_nodes, center_node = build_square_panel_mesh(n_body=N_BODY, L_body=L_BODY)

    import KratosMultiphysics as KM
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
    mp.GetNode(center_node + 1).Fix(KM.DISPLACEMENT_X)
    mp.GetNode(center_node + 1).Fix(KM.DISPLACEMENT_Y)

    n_dof, eq_map, ta = fom.SetUpDofEquationIdsAndDisplacementAdaptor(mp)

    fom._neo_hookean_pk2_2d_vectorized = MATERIAL_FUNCS[which]
    assembler = fom.VectorizedAssembler(mp, n_dof, eq_map, log_label=f"SquarePanelForceTraj[{which}]")

    dirichlet_dofs = set(eq_map[center_node].tolist())
    free_dofs = np.array([d for d in range(n_dof) if d not in dirichlet_dofs])

    u = np.zeros(n_dof)
    step_log = []
    targets = interpolated_targets(keyframes, steps_per_segment)
    t_wall_start = time.perf_counter()
    n_material_calls = 0

    for step, (mult, r) in enumerate(targets, start=1):
        f_unit_eq = np.zeros((coords.shape[0], 2))
        f_unit_eq += rcf.consistent_edge_force(coords, edge_nodes["px"], direction=(1.0, r))
        f_unit_eq += rcf.consistent_edge_force(coords, edge_nodes["mx"], direction=(-1.0, -r))
        f_unit_eq += rcf.consistent_edge_force(coords, edge_nodes["py"], direction=(r, 1.0))
        f_unit_eq += rcf.consistent_edge_force(coords, edge_nodes["my"], direction=(-r, -1.0))
        f_ext = np.zeros(n_dof)
        np.add.at(f_ext, eq_map[:, 0], f_unit_eq[:, 0] * mult * REF_FORCE_SQUARE)
        np.add.at(f_ext, eq_map[:, 1], f_unit_eq[:, 1] * mult * REF_FORCE_SQUARE)

        converged = False
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
            if it >= 2 * STALL_WINDOW:
                recent_best = min(res_history[-STALL_WINDOW:])
                prior_best = min(res_history[-2 * STALL_WINDOW:-STALL_WINDOW])
                if recent_best >= (1.0 - STALL_PATIENCE_REL) * prior_best:
                    break
            K_ff = K[free_dofs, :][:, free_dofs]
            try:
                du_free = spsolve(K_ff.tocsc(), res_free)
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"  [{which}] step {step}: linear solve failed: {exc}")
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
        elif np.isfinite(best_rel) and best_rel < STALL_ACCEPT_REL_TOL:
            status = "stalled"
        else:
            status = "diverged"
        if not converged:
            if not np.all(np.isfinite(best_u)):
                raise RuntimeError(f"[{which}] step {step}: no finite iterate was ever produced.")
            u = best_u.copy()

        step_log.append({"step": step, "target": (mult, r), "iters": n_iter,
                          "status": status, "best_rel": float(best_rel)})
        if verbose or status != "converged":
            tag = {"converged": "OK", "stalled": "STALLED", "diverged": "DIVERGED"}[status]
            print(f"  [{which}] step {step:3d}/{len(targets)}  mult={mult:+.2f}  r={r:+.3f}  "
                  f"iters={n_iter:2d}  {tag}  best_rel={best_rel:.3e}", flush=True)

    fully_converged = all(s["status"] == "converged" for s in step_log) and len(step_log) == len(targets)
    ever_diverged = any(s["status"] == "diverged" for s in step_log)
    wall_time = time.perf_counter() - t_wall_start
    print(f"  [{which}] TOTAL wall={wall_time:.1f}s, material_calls~{n_material_calls}, "
          f"fully_converged={fully_converged}, ever_diverged={ever_diverged}", flush=True)

    return {"which": which, "step_log": step_log, "fully_converged": fully_converged,
            "ever_diverged": ever_diverged, "wall_time": wall_time}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--which", type=str, default="pann_certified")
    a = p.parse_args()
    run_force_trajectory(a.which, verbose=True)
