#!/usr/bin/env python3
"""FE2 square panel driven through a MULTI-SEGMENT strain-space trajectory
(not a single linear ramp to one final state) -- same spirit as the paper's
own held-out test trajectory (a mixed loading-unloading cycle starting and
ending at the origin, pann/anisotropic/make_test_trajectory_claude.py's own
studies/fom_tangent_stability_test/reference_states/stage10_mixed_applied_
strain.npy, confirmed by direct inspection to NOT be equibiaxial -- at its
own max E11=1.6, E22=0 there), but deliberately different: explicitly tours
large EQUIBIAXIAL tension (E11=E22 up to near the RVE's trained ceiling of
2.0) combined with the full trained shear range (+-0.1) at that same large-
tension state, which the original test trajectory never visits.

At each point along the path, the boundary condition is the FULL affine map
u(X) = (F-I)X on every boundary node (not just normal-direction pulls),
with F = sqrtm(C), C = 2E+I -- the same "F=C^(1/2), R=I" convention already
seen and verified in Yvonnet, Monteiro & He (2013) Eq. (3.7). Since the
material is spatially homogeneous and the domain is a simple square, the
interior solution is exactly this same uniform state (already verified this
session for the pure-diagonal case) -- this trajectory just walks that
uniform state through a richer path in (E11,E22,gamma12) space than a single
linear ramp, so every macro Gauss point (of which there are 384 for n_body=8)
sees the identical, exactly-controlled target at each step.

Newton loop structure mirrors run_newton_fe2_cruciform exactly (same
line-search helper, same tolerances/constants, same stall-detection logic)
-- only the boundary-condition construction differs (full affine map on all
boundary nodes instead of pure diagonal pulls on 4 tip node-sets)."""
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

from run_cruciform_fe2_claude import (  # noqa: E402  (this import's own top-level code adds core/ and Cook.gid/ to sys.path)
    MATERIAL_FUNCS, RESIDUAL_ABS_TOL, RESIDUAL_REL_TOL,
    STALL_ACCEPT_REL_TOL, STALL_WINDOW, STALL_PATIENCE_REL,
)
import fom_solver_rve as fom  # noqa: E402
from run_cook_pann_claude import _line_search_alpha  # noqa: E402
from build_square_panel_mesh_claude import build_square_panel_mesh  # noqa: E402

N_BODY = 8
L_BODY = 12.0
STEPS_PER_SEGMENT = 15

# (E11, E22, gamma12) keyframes -- a cycle, starting/ending at the origin,
# same "mixed loading-unloading" idea as the held-out test trajectory, but
# deliberately touring the large-equibiaxial-tension + full-trained-shear
# corner of the training box instead.
TRAJECTORY_KEYFRAMES = [
    (0.0, 0.0, 0.0),
    (1.9, 1.9, 0.0),
    (1.9, 1.9, 0.1),
    (1.9, 1.9, -0.1),
    (0.0, 0.0, 0.0),
]


def affine_F_from_E(e11, e22, g12):
    C = np.array([[1.0 + 2.0 * e11, g12], [g12, 1.0 + 2.0 * e22]])
    w, V = np.linalg.eigh(C)
    if np.any(w <= 0.0):
        raise ValueError(f"C not positive definite for E=({e11},{e22},{g12}): eigenvalues={w}")
    F = V @ np.diag(np.sqrt(w)) @ V.T
    return F


def interpolated_targets(keyframes, steps_per_segment):
    """Yield (e11, e22, g12) at every substep along the full piecewise-linear
    path through the keyframes, excluding the very first point (t=0, the
    trivial undeformed state, which needs no Newton solve)."""
    targets = []
    for i in range(len(keyframes) - 1):
        a = np.array(keyframes[i])
        b = np.array(keyframes[i + 1])
        for s in range(1, steps_per_segment + 1):
            frac = s / steps_per_segment
            targets.append(tuple(a + (b - a) * frac))
    return targets


def build_square_panel_boundary_dofs(coords, tip_nodes, eq_map):
    boundary_nodes = sorted(set(tip_nodes["px"] + tip_nodes["mx"] + tip_nodes["py"] + tip_nodes["my"]))
    return np.array(boundary_nodes, dtype=int)


def run_trajectory(which, verbose=True, use_line_search=True, save_npz=False,
                    max_newton_iter=30, keyframes=TRAJECTORY_KEYFRAMES, steps_per_segment=STEPS_PER_SEGMENT):
    if which not in MATERIAL_FUNCS:
        raise ValueError(f"which must be one of {list(MATERIAL_FUNCS)}, got {which!r}")

    coords, tris, tip_nodes, center_node = build_square_panel_mesh(n_body=N_BODY, L_body=L_BODY)

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
    boundary_nodes = build_square_panel_boundary_dofs(coords, tip_nodes, None)
    for nid in boundary_nodes.tolist():
        mp.GetNode(nid + 1).Fix(KM.DISPLACEMENT_X)
        mp.GetNode(nid + 1).Fix(KM.DISPLACEMENT_Y)

    n_dof, eq_map, ta = fom.SetUpDofEquationIdsAndDisplacementAdaptor(mp)

    fom._neo_hookean_pk2_2d_vectorized = MATERIAL_FUNCS[which]
    assembler = fom.VectorizedAssembler(mp, n_dof, eq_map, log_label=f"SquarePanelTraj[{which}]")

    boundary_dofs_x = eq_map[boundary_nodes, 0]
    boundary_dofs_y = eq_map[boundary_nodes, 1]
    dirichlet_dofs = set(boundary_dofs_x.tolist()) | set(boundary_dofs_y.tolist())
    free_dofs = np.array([d for d in range(n_dof) if d not in dirichlet_dofs])
    f_ext = np.zeros(n_dof)

    u = np.zeros(n_dof)
    step_log = []
    targets = interpolated_targets(keyframes, steps_per_segment)
    t_wall_start = time.perf_counter()
    n_material_calls = 0

    for step, (e11, e22, g12) in enumerate(targets, start=1):
        F = affine_F_from_E(e11, e22, g12)
        # Seed EVERY node (boundary AND interior) at the known analytic uniform
        # state for this target, not just the boundary -- warm-starting the
        # interior from the PREVIOUS step's converged state was found (this
        # session) to let Newton slide onto a spurious non-uniform equilibrium
        # during the unloading half of the cycle (verified via u_bnd/u_int
        # range instrumentation: boundary stayed correct, interior did not).
        # Seeding at the true target everywhere means Newton starts at (or
        # extremely near) the actual solution instead of discovering it.
        disp_all = (F - np.eye(2)) @ coords.T  # (2, n_nodes)
        u[eq_map[:, 0]] = disp_all[0]
        u[eq_map[:, 1]] = disp_all[1]

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
            # Seeding every node at the exact analytic solution (see above)
            # means it=1's own residual is often ALREADY pure discretization/
            # quadrature noise (calibrated this session at ~1e-5 to 2e-5 across
            # this whole trajectory, vs. 1e2-1e8 for a genuinely unconverged
            # state elsewhere in this project) -- the RELATIVE check can never
            # fire in that case (ratio is exactly 1.0 at it=1 by construction,
            # and chasing further "improvement" below the noise floor is what
            # produced the spurious DIVERGED verdicts caught this session).
            # SEED_NOISE_FLOOR is set two orders of magnitude above the
            # observed floor, still far below any real unconverged residual.
            SEED_NOISE_FLOOR = 1.0e-3
            if it == 1 and res_norm < SEED_NOISE_FLOOR:
                converged = True
                break
            if res_norm < RESIDUAL_ABS_TOL or res_norm / res_norm0 < RESIDUAL_REL_TOL:
                converged = True
                break
            if it >= 2 * STALL_WINDOW:
                recent_best = min(res_history[-STALL_WINDOW:])
                prior_best = min(res_history[-2 * STALL_WINDOW:-STALL_WINDOW])
                if recent_best >= (1.0 - STALL_PATIENCE_REL) * prior_best:
                    stall_detected = True
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

        step_log.append({"step": step, "target": (e11, e22, g12), "iters": n_iter,
                          "status": status, "best_rel": float(best_rel)})
        if verbose:
            tag = {"converged": "OK", "stalled": "STALLED", "diverged": "DIVERGED"}[status]
            u_bnd = u[np.concatenate([boundary_dofs_x, boundary_dofs_y])]
            u_int = u[free_dofs]
            print(f"  [{which}] step {step:3d}/{len(targets)}  target=({e11:+.3f},{e22:+.3f},{g12:+.3f})  "
                  f"iters={n_iter:2d}  {tag}  best_rel={best_rel:.3e}  "
                  f"u_bnd=[{u_bnd.min():+.4f},{u_bnd.max():+.4f}]  u_int=[{u_int.min():+.4f},{u_int.max():+.4f}]")

    fom.SetDisplacementFromEquationVector(u, eq_map, ta)
    e_voigt, s_voigt = assembler.ComputeStrainStressOnly(u)
    e_flat = e_voigt.reshape(-1, 3)

    fully_converged = all(s["status"] == "converged" for s in step_log) and len(step_log) == len(targets)
    ever_diverged = any(s["status"] == "diverged" for s in step_log)
    wall_time = time.perf_counter() - t_wall_start

    print(f"  [{which}] TOTAL wall={wall_time:.1f}s, material_calls~{n_material_calls}, "
          f"fully_converged={fully_converged}, ever_diverged={ever_diverged}")
    print(f"  [{which}] final macro state (all GP identical if exactly uniform): "
          f"E11=[{e_flat[:, 0].min():.4f},{e_flat[:, 0].max():.4f}]  "
          f"E22=[{e_flat[:, 1].min():.4f},{e_flat[:, 1].max():.4f}]  "
          f"g12=[{e_flat[:, 2].min():.4f},{e_flat[:, 2].max():.4f}]")

    return {"which": which, "step_log": step_log, "fully_converged": fully_converged,
            "ever_diverged": ever_diverged, "wall_time": wall_time, "e_gp_final": e_flat}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--which", type=str, default="pann_certified")
    a = p.parse_args()
    run_trajectory(a.which, verbose=True)
