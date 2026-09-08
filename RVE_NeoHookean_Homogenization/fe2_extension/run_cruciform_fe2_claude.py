#!/usr/bin/env python3
"""Cruciform (plus-shaped) biaxial specimen driven by genuine FE^2 --
same material-law-swapping pattern as run_cook_hprom_ann_claude.py's
run_newton_fe2, but displacement-controlled (Dirichlet-ramped tip pulls)
instead of force-controlled, matching the RVE's own homogenization
solves' own loading convention (no external force term at all -- the
residual to drive to zero at every Newton iteration is just rhs_int on
the free dofs, exactly like core/fom_solver_rve.py's own FOM solves).

Default loading is exactly equibiaxial (same pull magnitude on all 4
arms), matching training trajectory_1's own direction (E11=E22,
gamma12=0) -- the most heavily-covered, safest direction in the RVE's
own training data.
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

from build_cruciform_mesh_claude import build_cruciform_mesh  # noqa: E402
from run_cook_pann_claude import _line_search_alpha  # noqa: E402  (generic Newton line search, material-law-agnostic)

import dhprom_ann_direct_law_float64_claude as dhprom_f64_module  # noqa: E402
import hprom_ann_iterative_law_float64_claude as hprom_iter_f64_module  # noqa: E402
import linear_hprom_iterative_law_float64_claude as linear_hprom_module  # noqa: E402
import pann_constitutive_law_claude as pann_law  # noqa: E402
from fom_nested_law_parallel_claude import fom_nested_pk2_2d_vectorized_parallel  # noqa: E402
from run_cook_linear_hprom_claude import LinearHpromContinuationWrapper  # noqa: E402
import linear_hprom_law_parallel_claude as linear_hprom_parallel_module  # noqa: E402


def _make_pann_material_func(which):
    def _f(e_voigt, young=None, poisson=None):
        return pann_law.pann_pk2_2d_vectorized(e_voigt, which=which)
    return _f


def _make_fom_nested_material_func(n_workers=16):
    def _f(e_voigt, young=None, poisson=None):
        return fom_nested_pk2_2d_vectorized_parallel(e_voigt, n_workers=n_workers, verbose=True)
    return _f


class HpromIterativeContinuationWrapper:
    """Fixes a real bug found this session: hprom_ann_iterative_pk2_2d_
    vectorized_consistent_float64 never threads q_prev/step_index through
    to evaluate_with_tangent, so every macro Gauss-point call is a cold
    mu-affine start regardless of the law's own qp_init_mode. On the
    cruciform's own hardest corner state (large combined E11/E22/gamma12)
    this caused the macro Newton loop's last (most extreme) load step to
    diverge -- a single-jump cold start was too far from that state's own
    solution for the internal correction to reach, even though a targeted
    single-point test confirmed the state itself is solvable via genuine
    path continuation. This wrapper tracks each macro Gauss point's own
    last-used q_p (by its stable positional index in the vectorized call,
    consistent across calls since the same VectorizedAssembler instance
    is reused for the whole run) and forces the law's own qp_init_mode=
    'continuation' branch (step_index=2) once a q_prev exists -- verified
    this session: fixes the cruciform's own step-20 divergence (4 iters,
    converged, same as the easy steps) at validation scale.
    """

    def __init__(self, law):
        self.law = law
        self.q_prev_by_point = {}

    def __call__(self, e_voigt, young=None, poisson=None):
        e_voigt = np.asarray(e_voigt, dtype=float)
        n = e_voigt.shape[0]
        q_prev_batch = np.zeros((n, self.law.n_primary), dtype=float)
        step_index_batch = np.ones(n, dtype=int)
        for i in range(n):
            q_prev = self.q_prev_by_point.get(i)
            if q_prev is not None:
                q_prev_batch[i] = q_prev
                step_index_batch[i] = 2

        # evaluate_with_tangent_batch (this session's addition) batches the
        # decoder-Jacobian/weighted-Hessian torch calls across whichever of
        # these n points are still mid-Newton-correction at each shared
        # iteration depth -- verified this session to reproduce a per-point
        # loop of evaluate_with_tangent to ~1e-13 relative error (n_iters
        # matching exactly) on real macro strains, ~3x faster. The per-
        # point warm-start/commit-only-if-converged logic below is
        # unchanged from before that change.
        _, S, q_p_batch, _n_it, converged, _, CC, _, _ = self.law.evaluate_with_tangent_batch(
            e_voigt, q_prev_batch=q_prev_batch, step_index_batch=step_index_batch,
        )

        n_nonconverged = 0
        for i in range(n):
            if not converged[i]:
                n_nonconverged += 1
            else:
                # Only commit a CONVERGED q_p as the next warm start. Every
                # macro Newton iteration triggers several Assemble() calls
                # (the main one, plus _line_search_alpha's own exploratory
                # trial evaluations at various candidate alphas, most of
                # which get rejected) -- committing unconditionally lets a
                # rejected trial's own (possibly garbage) q_p poison the
                # warm start for the NEXT call, compounding across calls
                # until the macro strain fed in is no longer even a valid
                # state (found this session: a full blowup at n_body=9,
                # traced to exactly this). A non-converged q_p is never
                # committed, so a bad trial can't propagate; the point
                # simply keeps its last known-good warm start instead.
                self.q_prev_by_point[i] = q_p_batch[i]
        if n_nonconverged:
            print(f"    [hprom-continuation] {n_nonconverged}/{n} points did not converge internally this call")
        return S, CC


def make_dhprom_ann_parallel_material_func(dhpromann_dir, n_workers=16):
    """Parallel counterpart for D-HPROM-ANN, no continuation wrapper
    needed (unlike HPROM-ANN/Linear-HPROM): D-HPROM-ANN carries no
    per-point state across calls at all (q_p is a direct closed-form
    affine map of E), so its own dhprom_ann_pk2_2d_vectorized_consistent_
    float64 is already a pure function of E_flat -- see dhprom_ann_law_
    parallel_claude.py's own module docstring.

    IMPORTANT: the calling driver script must call dhprom_ann_parallel_
    module.ensure_persistent_executor(n_workers=...) BEFORE importing this
    module, same fork-after-Kratos/torch-threading discipline as every
    other *_law_parallel_claude.py this session."""
    import dhprom_ann_law_parallel_claude as dhprom_ann_parallel_module
    return dhprom_ann_parallel_module.make_dhprom_ann_parallel_material_func(
        dhpromann_dir, n_workers=n_workers,
    )


def make_fom_nested_consistent_parallel_material_func(n_workers=16):
    """Parallel counterpart for the CORRECTED true-FOM-FE2 law
    (fom_nested_consistent_law_claude.py: reaction-force stress +
    analytic implicit-function-theorem tangent, replacing fom_nested_
    law_claude.py's naive-volume-average stress and 6-extra-full-solve
    FD tangent). No continuation wrapper needed, same reasoning as
    D-HPROM-ANN: pure function of E_flat, no per-point state at all.

    IMPORTANT: the calling driver script must call fom_nested_consistent_
    parallel_module.ensure_persistent_executor(n_workers=...) BEFORE
    importing this module, same fork-after-Kratos-threading discipline as
    every other *_law_parallel_claude.py this session."""
    import fom_nested_consistent_law_parallel_claude as fom_nested_consistent_parallel_module

    def _f(e_voigt, young=None, poisson=None):
        return fom_nested_consistent_parallel_module.fom_nested_consistent_pk2_2d_vectorized_parallel(
            e_voigt, n_workers=n_workers, verbose=True,
        )
    return _f


def make_hprom_continuation_material_func(hprom_ann_dir):
    """Builds (or reuses, via the module's own singleton) the
    HpromAnnIterativeLawFloat64 instance with qp_init_mode='continuation'
    actually honored, and wraps it for per-point warm-starting. Register
    the RETURNED callable into MATERIAL_FUNCS under a chosen key before
    calling run_newton_fe2_cruciform with that key -- a fresh call builds
    a fresh (empty q_prev history) wrapper, appropriate for a new run."""
    law = hprom_iter_f64_module.get_law_float64(hprom_ann_dir=str(hprom_ann_dir), qp_init_mode="continuation")
    assert law.qp_init_mode == "continuation", (
        f"qp_init_mode is {law.qp_init_mode!r}, not 'continuation' -- singleton already built elsewhere first?"
    )
    return HpromIterativeContinuationWrapper(law)


def make_hprom_ann_parallel_continuation_material_func(hprom_ann_dir, n_workers=16):
    """Parallel counterpart to make_hprom_continuation_material_func:
    splits the macro Gauss points across a persistent process pool, each
    worker batching its own chunk internally via evaluate_with_tangent_
    batch (see hprom_ann_law_parallel_claude.py's own module docstring).

    IMPORTANT: the calling driver script must call
    hprom_ann_parallel_module.ensure_persistent_executor(n_workers=...)
    BEFORE importing this module, same fork-after-Kratos/torch-threading
    discipline as every other *_law_parallel_claude.py this session."""
    import hprom_ann_law_parallel_claude as hprom_ann_parallel_module
    return hprom_ann_parallel_module.HpromAnnParallelContinuationWrapper(
        hprom_ann_dir=hprom_ann_dir, n_workers=n_workers,
    )


def make_linear_hprom_continuation_material_func():
    """Same per-point warm-start discipline as
    make_hprom_continuation_material_func, for the pure-POD (no ANN)
    Linear-HPROM law -- reuses LinearHpromContinuationWrapper unmodified
    from run_cook_linear_hprom_claude.py (already law-agnostic w.r.t. the
    caller, no qp_init_mode override needed since this law's own q_prev/
    step_index contract is already native)."""
    law = linear_hprom_module.get_law()
    return LinearHpromContinuationWrapper(law)


def make_linear_hprom_parallel_continuation_material_func(n_workers=16):
    """Parallel counterpart: splits the macro Gauss points across a
    persistent process pool instead of looping serially in this process
    (see linear_hprom_law_parallel_claude.py's own module docstring for
    why -- this law's dominant costs are native Kratos calls with no
    torch/vmap batching lever, unlike D-HPROM-ANN/HPROM-ANN).

    IMPORTANT: the calling driver script must call
    linear_hprom_parallel_module.ensure_persistent_executor(n_workers=...)
    BEFORE importing this module (run_cruciform_fe2_claude.py), matching
    the same fork-after-Kratos-threading discipline already established
    for fom_nested_law_parallel_claude.py -- this factory does not call it
    itself since by the time it runs, this file has already imported
    fom_solver_rve/KratosMultiphysics."""
    return linear_hprom_parallel_module.LinearHpromParallelContinuationWrapper(n_workers=n_workers)


MATERIAL_FUNCS = {
    "dhprom_f64_consistent": dhprom_f64_module.dhprom_ann_pk2_2d_vectorized_consistent_float64,
    "hprom_iterative_f64_consistent": hprom_iter_f64_module.hprom_ann_iterative_pk2_2d_vectorized_consistent_float64,
    "pann_certified": _make_pann_material_func("certified"),
    "pann_free": _make_pann_material_func("free"),
    "pann_ickan": _make_pann_material_func("ickan"),
    "pann_regression": _make_pann_material_func("regression"),
    "fom_nested": _make_fom_nested_material_func(n_workers=16),
}

N_STEPS = 20
RESIDUAL_REL_TOL = 1.0e-4
RESIDUAL_ABS_TOL = 1.0e-9
STALL_ACCEPT_REL_TOL = 1.0e-2
STALL_WINDOW = 2
STALL_PATIENCE_REL = 0.10


def build_cruciform_model_part(n_body, n_arm_len, L_body=12.0, arm_width_fraction=2.0 / 3.0, L_arm=8.0,
                                fix_tips=True):
    """fix_tips=True (default, every existing caller): the displacement-
    controlled protocol, tip nodes Kratos-level Fix()ed in their own pull
    direction. fix_tips=False: only the center node is fixed -- for a
    force-controlled driver that applies a nodal force at the tips
    instead, leaving them free."""
    coords, tris, tip_nodes, center_node = build_cruciform_mesh(
        n_body=n_body, n_arm_len=n_arm_len, L_body=L_body, arm_width_fraction=arm_width_fraction, L_arm=L_arm,
    )
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
    if fix_tips:
        for nid in tip_nodes["px"] + tip_nodes["mx"]:
            mp.GetNode(nid + 1).Fix(KM.DISPLACEMENT_X)
        for nid in tip_nodes["py"] + tip_nodes["my"]:
            mp.GetNode(nid + 1).Fix(KM.DISPLACEMENT_Y)

    return mp, coords, tris, tip_nodes, center_node


def run_newton_fe2_cruciform(
    which, n_body=6, n_arm_len=4, verbose=True, use_line_search=True, save_npz=True,
    max_newton_iter=30, n_steps=N_STEPS, delta_x_final=1.2, delta_y_final=1.2,
    stall_accept_rel_tol=STALL_ACCEPT_REL_TOL, stall_window=STALL_WINDOW, stall_patience_rel=STALL_PATIENCE_REL,
    max_steps_to_run=None, iter_callback=None, alpha_callback=None,
    enforce_residual_decrease=False, residual_decrease_min_alpha=1.0e-4,
):
    """max_steps_to_run: if set, stop after this many steps instead of all
    n_steps -- each executed step still uses frac=step/n_steps (the SAME
    per-step increment size as a full n_steps run), so e.g.
    max_steps_to_run=1 with n_steps=20 reproduces step 1 of the standard
    20-step ramp exactly, not a single step covering the full displacement
    (which n_steps=1 alone would do). fully_converged is correctly False
    for a partial run (len(step_log) < n_steps), by the existing check
    below -- no special-casing needed for that.

    iter_callback: if set, called as iter_callback(step, it, u, res_norm,
    assembler) right after every Assemble() call (before the convergence
    check), so it sees every iterate including the last. assembler._E_voigt/
    _S_voigt ((n_elems, n_gauss, 3)) hold this exact iterate's per-Gauss-point
    macro strain/stress. Purely a read hook -- must not mutate u/assembler.

    alpha_callback: if set, called as alpha_callback(step, it, alpha,
    du_free, free_dofs) right after the line-search alpha is chosen (or set
    to 1.0 without search), before it's applied to u -- lets a caller record
    the actual step size/direction magnitude chosen each iteration. Purely a
    read hook -- must not mutate its arguments.

    enforce_residual_decrease: default False (zero behavior change for any
    existing caller -- e.g. all already-reported paper results). When True,
    adds a safety net _line_search_alpha itself doesn't provide (it's a
    directional-derivative secant search, not a strict ||res|| decrease
    guarantee): after alpha is chosen, trial-evaluate it and halve alpha
    (down to residual_decrease_min_alpha) until the resulting ||res|| is
    actually smaller than this iteration's own res_norm, or until the floor
    is hit (accepted anyway; the existing best_u/stall/diverged bookkeeping
    below still applies, so this never raises on its own). A trial that
    itself raises (e.g. an invalid Green-Lagrange strain state) is treated
    as an infinitely bad trial -- caught and backtracked past, not
    propagated. Costs exactly one extra Assemble() call per iteration in
    the common case where the original alpha already decreases the
    residual (i.e. every already-converging law/mesh combination checked
    this session)."""
    if which not in MATERIAL_FUNCS:
        raise ValueError(f"which must be one of {list(MATERIAL_FUNCS)}, got {which!r}")
    mp, coords, tris, tip_nodes, center_node = build_cruciform_model_part(n_body, n_arm_len)
    n_dof, eq_map, ta = fom.SetUpDofEquationIdsAndDisplacementAdaptor(mp)

    fom._neo_hookean_pk2_2d_vectorized = MATERIAL_FUNCS[which]
    assembler = fom.VectorizedAssembler(mp, n_dof, eq_map, log_label=f"CruciformFE2[{which}]")

    dirichlet_dofs = set(eq_map[center_node].tolist())
    px_x_dofs = eq_map[tip_nodes["px"], 0].tolist()
    mx_x_dofs = eq_map[tip_nodes["mx"], 0].tolist()
    py_y_dofs = eq_map[tip_nodes["py"], 1].tolist()
    my_y_dofs = eq_map[tip_nodes["my"], 1].tolist()
    dirichlet_dofs.update(px_x_dofs + mx_x_dofs + py_y_dofs + my_y_dofs)
    free_dofs = np.array([d for d in range(n_dof) if d not in dirichlet_dofs])
    f_ext = np.zeros(n_dof)

    u = np.zeros(n_dof)
    step_log = []
    residual_histories = {}
    t_wall_start = time.perf_counter()
    n_material_calls = 0

    for step in range(1, n_steps + 1):
        frac = step / n_steps
        u[px_x_dofs] = +delta_x_final * frac
        u[mx_x_dofs] = -delta_x_final * frac
        u[py_y_dofs] = +delta_y_final * frac
        u[my_y_dofs] = -delta_y_final * frac

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
            if iter_callback is not None:
                iter_callback(step, it, u, res_norm, assembler)
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

            if enforce_residual_decrease:
                while True:
                    u_trial = u.copy()
                    u_trial[free_dofs] += alpha * du_free
                    try:
                        _, rhs_trial = assembler.Assemble(u_trial)
                        n_material_calls += 1
                        res_trial_norm = np.linalg.norm((rhs_trial + f_ext)[free_dofs])
                    except Exception as exc:  # noqa: BLE001
                        res_trial_norm = np.inf
                        if verbose:
                            print(f"      [{which}] step {step}: alpha={alpha:.2e} trial raised "
                                  f"{type(exc).__name__}: {exc}; backtracking")
                    if np.isfinite(res_trial_norm) and res_trial_norm < res_norm:
                        break
                    if alpha <= residual_decrease_min_alpha:
                        if verbose:
                            print(f"      [{which}] step {step}: residual-decrease backtracking hit floor "
                                  f"alpha={alpha:.2e} without improving |res|={res_norm:.3e}")
                        break
                    alpha *= 0.5

            if alpha_callback is not None:
                alpha_callback(step, it, alpha, du_free, free_dofs)
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

    # One extra Assemble() at the final converged u, purely to report the
    # tip reaction forces (sum of internal force over each tip's own dofs;
    # at equilibrium rhs_int[free_dofs]~=0, so this is the actual force the
    # actuator would need to apply) -- a physically meaningful GLOBAL QoI
    # for step-count convergence checks, complementing the per-Gauss-point
    # e_gp/s_gp field already saved above (which catches local hotspots a
    # global scalar could mask).
    _, rhs_int_final = assembler.Assemble(u)
    n_material_calls += 1
    reaction_px = float(np.sum(rhs_int_final[px_x_dofs]))
    reaction_mx = float(np.sum(rhs_int_final[mx_x_dofs]))
    reaction_py = float(np.sum(rhs_int_final[py_y_dofs]))
    reaction_my = float(np.sum(rhs_int_final[my_y_dofs]))

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
            HERE / f"cruciform_results_{which}_claude.npz",
            coords=coords, tris=tris, u_nodal=np.stack([u[eq_map[:, 0]], u[eq_map[:, 1]]], axis=1),
            e_gp=e_flat, s_gp=s_flat,
            iters_per_step=np.array([s["iters"] for s in step_log]),
            converged_per_step=np.array([s["converged"] for s in step_log]),
            status_per_step=np.array([s["status"] for s in step_log]),
            best_rel_per_step=np.array([s["best_rel"] for s in step_log]),
            residual_history_step1=residual_histories.get(1, np.zeros(0)),
            fully_converged=fully_converged, ever_diverged=ever_diverged,
            delta_x_final=delta_x_final, delta_y_final=delta_y_final,
            reaction_px=reaction_px, reaction_mx=reaction_mx,
            reaction_py=reaction_py, reaction_my=reaction_my,
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
        "reaction_px": reaction_px, "reaction_mx": reaction_mx,
        "reaction_py": reaction_py, "reaction_my": reaction_my,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--which", default="dhprom_f64_consistent", choices=list(MATERIAL_FUNCS))
    p.add_argument("--n-body", type=int, default=6)
    p.add_argument("--n-arm-len", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=N_STEPS)
    p.add_argument("--delta", type=float, default=1.2)
    a = p.parse_args()

    res = run_newton_fe2_cruciform(
        a.which, n_body=a.n_body, n_arm_len=a.n_arm_len, n_steps=a.n_steps,
        delta_x_final=a.delta, delta_y_final=a.delta, verbose=True, use_line_search=True, save_npz=True,
    )
    print("\n=== summary ===")
    print(f"{res['which']}: fully_converged={res['fully_converged']}, ever_diverged={res['ever_diverged']}, "
          f"E11={res['e11_range']}, E22={res['e22_range']}, g12={res['g12_range']}, wall={res['wall_time']:.1f}s")
