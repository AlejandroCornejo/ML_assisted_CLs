#!/usr/bin/env python3
"""Implicit Newmark-beta (beta=0.25, gamma=0.5 -- the unconditionally
stable, for linear systems, "average constant acceleration"/trapezoidal
choice) replacement for the explicit central-difference driver, which
showed a puzzling dt-independent blow-up time even for the consistent
ICNN law -- switching to an implicit scheme both sidesteps CFL concerns
by construction and lets us reuse our own already-validated static
Newton-loop machinery almost verbatim (the whole change is: add
M/(beta*dt^2) to the tangent, and to the residual).

Free-floating panel (no Dirichlet constraints), uniform initial velocity,
zero external force -- same physical setup as the explicit version and as
As'ad, Avery & Farhat (2022)'s own Section 4.3/Figure 9, just integrated
differently in time.

Newmark displacement-based formulation: given u_n, v_n, a_n and a
predictor u_pred = u_n + dt*v_n + dt^2*(0.5-beta)*a_n (known from the
previous step), the unknown u_{n+1} solves
    M*(u_{n+1}-u_pred)/(beta*dt^2) + f_int(u_{n+1}) - f_ext = 0
via Newton iteration with tangent M/(beta*dt^2) + K_tangent(u) -- K_tangent
being exactly what VectorizedAssembler.Assemble already returns."""
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
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from run_cruciform_fe2_claude import MATERIAL_FUNCS, RESIDUAL_ABS_TOL  # noqa: E402
import fom_solver_rve as fom  # noqa: E402
from run_square_panel_dynamic_consistency_claude import build_free_model_part, lumped_mass_vector  # noqa: E402

def _newmark_line_search_alpha(assembler, u_base, du, u_pred, m_vec, f_ext,
                                first_alpha=0.5, second_alpha=1.0, max_it=10,
                                min_alpha=1.0e-3, max_alpha=1.0, tol=0.5):
    """Same secant/interpolation scheme as run_cook_pann_claude.py's own
    _line_search_alpha, adapted to the Newmark residual (inertial term
    included) and defensive against invalid trial strain states raised by
    the material law during the search itself (treated as a very bad
    residual, not a crash -- the search backs off, exactly as
    enforce_residual_decrease does elsewhere this session)."""
    u_trial = u_base.copy()

    def eval_r(alpha):
        u_trial[:] = u_base + alpha * du
        try:
            _, rhs_int = assembler.Assemble(u_trial)
        except Exception:  # noqa: BLE001
            return 1.0e30
        a_trial = (u_trial - u_pred) / (BETA * DT ** 2)
        residual = m_vec * a_trial + rhs_int - f_ext
        return float(alpha * np.dot(du, residual))

    x1, x2 = float(first_alpha), float(second_alpha)
    r1, r2 = eval_r(x1), eval_r(x2)
    rmax = max(abs(r1), abs(r2))
    x = x2
    for _ in range(int(max_it)):
        rmin = min(abs(r1), abs(r2))
        x = (r1 * x2 - r2 * x1) / (r1 - r2) if abs(r1 - r2) > 1e-10 else min_alpha
        x = min(max(x, min_alpha), max_alpha)
        rf = eval_r(x)
        if rmin < tol * rmax or abs(rf) < tol * rmax:
            break
        if abs(r1) > abs(r2):
            r1, x1 = rf, x
        else:
            r2, x2 = r1, x1
            r1, x1 = rf, x
        rmax = max(rmax, abs(rf))
    return float(x)


BETA = 0.25
GAMMA = 0.5
V0 = np.array([1.0, 0.0])
DT = 5.0e-4  # can afford a much larger step than the explicit scheme needed
N_STEPS = 300  # -> T = 0.15s, beyond As'ad's own 0.05s reference window


def run_newmark_test(which, max_newton_iter=30, use_line_search=True, verbose_every=10):
    if which not in MATERIAL_FUNCS:
        raise ValueError(f"which must be one of {list(MATERIAL_FUNCS)}, got {which!r}")

    mp, coords, tris = build_free_model_part()
    n_dof, eq_map, ta = fom.SetUpDofEquationIdsAndDisplacementAdaptor(mp)

    fom._neo_hookean_pk2_2d_vectorized = MATERIAL_FUNCS[which]
    assembler = fom.VectorizedAssembler(mp, n_dof, eq_map, log_label=f"NewmarkConsistency[{which}]")

    m_vec = lumped_mass_vector(mp, n_dof, eq_map)
    assert np.all(m_vec > 0), "expected strictly positive lumped mass at every DOF"
    M_over_beta_dt2 = diags(m_vec / (BETA * DT ** 2))

    v0_eq = np.zeros(n_dof)
    v0_eq[eq_map[:, 0]] = V0[0]
    v0_eq[eq_map[:, 1]] = V0[1]

    u = np.zeros(n_dof)
    v = v0_eq.copy()
    _, f_int0 = assembler.Assemble(u)
    a = -f_int0 / m_vec  # f_ext=0 always
    f_ext = np.zeros(n_dof)

    u_rigid = np.zeros(n_dof)
    max_deviation_history = []
    ke_history = []
    t0 = time.perf_counter()
    blew_up_at = None
    n_newton_calls = 0

    for n in range(N_STEPS):
        t_next = (n + 1) * DT
        u_pred = u + DT * v + DT ** 2 * (0.5 - BETA) * a
        u_next = u.copy()  # Newton unknown, warm-started at u (small-step assumption)

        for it in range(1, max_newton_iter + 1):
            n_newton_calls += 1
            try:
                K, rhs_int = assembler.Assemble(u_next)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{which}] step {n}: invalid state at Newton it {it}: {exc}")
                blew_up_at = n
                break
            a_next = (u_next - u_pred) / (BETA * DT ** 2)
            residual = m_vec * a_next + rhs_int - f_ext
            res_norm = np.linalg.norm(residual)
            if res_norm < RESIDUAL_ABS_TOL or (it > 1 and res_norm < 1.0e-3 * res_norm0):
                break
            if it == 1:
                res_norm0 = max(res_norm, 1e-12)
            K_eff = (K + M_over_beta_dt2).tocsc()
            try:
                du = spsolve(K_eff, -residual)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{which}] step {n}: linear solve failed at Newton it {it}: {exc}")
                du = None
                blew_up_at = n
                break
            if du is None or not np.all(np.isfinite(du)):
                blew_up_at = n
                break
            alpha = 1.0
            if use_line_search:
                alpha = _newmark_line_search_alpha(assembler, u_next, du, u_pred, m_vec, f_ext)
            u_next = u_next + alpha * du
        if blew_up_at is not None or not np.all(np.isfinite(u_next)):
            blew_up_at = n
            break

        a_next = (u_next - u_pred) / (BETA * DT ** 2)
        v_next = v + DT * (1.0 - GAMMA) * a + DT * GAMMA * a_next

        u, v, a = u_next, v_next, a_next

        u_rigid[:] = v0_eq * t_next
        deviation = u - u_rigid
        max_dev = float(np.max(np.abs(deviation)))
        max_deviation_history.append(max_dev)
        ke = float(0.5 * np.sum(m_vec * v ** 2))
        ke_history.append(ke)

        if max_dev > 1.0e6 or not np.isfinite(ke):
            blew_up_at = n
            break

        if n % verbose_every == 0:
            print(f"  [{which}] step {n:4d}  t={t_next:.4f}s  max|dev from rigid|={max_dev:.4e}  "
                  f"KE={ke:.4e}  newton_iters={it}", flush=True)

    wall = time.perf_counter() - t0
    status = f"BLEW UP at step {blew_up_at}" if blew_up_at is not None else "completed all steps"
    print(f"[{which}] {status}, wall={wall:.1f}s, {n_newton_calls} total Newton iterations, "
          f"final max|dev|={max_deviation_history[-1] if max_deviation_history else float('nan'):.4e}, "
          f"peak max|dev|={max(max_deviation_history) if max_deviation_history else float('nan'):.4e}, "
          f"KE: initial={ke_history[0] if ke_history else float('nan'):.4e} "
          f"final={ke_history[-1] if ke_history else float('nan'):.4e}")
    return {"which": which, "max_deviation_history": max_deviation_history, "ke_history": ke_history,
            "blew_up_at": blew_up_at}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--which", type=str, default="pann_certified")
    a = p.parse_args()
    run_newmark_test(a.which)
