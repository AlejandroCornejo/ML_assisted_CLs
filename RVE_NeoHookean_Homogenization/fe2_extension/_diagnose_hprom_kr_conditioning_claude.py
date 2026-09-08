#!/usr/bin/env python3
"""Test the hypothesis that the remaining evaluate_with_tangent_batch vs
evaluate_with_tangent mismatch (after fixing the r_full/K_total_support
aliasing bugs) is genuine ill-conditioning of K_r amplifying the ~1e-15
relative difference between torch.func.hessian+vmap and
torch.autograd.functional.hessian, rather than a remaining bug: for each
of the 40 sample points, compute K_r's own condition number (via the OLD,
unbatched, reference path) and see if it correlates with the observed
dSig_hom_dE mismatch."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import hprom_ann_iterative_law_float64_claude as m  # noqa: E402

HPROMANN_DIR = HERE / "maw_dynamic_reaction_force_hpromann"
d = np.load(HERE / "cruciform_results_hprom_iterative_f64_continuation_claude.npz")
e_gp = d["e_gp"]

law = m.HpromAnnIterativeLawFloat64(hprom_ann_dir=str(HPROMANN_DIR), qp_init_mode="continuation")
_ = law.evaluate_with_tangent(e_gp[0], q_prev=np.zeros(law.n_primary), step_index=1)

N = 40
rng = np.random.default_rng(1)
idx = rng.choice(e_gp.shape[0], size=N, replace=False)
E_batch = e_gp[idx]
q_prev_zero = np.zeros((N, law.n_primary))

kr_cond = np.zeros(N)
kr_used_cache = np.zeros(N, dtype=bool)
old = []
for i in range(N):
    E = E_batch[i]
    _cache = {}
    hom_eps, hom_sig, q_p, n_iters, converged = law._evaluate_impl(E, q_prev=q_prev_zero[i], step_index=1, _cache_out=_cache)
    kr_used_cache[i] = bool(_cache)
    if _cache:
        K_r = _cache["K_r"]
    else:
        u_aff_free = law._affine(E, law.x_free, law.y_free, law.is_x_free)
        disp_base = np.zeros(law.n_total_dof, dtype=float)
        disp_base[law.dir_dofs] = law._affine(E, law.x_dir, law.y_dir, law.is_x_dir)
        _J, K_r, _w, _u, _r = law._residual_jacobian_at(q_p, E, disp_base, u_aff_free)
    kr_cond[i] = np.linalg.cond(K_r)
    old.append(law.evaluate_with_tangent(E, q_prev=q_prev_zero[i], step_index=1))

new = law.evaluate_with_tangent_batch(E_batch, q_prev_batch=q_prev_zero, step_index_batch=np.ones(N, dtype=int))
dsig_old = np.stack([o[6] for o in old])
dsig_new = new[6]

print(f"{'i':>3} {'used_cache':>10} {'cond(K_r)':>12} {'dsig_reldiff':>13}")
for i in range(N):
    s_diff = np.max(np.abs(dsig_old[i] - dsig_new[i])) / max(np.max(np.abs(dsig_old[i])), 1e-300)
    print(f"{i:>3} {str(kr_used_cache[i]):>10} {kr_cond[i]:>12.3e} {s_diff:>13.3e}")

corr = np.corrcoef(np.log10(kr_cond + 1e-300),
                    np.log10(np.array([
                        max(np.max(np.abs(dsig_old[i] - dsig_new[i])) / max(np.max(np.abs(dsig_old[i])), 1e-300), 1e-300)
                        for i in range(N)
                    ])))[0, 1]
print(f"\n[diag] correlation(log10 cond(K_r), log10 dsig_reldiff) = {corr:.3f}")
print(f"[diag] cond(K_r) range: min={kr_cond.min():.3e}, max={kr_cond.max():.3e}, median={np.median(kr_cond):.3e}")
print("DIAG_DONE_MARKER", flush=True)
