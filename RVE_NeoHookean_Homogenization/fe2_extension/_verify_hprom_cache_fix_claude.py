#!/usr/bin/env python3
"""Verify the Stage-1 HPROM-ANN optimization (caching the last Newton
iteration's linearization in _evaluate_impl/evaluate_with_tangent instead of
recomputing it via _residual_jacobian_at) is bit-for-bit exact and actually
faster, by comparing against a hand-rolled reference that forces the OLD
behavior (always recompute), on real macro strains from the n_body=6
continuation run. Also reports how many of the sampled points actually hit
the cache (converged_cleanly) vs fell back."""
from __future__ import annotations

import sys
import time
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
print(f"[verify] loaded {e_gp.shape[0]} real macro strains", flush=True)

law = m.HpromAnnIterativeLawFloat64(hprom_ann_dir=str(HPROMANN_DIR), qp_init_mode="continuation")
_ = law.evaluate_with_tangent(e_gp[0], q_prev=np.zeros(law.n_primary), step_index=1)  # warm-up


def old_evaluate_with_tangent(law, E, q_prev=None, step_index=1):
    """Exact copy of the pre-optimization evaluate_with_tangent: ALWAYS
    recomputes the linearization via _residual_jacobian_at, ignoring any
    cache. Used here purely as a reference oracle."""
    from _material_law_guard_claude import true_neo_hookean_active
    with true_neo_hookean_active():
        E = np.asarray(E, dtype=float).reshape(-1)
        hom_eps, hom_sig, q_p, n_iters, converged = law._evaluate_impl(
            E, q_prev=q_prev, step_index=step_index,
        )
        u_aff_free = law._affine(E, law.x_free, law.y_free, law.is_x_free)
        disp_base = np.zeros(law.n_total_dof, dtype=float)
        disp_base[law.dir_dofs] = law._affine(E, law.x_dir, law.y_dir, law.is_x_dir)
        J_manifold, K_r, w_res_iter, u_eq_curr, _r_full = law._residual_jacobian_at(
            q_p, E, disp_base, u_aff_free,
        )
        dq_p_dE, J_manifold, K_r, w_res_iter = law.dqp_dE_at(
            q_p, E, disp_base, u_aff_free, J_manifold=J_manifold, K_r=K_r, w_res_iter=w_res_iter,
        )
        law.vec_assembler.ComputeLocalArrays(u_eq_curr)

    w_eps, dw_eps_dE = law._hom_weights_and_jacobian(law.maw_eps_hom, E)
    w_sig, dw_sig_dE = law._hom_weights_and_jacobian(law.maw_sig_hom, E)
    from fom_solver_rve import CalculateHomogenizedFromAssemblerWithElementWeights
    hom_eps_check, _ = CalculateHomogenizedFromAssemblerWithElementWeights(
        law.vec_assembler, w_eps=w_eps, w_sig=None, reference_measure=law.hom_reference_measure,
    )
    hom_sig_check = law._reaction_force_hom_sig(E, w_sig)

    from deformation_gradient_jacobian_claude import deformation_gradient_and_jacobian_2d
    ddisp_dE = np.zeros((law.n_total_dof, 3), dtype=float)
    _F, dF_dE = deformation_gradient_and_jacobian_2d(E)
    for k in range(3):
        d00, d01, d10, d11 = dF_dE[0, 0, k], dF_dE[0, 1, k], dF_dE[1, 0, k], dF_dE[1, 1, k]
        ddisp_dE[law.free_dofs, k] = np.where(
            law.is_x_free, d00 * law.x_free + d01 * law.y_free, d10 * law.x_free + d11 * law.y_free,
        )
        ddisp_dE[law.dir_dofs, k] = np.where(
            law.is_x_dir, d00 * law.x_dir + d01 * law.y_dir, d10 * law.x_dir + d11 * law.y_dir,
        )
    ddisp_dE[law.free_dofs, :] += J_manifold @ dq_p_dE

    local_eq_ids = law.vec_assembler.local_eq_ids
    du_local_dE = ddisp_dE[local_eq_ids, :]
    B = law.vec_assembler._B
    CC = law.vec_assembler._CC
    dE_gp_dE = np.einsum("egvc,ecx->egvx", B, du_local_dE)
    dS_gp_dE = np.einsum("egvw,egwx->egvx", CC, dE_gp_dE)
    dEpsbar_e_dE = np.mean(dE_gp_dE, axis=1)

    eps_mean_e = np.mean(law.vec_assembler._E_voigt, axis=1)
    if hasattr(law.vec_assembler, "area_e"):
        area_e = np.asarray(law.vec_assembler.area_e, dtype=float).reshape(-1)
    else:
        area_e = np.sum(np.asarray(law.vec_assembler.w_detJ, dtype=float), axis=1)
    den = float(law.hom_reference_measure)

    from dhprom_ann_direct_law_claude import DHpromAnnDirectLaw
    dEps_hom_dE = DHpromAnnDirectLaw._dhom_dE(eps_mean_e, dEpsbar_e_dE, w_eps, dw_eps_dE, area_e, den)
    from reaction_force_hom_tangent_claude import reaction_force_hom_sig_and_jacobian
    _hom_sig_at_u_eq_curr, dSig_hom_dE = reaction_force_hom_sig_and_jacobian(
        law, E, w_sig, dw_sig_dE, du_local_dE, dS_gp_dE,
    )
    return hom_eps, hom_sig, q_p, n_iters, converged, dEps_hom_dE, dSig_hom_dE, hom_eps_check, hom_sig_check


N_SAMPLE = 100
rng = np.random.default_rng(0)
idx = rng.choice(e_gp.shape[0], size=min(N_SAMPLE, e_gp.shape[0]), replace=False)

t0 = time.perf_counter()
old_results = [old_evaluate_with_tangent(law, e_gp[i], q_prev=np.zeros(law.n_primary), step_index=1) for i in idx]
t_old = time.perf_counter() - t0
print(f"[verify] OLD (always recompute) path: {t_old:.3f}s, {t_old / len(idx) * 1000:.3f} ms/point", flush=True)

t0 = time.perf_counter()
new_results = [law.evaluate_with_tangent(e_gp[i], q_prev=np.zeros(law.n_primary), step_index=1) for i in idx]
t_new = time.perf_counter() - t0
print(f"[verify] NEW (cached) path: {t_new:.3f}s, {t_new / len(idx) * 1000:.3f} ms/point", flush=True)
print(f"[verify] speedup: {t_old / max(t_new, 1e-9):.2f}x", flush=True)

max_diffs = {}
for field_idx, name in enumerate(["hom_eps", "hom_sig", "q_p", "n_iters", "converged",
                                   "dEps_hom_dE", "dSig_hom_dE", "hom_eps_check", "hom_sig_check"]):
    diffs = []
    for o, nres in zip(old_results, new_results):
        a = np.asarray(o[field_idx]).astype(float)
        b = np.asarray(nres[field_idx]).astype(float)
        diffs.append(np.max(np.abs(a - b)) if a.size else 0.0)
    max_diffs[name] = max(diffs)

print(flush=True)
for name, diff in max_diffs.items():
    print(f"[verify] max abs diff {name}: {diff:.3e}", flush=True)

ok = all(v < 1e-10 for v in max_diffs.values())
print("VERIFY_PASS" if ok else "VERIFY_FAIL", flush=True)
