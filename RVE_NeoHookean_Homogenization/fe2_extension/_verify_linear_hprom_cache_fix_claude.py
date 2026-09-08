#!/usr/bin/env python3
"""Verify the Linear-HPROM redundant-recompute cache fix (skip the second
res_assembler.Assemble call when the Newton loop broke via the res_norm
condition; unconditionally skip the second vec_assembler.ComputeLocalArrays
call) is bit-for-bit exact and faster, by comparing against a hand-rolled
reference that forces the OLD (always-recompute) behavior, on real macro
strains reused from the HPROM-ANN continuation run."""
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

import linear_hprom_iterative_law_float64_claude as m  # noqa: E402

d = np.load(HERE / "cruciform_results_hprom_iterative_f64_continuation_claude.npz")
e_gp = d["e_gp"]
print(f"[verify] loaded {e_gp.shape[0]} real macro strains", flush=True)

law = m.LinearHpromIterativeLawFloat64()
_ = law.evaluate_with_tangent(e_gp[0], q_prev=np.zeros(law.n_primary), step_index=1)  # warm-up


def old_evaluate_with_tangent(law, E, q_prev=None, step_index=1):
    """Exact copy of the pre-optimization evaluate_with_tangent: ALWAYS
    recomputes via a fresh res_assembler.Assemble + vec_assembler.
    ComputeLocalArrays call, ignoring any cache. Reference oracle."""
    from _material_law_guard_claude import true_neo_hookean_active
    from deformation_gradient_jacobian_claude import deformation_gradient_and_jacobian_2d
    from maw_hom_weight_jacobian_claude import maw_hom_weight_and_jacobian_single_model
    from reaction_force_hom_tangent_claude import reaction_force_hom_sig_and_jacobian
    from fom_solver_rve import InitializeNonLinearIteration, FinalizeNonLinearIteration

    with true_neo_hookean_active():
        E = np.asarray(E, dtype=float).reshape(-1)
        hom_eps, hom_sig, q_p, n_iters, converged = law._evaluate_impl(E, q_prev=q_prev, step_index=step_index)

        u_aff_free = law._affine(E, law.x_free, law.y_free, law.is_x_free)
        disp_base = np.zeros(law.n_total_dof, dtype=float)
        disp_base[law.dir_dofs] = law._affine(E, law.x_dir, law.y_dir, law.is_x_dir)
        u_free = u_aff_free + law.phi_f @ q_p
        u_eq_curr = disp_base.copy()
        u_eq_curr[law.free_dofs] = u_free

        phi_full, phi_full_T = law._phi_full, law._phi_full_T

        InitializeNonLinearIteration(law.entities, law.mp.ProcessInfo)
        K_hp, _rhs_hp = law.res_assembler.Assemble(u_eq_curr)
        FinalizeNonLinearIteration(law.entities, law.mp.ProcessInfo)
        K_r = phi_full_T @ (K_hp @ phi_full)

        _F, dF_dE = deformation_gradient_and_jacobian_2d(E)
        du_aff_dE = np.zeros((law.n_total_dof, 3), dtype=float)
        for k in range(3):
            d00, d01, d10, d11 = dF_dE[0, 0, k], dF_dE[0, 1, k], dF_dE[1, 0, k], dF_dE[1, 1, k]
            du_aff_dE[law.free_dofs, k] = np.where(
                law.is_x_free, d00 * law.x_free + d01 * law.y_free, d10 * law.x_free + d11 * law.y_free,
            )
            du_aff_dE[law.dir_dofs, k] = np.where(
                law.is_x_dir, d00 * law.x_dir + d01 * law.y_dir, d10 * law.x_dir + d11 * law.y_dir,
            )

        support_assembler = law.res_assembler
        local_eq_ids_support = support_assembler.local_eq_ids
        du_local_dE_support = du_aff_dE[local_eq_ids_support, :]
        K_total_support = support_assembler._K_total
        d_neg_fint_dE_support = -np.einsum("eij,ejk->eik", K_total_support, du_local_dE_support)

        dr_full_dE = np.zeros((law.n_total_dof, 3), dtype=float)
        for k in range(3):
            np.add.at(dr_full_dE[:, k], support_assembler.rows_R, d_neg_fint_dE_support[:, :, k].reshape(-1))

        dG_dE = phi_full_T @ dr_full_dE
        dq_p_dE = law._solve_reduced_system(K_r, dG_dE, law.regularization) if dG_dE.ndim == 1 else np.linalg.solve(K_r, dG_dE)

        law.vec_assembler.ComputeLocalArrays(u_eq_curr)

    w_sig, dw_sig_dE = maw_hom_weight_and_jacobian_single_model(
        E, law.maw_sig_hom, n_elem_reference=law.n_elem_reference,
        n_current_elements=law.n_current_elements, full_to_local=law.full_to_local_hom,
    )
    hom_sig_check = law._reaction_force_hom_sig(E, w_sig)

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
    ddisp_dE[law.free_dofs, :] += law.phi_f @ dq_p_dE

    local_eq_ids = law.vec_assembler.local_eq_ids
    du_local_dE = ddisp_dE[local_eq_ids, :]
    B = law.vec_assembler._B
    CC = law.vec_assembler._CC
    dE_gp_dE = np.einsum("egvc,ecx->egvx", B, du_local_dE)
    dS_gp_dE = np.einsum("egvw,egwx->egvx", CC, dE_gp_dE)

    _hom_sig_at_u_eq_curr, dSig_hom_dE = reaction_force_hom_sig_and_jacobian(
        law, E, w_sig, dw_sig_dE, du_local_dE, dS_gp_dE,
    )

    hom_eps_check = np.zeros(3, dtype=float)
    dEps_hom_dE = np.zeros((3, 3), dtype=float)
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

field_names = ["hom_eps", "hom_sig", "q_p", "n_iters", "converged",
               "dEps_hom_dE", "dSig_hom_dE", "hom_eps_check", "hom_sig_check"]
max_diffs = {}
for fi, fname in enumerate(field_names):
    diffs = []
    for o, nres in zip(old_results, new_results):
        a = np.asarray(o[fi]).astype(float)
        b = np.asarray(nres[fi]).astype(float)
        diffs.append(np.max(np.abs(a - b)) if a.size else 0.0)
    max_diffs[fname] = max(diffs)

print(flush=True)
for name, diff in max_diffs.items():
    print(f"[verify] max abs diff {name}: {diff:.3e}", flush=True)

ok = all(v < 1e-8 for v in max_diffs.values())
print("VERIFY_PASS" if ok else "VERIFY_FAIL", flush=True)
