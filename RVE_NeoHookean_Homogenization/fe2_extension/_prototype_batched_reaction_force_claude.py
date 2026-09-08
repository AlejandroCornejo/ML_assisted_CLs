#!/usr/bin/env python3
"""Proof of concept: batch reaction_force_hom_sig_and_jacobian across
Gauss points, the same way the decoder was batched -- capture the
per-point Kratos-state inputs it needs (which DO vary per point and get
overwritten by the next ComputeLocalArrays call, so must be copied out
during the loop) on real macro strains, then compare a batched-einsum
version against the current per-point-call version for both speed and
correctness.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dhprom_ann_direct_law_float64_claude import DHpromAnnDirectLawFloat64  # noqa: E402
from reaction_force_hom_tangent_claude import (  # noqa: E402
    reaction_force_hom_sig_and_jacobian, _sens_from_dF_dE,
)
from deformation_gradient_jacobian_claude import deformation_gradient_and_jacobian_2d  # noqa: E402
from _material_law_guard_claude import true_neo_hookean_active  # noqa: E402
from fom_solver_rve import SetDisplacementFromEquationVector, UpdateCurrentCoordinatesFromDisplacement  # noqa: E402

DHPROMANN_DIR = HERE / "maw_dynamic_reaction_force_dhpromann"

d = np.load(HERE / "cruciform_results_dhprom_f64_consistent_claude.npz")
e_gp = d["e_gp"]
print(f"[proto2] loaded {e_gp.shape[0]} real macro strains", flush=True)

law = DHpromAnnDirectLawFloat64(hprom_ann_dir=str(DHPROMANN_DIR))
_ = law.evaluate_with_tangent(e_gp[0])  # warm-up

N_SAMPLE = 300
rng = np.random.default_rng(0)
sample_idx = rng.choice(e_gp.shape[0], size=min(N_SAMPLE, e_gp.shape[0]), replace=False)
E_batch = e_gp[sample_idx]

# ---- Capture: run the FULL evaluate_with_tangent per point (as today),
# but ALSO snapshot the per-point Kratos-state arrays reaction_force_hom_
# sig_and_jacobian consumes, before the NEXT call overwrites them. ----
w_sig_list, dw_sig_dE_list, du_local_dE_list, dS_gp_dE_list = [], [], [], []
f_int_flat_list, F_micro_list, St_list = [], [], []
hom_sig_ref_list, dSig_ref_list = [], []

t0 = time.perf_counter()
for E in E_batch:
    hom_eps, hom_sig, dEps_hom_dE, dSig_hom_dE = law.evaluate_with_tangent(E)
    hom_sig_ref_list.append(hom_sig.copy())
    dSig_ref_list.append(dSig_hom_dE.copy())
t_current_full = time.perf_counter() - t0
print(f"[proto2] CURRENT full evaluate_with_tangent (decoder already batched separately in "
      f"deployed code -- here plain per-point for a clean baseline): {t_current_full:.3f}s, "
      f"{t_current_full/len(E_batch)*1000:.3f} ms/point", flush=True)

# Now re-run, this time capturing the exact inputs reaction_force_hom_sig_and_jacobian
# sees, by monkeypatching it to snapshot its own arguments before computing.
import reaction_force_hom_tangent_claude as rf_mod

_orig_fn = rf_mod.reaction_force_hom_sig_and_jacobian


def _capturing_fn(law_, E_, w_sig_, dw_sig_dE_, du_local_dE_, dS_gp_dE_, heps=1.0e-6):
    w_sig_list.append(np.array(w_sig_, dtype=float).copy())
    dw_sig_dE_list.append(np.array(dw_sig_dE_, dtype=float).copy())
    du_local_dE_list.append(np.array(du_local_dE_, dtype=float).copy())
    dS_gp_dE_list.append(np.array(dS_gp_dE_, dtype=float).copy())
    f_int_flat_list.append(law_.vec_assembler._f_int.reshape(law_.vec_assembler.n_elems, -1).copy())
    F_micro_list.append(law_.vec_assembler._F.copy())
    St_list.append(law_.vec_assembler._St.copy())
    return _orig_fn(law_, E_, w_sig_, dw_sig_dE_, du_local_dE_, dS_gp_dE_, heps=heps)


# dhprom_ann_direct_law_float64_claude imported reaction_force_hom_sig_and_jacobian
# by name at its own module top, so patch it there too.
import dhprom_ann_direct_law_float64_claude as law_mod
law_mod.reaction_force_hom_sig_and_jacobian = _capturing_fn

for E in E_batch:
    law.evaluate_with_tangent(E)

law_mod.reaction_force_hom_sig_and_jacobian = _orig_fn  # restore

print(f"[proto2] captured {len(w_sig_list)} real per-point snapshots", flush=True)

E_arr = np.asarray(E_batch, dtype=float)
w_sig_b = np.stack(w_sig_list, axis=0)
dw_sig_dE_b = np.stack(dw_sig_dE_list, axis=0)
du_local_dE_b = np.stack(du_local_dE_list, axis=0)
dS_gp_dE_b = np.stack(dS_gp_dE_list, axis=0)
f_int_flat_b = np.stack(f_int_flat_list, axis=0)
F_micro_b = np.stack(F_micro_list, axis=0)
St_b = np.stack(St_list, axis=0)

# ---- Time the CURRENT (per-point call) reaction-force function alone, on captured data ----
assembler = law.vec_assembler
t0 = time.perf_counter()
out_ref = []
for i in range(len(E_batch)):
    assembler._f_int = f_int_flat_b[i].reshape(assembler._f_int.shape)
    assembler._F = F_micro_b[i]
    assembler._St = St_b[i]
    out_ref.append(_orig_fn(law, E_arr[i], w_sig_b[i], dw_sig_dE_b[i], du_local_dE_b[i], dS_gp_dE_b[i]))
t_old = time.perf_counter() - t0
print(f"[proto2] OLD (per-point calls) on captured data: {t_old:.3f}s, {t_old/len(E_batch)*1000:.3f} ms/point",
      flush=True)


def reaction_force_hom_sig_and_jacobian_batch(
    law, E_b, w_sig_b, dw_sig_dE_b, du_local_dE_b, dS_gp_dE_b, f_int_flat_b, F_micro_b, St_b, heps=1.0e-6,
):
    """Batched version: same math as reaction_force_hom_sig_and_jacobian,
    with an extra leading batch axis (n_batch) threaded through every
    einsum via one extra index ('b'). f_int_flat_b/F_micro_b/St_b are the
    per-point Kratos-state snapshots (captured during the per-point loop,
    since ComputeLocalArrays itself cannot be batched)."""
    n_batch = E_b.shape[0]
    ne, ng, nn = law.vec_assembler.n_elems, law.vec_assembler.n_gauss, law.vec_assembler.n_nodes
    local_dirpos = law.dof_to_dirpos_local[law.vec_assembler.local_eq_ids]
    valid = local_dirpos >= 0

    # deformation_gradient_and_jacobian_2d is single-state; batch it with a Python loop
    # here (cheap: closed-form 2x2 algebra, not the bottleneck) rather than assuming
    # it supports batched input.
    sens_b = np.empty((n_batch, 3, law.dir_dofs.size), dtype=float)
    dsens_dE_b = np.empty((n_batch, 3, law.dir_dofs.size, 3), dtype=float)
    for b in range(n_batch):
        E = E_b[b]
        _, dF_dE = deformation_gradient_and_jacobian_2d(E)
        sens_b[b] = _sens_from_dF_dE(dF_dE, law.x_dir, law.y_dir, law.is_x_dir)
        for x in range(3):
            step = heps if abs(E[x]) < 1.0 else heps * max(1.0, abs(E[x]))
            Ep, Em = E.copy(), E.copy()
            Ep[x] += step
            Em[x] -= step
            _, dF_dE_p = deformation_gradient_and_jacobian_2d(Ep)
            _, dF_dE_m = deformation_gradient_and_jacobian_2d(Em)
            sens_p = _sens_from_dF_dE(dF_dE_p, law.x_dir, law.y_dir, law.is_x_dir)
            sens_m = _sens_from_dF_dE(dF_dE_m, law.x_dir, law.y_dir, law.is_x_dir)
            dsens_dE_b[b, :, :, x] = (sens_p - sens_m) / (2.0 * step)

    denom = law.thickness_scalar * float(law.hom_reference_measure)

    sk_by_k_b = np.zeros((n_batch, 3, ne, nn * 2), dtype=float)
    for k in range(3):
        sk = np.zeros((n_batch, ne, nn * 2), dtype=float)
        sk[:, valid] = sens_b[:, k, local_dirpos[valid]]
        sk_by_k_b[:, k] = sk
    c_e_b = np.einsum("bkep,bep->bek", sk_by_k_b, f_int_flat_b) / denom

    dc_e_dE_b = np.zeros((n_batch, ne, 3, 3), dtype=float)
    for k in range(3):
        sk_x = np.zeros((n_batch, ne, nn * 2, 3), dtype=float)
        sk_x[:, valid, :] = dsens_dE_b[:, k, local_dirpos[valid], :]
        dc_e_dE_b[:, :, k, :] += np.einsum("bepx,bep->bex", sk_x, f_int_flat_b) / denom

    DN = law.vec_assembler.DN
    du_local_dE_4d_b = du_local_dE_b.reshape(n_batch, ne, nn, 2, 3)
    dgrad_u_dE_b = np.einsum("beaix,egaj->begijx", du_local_dE_4d_b, DN)

    dSt_dE_b = np.zeros((n_batch,) + law.vec_assembler._St.shape + (3,), dtype=float)
    dS_gp_dE_b = np.asarray(dS_gp_dE_b, dtype=float)
    dSt_dE_b[..., 0, 0, :] = dS_gp_dE_b[..., 0, :]
    dSt_dE_b[..., 1, 1, :] = dS_gp_dE_b[..., 1, :]
    dSt_dE_b[..., 0, 1, :] = dS_gp_dE_b[..., 2, :]
    dSt_dE_b[..., 1, 0, :] = dS_gp_dE_b[..., 2, :]

    dP_dE_b = (
        np.einsum("begijx,begjk->begikx", dgrad_u_dE_b, St_b)
        + np.einsum("begij,begjkx->begikx", F_micro_b, dSt_dE_b)
    )
    df_int_dE_b = np.einsum("egak,begikx,eg->beaix", DN, dP_dE_b, law.vec_assembler.w_detJ)
    df_int_flat_dE_b = df_int_dE_b.reshape(n_batch, ne, -1, 3)

    for k in range(3):
        dc_e_dE_b[:, :, k, :] += np.einsum("bep,bepx->bex", sk_by_k_b[:, k], df_int_flat_dE_b) / denom

    # hom_sig = c_e.T @ w_sig  (per point);  batched: sum_e c_e_b[b,e,k]*w_sig_b[b,e]
    hom_sig_b = np.einsum("bek,be->bk", c_e_b, w_sig_b)
    dSig_hom_dE_b = (
        np.einsum("bekx,be->bkx", dc_e_dE_b, w_sig_b)
        + np.einsum("bek,bex->bkx", c_e_b, dw_sig_dE_b)
    )
    return hom_sig_b, dSig_hom_dE_b


t0 = time.perf_counter()
hom_sig_new, dSig_new = reaction_force_hom_sig_and_jacobian_batch(
    law, E_arr, w_sig_b, dw_sig_dE_b, du_local_dE_b, dS_gp_dE_b, f_int_flat_b, F_micro_b, St_b,
)
t_new = time.perf_counter() - t0
print(f"[proto2] BATCHED (one call): {t_new:.3f}s, {t_new/len(E_batch)*1000:.3f} ms/point", flush=True)
print(f"[proto2] speedup: {t_old/max(t_new,1e-9):.2f}x", flush=True)

hom_sig_old_arr = np.array([o[0] for o in out_ref])
dSig_old_arr = np.array([o[1] for o in out_ref])
sig_err = np.max(np.abs(hom_sig_old_arr - hom_sig_new))
dsig_err = np.max(np.abs(dSig_old_arr - dSig_new))
sig_rel = sig_err / max(np.max(np.abs(hom_sig_old_arr)), 1e-300)
dsig_rel = dsig_err / max(np.max(np.abs(dSig_old_arr)), 1e-300)
print(f"[proto2] max abs diff hom_sig: {sig_err:.3e} (rel {sig_rel:.3e})", flush=True)
print(f"[proto2] max abs diff dSig_hom_dE: {dsig_err:.3e} (rel {dsig_rel:.3e})", flush=True)
print("PROTOTYPE2_DONE_MARKER", flush=True)
