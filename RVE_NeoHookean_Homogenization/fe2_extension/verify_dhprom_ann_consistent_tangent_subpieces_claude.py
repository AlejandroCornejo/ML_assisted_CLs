#!/usr/bin/env python3
"""Decisive, isolated checks of the two pieces of evaluate_with_tangent's
chain rule that are NOT already independently verified elsewhere:

(A) Decoder Jacobian at q_p=0 must reduce to the already-trusted J0_const
    (same function, same point, computed by the same torch.autograd
    pattern __init__ already uses) -- a near-exact-equality sanity check.
(B) Decoder Jacobian at a NONZERO q_p, checked against a hand finite
    difference of self.ann_model directly (bypassing E, the affine lift,
    the RVE mesh, and the ECM weights entirely) -- isolates the
    torch.autograd.functional.jacobian usage from every other moving part.
(C) The _dhom_dE aggregation formula itself, checked against finite
    differences of a SYNTHETIC, float64, hand-differentiable stand-in for
    (mean_e(E), w(E)) -- isolates the aggregation algebra (nz-active-set
    handling, the two-term product rule, the area_e/den normalization)
    from the ANN/decoder/geometry pieces entirely, so there is no float32
    noise anywhere in this one.

deformation_gradient_and_jacobian_2d and the ANN homogenization-weight
Jacobian are NOT re-tested here -- both already have their own __main__
self-tests against finite differences on the real trained models/functions
(deformation_gradient_jacobian_claude.py, maw_hom_weight_jacobian_claude.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for sub in ("core", "prom/ann", "hprom/ann"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from dhprom_ann_direct_law_claude import DHpromAnnDirectLaw


def check_A_B(law):
    print("[A] decoder Jacobian at q_p=0 vs. already-trusted J0_const:")
    q_zero = torch.zeros((1, law.n_primary), device=law.device)
    with torch.enable_grad():
        q_in = q_zero.reshape(-1).clone().detach().requires_grad_(True)

        def ann_from_qvec(qvec):
            return law.ann_model(qvec.view(1, -1)).reshape(-1)

        J0_recomputed = torch.autograd.functional.jacobian(ann_from_qvec, q_in).reshape(
            law.n_secondary, law.n_primary
        ).detach()
    diff = torch.linalg.norm(J0_recomputed - law.J0_const_torch).item()
    ref = torch.linalg.norm(law.J0_const_torch).item()
    rel = diff / max(ref, 1e-30)
    print(f"    ||J0_recomputed - J0_const||/||J0_const|| = {rel:.3e} [{'ok' if rel < 1e-6 else 'FAIL'}]")

    print("\n[B] decoder Jacobian at nonzero q_p vs. hand finite difference of ann_model directly:")
    rng = np.random.default_rng(7)
    all_ok = True
    for trial in range(3):
        q_p0 = (rng.standard_normal(law.n_primary) * 0.05).astype(np.float32)
        q_p0_t = torch.from_numpy(q_p0).reshape(1, -1).to(law.device)

        with torch.enable_grad():
            q_in = q_p0_t.reshape(-1).clone().detach().requires_grad_(True)

            def ann_from_qvec(qvec):
                return law.ann_model(qvec.view(1, -1)).reshape(-1)

            J_analytic = torch.autograd.functional.jacobian(ann_from_qvec, q_in).reshape(
                law.n_secondary, law.n_primary
            ).detach().cpu().numpy()

        h = 1.0e-3
        J_fd = np.zeros((law.n_secondary, law.n_primary), dtype=float)
        with torch.no_grad():
            for k in range(law.n_primary):
                qp = q_p0.copy(); qp[k] += h
                qm = q_p0.copy(); qm[k] -= h
                yp = law.ann_model(torch.from_numpy(qp).reshape(1, -1).to(law.device)).reshape(-1).cpu().numpy()
                ym = law.ann_model(torch.from_numpy(qm).reshape(1, -1).to(law.device)).reshape(-1).cpu().numpy()
                J_fd[:, k] = (yp - ym) / (2 * h)

        rel = np.linalg.norm(J_analytic - J_fd) / max(np.linalg.norm(J_fd), 1e-30)
        ok = rel < 1e-2  # float32 decoder + h=1e-3 FD -> loose but decisive tolerance
        all_ok = all_ok and ok
        print(f"    trial {trial}: ||J_analytic-J_fd||/||J_fd||={rel:.3e} [{'ok' if ok else 'FAIL'}]")
    print(f"    -> {'PASS' if all_ok else 'FAIL'}")


def check_C():
    print("\n[C] _dhom_dE aggregation formula vs. FD of a synthetic float64 (mean_e(E), w(E)):")
    rng = np.random.default_rng(3)
    n_elem = 12
    area_e = np.abs(rng.standard_normal(n_elem)) + 0.5
    den = 3.7

    # Synthetic smooth functions of E (quadratic in E so the derivative is
    # exact and simple, and there is no float32/ANN/geometry anywhere).
    A_mean = rng.standard_normal((n_elem, 3, 3)) * 0.3   # mean_e(E)_j = A_mean[e,j,:] . E + b_mean[e,j]
    b_mean = rng.standard_normal((n_elem, 3))
    Q_mean = rng.standard_normal((n_elem, 3, 3, 3)) * 0.1  # quadratic term, mean_e(E)_j += E.Q[e,j].E

    A_w = rng.standard_normal((n_elem, 3)) * 0.2         # w(E)_e = A_w[e,:].E + b_w[e]  (kept positive via offset)
    b_w = np.abs(rng.standard_normal(n_elem)) + 1.0

    def mean_e_fn(E):
        lin = np.einsum("ejk,k->ej", A_mean, E) + b_mean
        quad = np.einsum("k,ejkl,l->ej", E, Q_mean, E)
        return lin + quad

    def dmean_e_fn(E):
        # d/dE_x of (A_mean[e,j,:].E + E.Q[e,j].E) = A_mean[e,j,x] + sum_l (Q[e,j,x,l]+Q[e,j,l,x])*E_l
        lin_part = A_mean  # (n_elem,3,3) already [e,j,x]
        quad_part = np.einsum("ejxl,l->ejx", Q_mean + np.swapaxes(Q_mean, 2, 3), E)
        return lin_part + quad_part

    def w_fn(E):
        return np.einsum("ek,k->e", A_w, E) + b_w

    def dw_fn(E):
        return A_w  # (n_elem,3), constant since w is linear in E

    sys.path.insert(0, str(HERE))
    from dhprom_ann_direct_law_claude import DHpromAnnDirectLaw as _DHL  # for _dhom_dE staticmethod

    E0 = np.array([0.2, -0.35, 0.15])
    mean_e0 = mean_e_fn(E0)
    dmean_e0 = dmean_e_fn(E0)
    w0 = w_fn(E0)
    dw0 = dw_fn(E0)

    dhom_dE_analytic = _DHL._dhom_dE(mean_e0, dmean_e0, w0, dw0, area_e, den)

    h = 1.0e-6
    dhom_dE_fd = np.zeros((3, 3), dtype=float)
    for k in range(3):
        Ep, Em = E0.copy(), E0.copy()
        Ep[k] += h
        Em[k] -= h

        def hom_from_E(E):
            me = mean_e_fn(E)
            ww = w_fn(E)
            nz = np.flatnonzero(np.abs(ww) > 1.0e-14)
            out = np.zeros(3)
            for j in range(3):
                out[j] = np.dot(ww[nz] * area_e[nz], me[nz, j]) / den
            return out

        dhom_dE_fd[:, k] = (hom_from_E(Ep) - hom_from_E(Em)) / (2 * h)

    rel = np.linalg.norm(dhom_dE_analytic - dhom_dE_fd) / max(np.linalg.norm(dhom_dE_fd), 1e-30)
    print(f"    dhom_dE_analytic=\n{dhom_dE_analytic}")
    print(f"    dhom_dE_fd=\n{dhom_dE_fd}")
    print(f"    rel_err={rel:.3e} [{'ok' if rel < 1e-6 else 'FAIL'}]")


def main():
    print("[verify-subpieces] building DHpromAnnDirectLaw ...")
    law = DHpromAnnDirectLaw()
    check_A_B(law)
    check_C()


if __name__ == "__main__":
    main()
