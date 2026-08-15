#!/usr/bin/env python3
"""Analytic derivative of core/fom_solver_rve.py's
DeformationGradientFromGreenLagrange2D: F = sqrt(C), C = 2E+I, via the
symmetric matrix square root (F = Q sqrt(Lambda) Q^T from C's own
eigendecomposition C = Q Lambda Q^T).

Needed for a consistent tangent: the affine lifting u_aff = (F(E)-I)(X-Xc)
is a real part of the total displacement field, so a genuinely consistent
d(stress)/d(E_macro) needs dF/dE, not just dq_p/dE and the decoder's own
Jacobian.

Uses the standard "divided difference" formula for the derivative of a
matrix function applied to a symmetric matrix (Daleckii-Krein): in C's
own eigenbasis, d(F)_ij = d(C)_ij / (sqrt(lambda_i) + sqrt(lambda_j)),
which reduces to the ordinary scalar derivative 1/(2 sqrt(lambda_i)) on
the diagonal (i=j), including in the degenerate lambda_i=lambda_j limit.
"""
from __future__ import annotations

import numpy as np


def deformation_gradient_and_jacobian_2d(strain_voigt):
    """strain_voigt = (E11, E22, gamma12=2*E12). Returns (F (2,2),
    dF_dE: (2,2,3) with dF_dE[:,:,k] = dF/dE_voigt[k])."""
    E11, E22, g12 = (float(v) for v in strain_voigt)
    C = np.array([[1.0 + 2.0 * E11, g12], [g12, 1.0 + 2.0 * E22]])
    eigvals, Q = np.linalg.eigh(C)
    if np.min(eigvals) <= 0.0:
        raise RuntimeError("Invalid Green-Lagrange strain state: C=2E+I is not positive definite.")
    sqrt_lam = np.sqrt(eigvals)
    F = Q @ np.diag(sqrt_lam) @ Q.T

    denom = sqrt_lam[:, None] + sqrt_lam[None, :]  # (2,2), denom[i,j] = sqrt(l_i)+sqrt(l_j)

    dC_dE = np.zeros((2, 2, 3))
    dC_dE[0, 0, 0] = 2.0  # dC11/dE11
    dC_dE[1, 1, 1] = 2.0  # dC22/dE22
    dC_dE[0, 1, 2] = 1.0  # dC12/dgamma12
    dC_dE[1, 0, 2] = 1.0  # dC21/dgamma12

    dF_dE = np.zeros((2, 2, 3))
    for k in range(3):
        dC_tilde = Q.T @ dC_dE[:, :, k] @ Q  # in C's eigenbasis
        dF_tilde = dC_tilde / denom
        dF_dE[:, :, k] = Q @ dF_tilde @ Q.T
    return F, dF_dE


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/kratos/ML_assisted_CLs_clean/RVE_NeoHookean_Homogenization/core")
    from fom_solver_rve import DeformationGradientFromGreenLagrange2D

    rng = np.random.default_rng(0)
    test_states = [
        np.array([0.05, 0.0, 0.0]),
        np.array([0.3, -0.1, 0.05]),
        np.array([0.8, 0.4, -0.05]),
        np.array([1.5, -0.3, 0.06]),
    ]
    h = 1.0e-6
    all_ok = True
    for E in test_states:
        F, dF_dE = deformation_gradient_and_jacobian_2d(E)
        F_check = DeformationGradientFromGreenLagrange2D(E)
        val_err = np.linalg.norm(F - F_check)

        dF_dE_fd = np.zeros((2, 2, 3))
        for k in range(3):
            Ep, Em = E.copy(), E.copy()
            Ep[k] += h
            Em[k] -= h
            Fp = DeformationGradientFromGreenLagrange2D(Ep)
            Fm = DeformationGradientFromGreenLagrange2D(Em)
            dF_dE_fd[:, :, k] = (Fp - Fm) / (2 * h)
        rel_err = np.linalg.norm(dF_dE - dF_dE_fd) / max(np.linalg.norm(dF_dE_fd), 1e-30)
        ok = val_err < 1e-12 and rel_err < 1e-6
        all_ok = all_ok and ok
        print(f"E={E}: F value_err={val_err:.3e}, dF/dE rel_err={rel_err:.3e} [{'ok' if ok else 'FAIL'}]")
    print("PASS" if all_ok else "FAIL")
