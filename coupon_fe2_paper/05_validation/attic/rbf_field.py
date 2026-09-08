#!/usr/bin/env python3
"""An RBF weight field whose coefficients are solved DIRECTLY against the
constraints -- a linear least-squares problem, not an optimization.

WHY RBF AND NOT GPR. A Gaussian process would tune kernel, length scales and
noise by marginal likelihood, which is a genuinely better hyperparameter search
than any grid. But the likelihood it maximizes is that of reproducing the ORACLE
WEIGHTS, and this project has already measured that this is the wrong target:
the trained fields sit 23%-195% away from the oracle weights in the residual
rule and still satisfy the constraints, because with 10 weights against 4 rows
the system is underdetermined and many weight vectors are equally valid. GPR
would optimize impeccably against the wrong criterion.

WHY THE LINEARITY IS THE REAL WIN, bigger than the choice of kernel. With

    w(q) = w0 + N @ sum_m phi_m(q) c_m

the map from coefficients c to A(q) w(q) is LINEAR, so

    min_c  sum_j || A_j w(q_j) - b_j ||^2 / || b_j ||^2

is a linear least-squares problem with an exact solution. No Adam, no learning
rate, no plateaus -- and it minimizes the reported objective itself. The network
route took 100000 epochs, needed its scheduler debugged, and produced four
premature convergence calls from me.

EXACT VOLUME CONSERVATION SURVIVES, BY CONSTRUCTION. N holds an orthonormal
basis of {v : sum(v) = 0}, so every correction has zero sum and

    sum_i w_i(q) = sum_i w0_i = n_elements

for ANY q and ANY coefficients. The volume row of A is all ones, so
(A_j N)[vol, :] = 1^T N = 0 and that row contributes nothing to the system --
it is satisfied identically rather than fitted, which the self-check verifies.

NON-NEGATIVITY DOES NOT SURVIVE, and that is a real loss against the softmax
parametrization, which gave w >= 0 for any q however wrong the coefficients.
Here it can only be clipped at evaluation, and clipping is non-differentiable --
which matters for the residual rule specifically, since it enters the Newton
loop and a consistent tangent needs dw/dq. Both the unclipped minimum weight and
the error with and without clip+renormalize are reported so the size of the loss
is visible rather than assumed.

The Gram matrix is formed once per (kernel, epsilon, centers) and reused across
the regularization sweep, which is what makes a real grid search affordable.
Normal equations square the condition number -- a trap this project has already
paid for once in the displacement POD -- so the winning configuration is
re-solved with a direct least-squares factorization and the two are compared.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "12")

import numpy as np
from scipy.spatial.distance import cdist  # noqa: F401  (re-exported)

KERNELS = ("gaussian", "matern32", "matern52", "multiquadric",
           "inv_multiquadric")


def kernel_matrix(Q, centers, name, eps):
    r = cdist(np.atleast_2d(Q), centers) * float(eps)
    if name == "gaussian":
        return np.exp(-(r ** 2))
    if name == "matern32":
        a = np.sqrt(3.0) * r
        return (1.0 + a) * np.exp(-a)
    if name == "matern52":
        a = np.sqrt(5.0) * r
        return (1.0 + a + a ** 2 / 3.0) * np.exp(-a)
    if name == "multiquadric":
        return np.sqrt(1.0 + r ** 2)
    if name == "inv_multiquadric":
        return 1.0 / np.sqrt(1.0 + r ** 2)
    raise ValueError(f"unknown kernel {name}")


def basis(Q, centers, name, eps):
    """Kernel columns plus a constant term."""
    K = kernel_matrix(Q, centers, name, eps)
    return np.hstack([np.ones((K.shape[0], 1)), K])


def zero_sum_basis(k):
    """Orthonormal basis of {v in R^k : sum(v) = 0}."""
    M = np.eye(k) - np.ones((k, k)) / k
    U, s, _ = np.linalg.svd(M)
    return np.ascontiguousarray(U[:, s > 1e-12])


def _blocks(A, b, Phi, N, w0, chunk=400):
    """Accumulate D^T D and D^T y without ever forming D."""
    n, m, kk = A.shape
    p = Phi.shape[1] * N.shape[1]
    G = np.zeros((p, p))
    g = np.zeros(p)
    nrm = np.maximum(np.linalg.norm(b, axis=1), 1e-300)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        Ac, bc, Pc, nc = A[s:e], b[s:e], Phi[s:e], nrm[s:e]
        AN = np.matmul(Ac, N)                              # (c, m, k-1)
        # D[j] = kron(Phi[j], AN[j]) / nrm[j]
        D = (Pc[:, None, :, None] * AN[:, :, None, :]).reshape(e - s, m, p)
        D /= nc[:, None, None]
        y = (bc - np.einsum("jmk,k->jm", Ac, w0)) / nc[:, None]
        Df = D.reshape(-1, p)
        G += Df.T @ Df
        g += Df.T @ y.reshape(-1)
    return G, g


def fit_rbf(q_fit, A_fit, b_fit, centers, kernel, eps, lams, N, w0):
    Phi = basis(q_fit, centers, kernel, eps)
    G, g = _blocks(A_fit, b_fit, Phi, N, w0)
    d = np.mean(np.diag(G)) + 1e-300
    out = {}
    for lam in lams:
        try:
            c = np.linalg.solve(G + lam * d * np.eye(G.shape[0]), g)
        except np.linalg.LinAlgError:
            continue
        out[lam] = c.reshape(Phi.shape[1], N.shape[1])
    return out


def eval_rbf(q, C, centers, kernel, eps, N, w0, clip=False):
    Phi = basis(q, centers, kernel, eps)
    W = w0[None, :] + (Phi @ C) @ N.T                      # (n, k)
    if clip:
        W = np.maximum(W, 0.0)
        tot = W.sum(axis=1, keepdims=True)
        W = W * (w0.sum() / np.maximum(tot, 1e-300))
    return np.ascontiguousarray(W.T)                       # (k, n)
