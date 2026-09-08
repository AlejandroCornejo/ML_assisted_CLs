#!/usr/bin/env python3
"""The weight-field regression as Hernandez actually specifies it, applied to
the 10-point residual rule.

WHY THIS REPLACES MY OWN FORMULATION. I fitted the RBF coefficients directly
against the integration conditions, on the principle -- correct twice before in
this project -- that one should optimize the objective and not an intermediate.
That principle does not transfer here, and applying it by inertia was the
mistake. With a softmax network, positivity is structural, so the intermediate
only gets in the way. With an UNCONSTRAINED RBF, the oracle weight values are
precisely what holds the solution inside the feasible region: fitting the
conditions alone leaves the 6-dimensional null space unconstrained, and the
solver walks off into it. Measured: weights reaching -2.5e+05 against a total of
1546, negative at 100% of states, and clip+renormalize then destroying the
result (7.28e+00 against 2.63e-04 raw).

The paper is explicit that the regression is unconstrained and that this is a
known limitation, not an oversight:

    "Generic regression procedures ... do not guarantee positivity or exact
     preservation of the total volume, particularly outside the convex hull of
     the training data. ... The study of such constraint-preserving regression
     procedures lies beyond the scope of the present work; in the numerical
     examples discussed in the sequel, we simply employ the same unconstrained
     regression techniques used for the nonlinear decoder."

while positivity IS required of the pruned per-state weights (condition
`item:positive`), together with exact volume, and smoothness of each weight
field over the manifold (`item:smooth`) -- that last one existing precisely
because a regression has to represent them.

THE SPECIFICATION, followed rather than reinvented: anisotropic Gaussian RBFs
with per-direction characteristic lengths, augmented with QUADRATIC polynomial
terms, coefficients from standard-form Tikhonov-regularized least squares with
identity regularization matrix, only a subset of snapshots retained as centers,
lambda_RBF = 1e-10. My own grid search had independently landed on a Gaussian
kernel, a center subset and lambda = 1e-10, so the differences that remain are
the anisotropy, the quadratic augmentation, and the fitting target.

Anisotropy is taken from the data here (l_d proportional to the spread of each
latent coordinate) rather than hand-tuned as in the paper, with the global scale
and lambda swept.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "12")

import sys

import numpy as np

import maw_lab as L

RULE = "res"
PTS = 10
CENTER_STRIDE = 3            # structured subsampling, as in the paper
ALPHAS = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6)
LAMS = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4)
REFERENCE_88 = 1.5805e-03
TARGET_ANN = 1.9725e-02


def quad_terms(Q):
    """1, q_i, q_i q_j -- the quadratic polynomial augmentation."""
    n, dim = Q.shape
    cols = [np.ones((n, 1)), Q]
    for i in range(dim):
        for j in range(i, dim):
            cols.append((Q[:, i] * Q[:, j])[:, None])
    return np.hstack(cols)


def aniso_gauss(Q, centers, ell):
    d = (Q[:, None, :] - centers[None, :, :]) / ell[None, None, :]
    return np.exp(-np.sum(d ** 2, axis=2))


def design(Q, centers, ell):
    return np.hstack([quad_terms(Q), aniso_gauss(Q, centers, ell)])


def main():
    d = L.load()
    q, ne = d["q"], d["ne"]
    C = d["C_res"] if RULE == "res" else d["C_sig"]
    b = L.targets(C)
    fi, vi = L.split(q.shape[0])
    h = vi.size // 2
    si, ri = vi[:h], vi[h:]
    ph = np.load(L.HERE / f"maw_phase2_{RULE}.npz")
    Z = ph[f"{RULE}_{PTS}_z"]
    Wor = ph[f"{RULE}_{PTS}_W_oracle"]        # (k, n_states), non-negative
    A = L.blocks(C, Z)
    k = Z.size

    cen = q[fi[::CENTER_STRIDE]]
    spread = q[fi].std(axis=0)
    print(f"{RULE}, {k} points, {cen.shape[0]} centers of {fi.size} fit states")
    print(f"  oracle weights: min {Wor.min():.3e}, "
          f"sums {Wor.sum(0).min():.4f}..{Wor.sum(0).max():.4f}")
    print(f"  network 100k epochs (softmax)  {TARGET_ANN:.4e}")
    print(f"  classic ECM, 88 points         {REFERENCE_88:.4e}\n", flush=True)

    print(f"{'alpha':>7} {'lambda':>9} {'w rel err':>11} {'constraint':>11} "
          f"{'min w':>11} {'neg%':>6}")
    print("-" * 62)
    best = None
    for a in ALPHAS:
        ell = a * spread
        Pf = design(q[fi], cen, ell)
        G = Pf.T @ Pf
        g = Pf.T @ Wor[:, fi].T
        dg = np.mean(np.diag(G))
        Ps = design(q[si], cen, ell)
        for lam in LAMS:
            try:
                Cc = np.linalg.solve(G + lam * dg * np.eye(G.shape[0]), g)
            except np.linalg.LinAlgError:
                continue
            Ws = (Ps @ Cc).T                       # (k, n_sel)
            ew = (np.linalg.norm(Ws - Wor[:, si])
                  / np.linalg.norm(Wor[:, si]))
            ec = np.median(L.const_err(A[si], b[si], Ws))
            neg = 100.0 * np.mean(Ws.min(axis=0) < 0.0)
            print(f"{a:>7.2f} {lam:>9.0e} {ew:>11.4e} {ec:>11.4e} "
                  f"{Ws.min():>+11.3e} {neg:>5.1f}%", flush=True)
            if np.isfinite(ec) and (best is None or ec < best["ec"]):
                best = dict(ec=ec, a=a, lam=lam, C=Cc, ell=ell, ew=ew)

    if best is None:
        print("no finite configuration")
        return 1

    print(f"\n=== best on selection: alpha={best['a']}, "
          f"lambda={best['lam']:.0e} ===")
    Pr = design(q[ri], cen, best["ell"])
    Wr = (Pr @ best["C"]).T
    ec = np.median(L.const_err(A[ri], b[ri], Wr))
    ew = np.linalg.norm(Wr - Wor[:, ri]) / np.linalg.norm(Wor[:, ri])
    s = Wr.sum(axis=0)
    print(f"  weight field rel err   {ew:.4e}")
    print(f"  constraint (REPORT)    {ec:.4e}")
    print(f"  min weight             {Wr.min():+.4e}   "
          f"states with a negative weight {100 * np.mean(Wr.min(axis=0) < 0):.1f}%")
    print(f"  volume sum             {s.min():.4f}..{s.max():.4f} "
          f"(target {ne}, NOT enforced)")

    Wc = np.maximum(Wr, 0.0)
    Wc = Wc * (float(ne) / np.maximum(Wc.sum(axis=0, keepdims=True), 1e-300))
    print(f"  after clip+renorm      "
          f"{np.median(L.const_err(A[ri], b[ri], Wc)):.4e}   "
          f"min w {Wc.min():+.3e}   sum {Wc.sum(0).min():.4f}..{Wc.sum(0).max():.4f}")

    print(f"\n  vs network (100k epochs) {TARGET_ANN / ec:.2f}x")
    print(f"  vs classic at 88 points  {REFERENCE_88 / ec:.2f}x")
    np.savez_compressed(L.HERE / f"rbf_joaquin_{RULE}_{k}.npz",
                        C=best["C"], centers=cen, ell=best["ell"],
                        lam=best["lam"], z=Z)
    print("\nRBF_JOAQUIN_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
