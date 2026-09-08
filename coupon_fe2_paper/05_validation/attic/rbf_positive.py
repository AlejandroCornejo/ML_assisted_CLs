#!/usr/bin/env python3
"""Is NON-NEGATIVITY the binding constraint on the residual rule, or the
function class? Solve the RBF field with w >= 0 imposed, not clipped.

WHAT THE UNCONSTRAINED SOLVE SHOWED. On the 10-point residual support, an RBF
field whose coefficients are solved directly against the constraints reaches
2.71e-04 on held-out states -- 5.8x BETTER than the classic 88-point
fixed-weight reference, with 10 elements, and 73x better than the network's
1.97e-02 after 100000 epochs. But its weights run to -3953 against a total of
1546, and 100% of states carry a negative weight. Clipping and renormalizing
destroys it completely: 7.28e+00, a factor 1800 worse.

WHY THAT IS COHERENT rather than a curiosity. The projected residual is a 98%
cancellation of its element contributions (|sum| / sum|.| = 2.05e-02 measured,
against 9.51e-01 for the homogenized stress). Reproducing a near-cancellation
from a handful of points needs OPPOSING SIGNS. The softmax parametrization
forbids them by construction, which is exactly why every network fit stalled
around 1e-02; the RBF is free to use them, and does, immediately.

THE QUESTION THIS SETTLES. If the best achievable field with w >= 0 imposed as
a CONSTRAINT lands near 1e-02, then the function class was never the limit and
the network was not underperforming -- non-negativity is the binding constraint,
and no regressor of any kind can do better while keeping the guarantee. If
instead a constrained solve reaches, say, 1e-03, then the network really was
leaving accuracy on the table.

METHOD. Non-negativity at every fit state is a linear inequality in the
coefficients, w(q_j) = w0 + N c^T phi_j >= 0, which makes this a
least-squares problem with 4208 x 10 inequality constraints -- too large for a
dense QP. Solved instead by an active-set style iteration: solve, find the
violated (state, component) pairs, add heavily weighted rows driving those to
zero, re-solve, repeat. That converges to a feasible-on-the-data solution
without ever forming the full constraint matrix, and any residual violation is
reported rather than assumed away.

Uses a direct least-squares factorization throughout: on this problem the
normal equations cost a factor of 15 in accuracy (3.97e-03 against 2.71e-04).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "12")

import sys
import time

import numpy as np

import maw_lab as L
import rbf_field as R

RULE = "res"
PTS = 10
N_CENTERS = 600
KERNEL = "gaussian"
EPS_SCALE = 0.25
ITERS = 12
PEN0 = 1.0e2
REFERENCE_88 = 1.5805e-03
TARGET_ANN = 1.9725e-02


def design(q, A, b, centers, kernel, eps, N, w0):
    Phi = R.basis(q, centers, kernel, eps)
    nrm = np.maximum(np.linalg.norm(b, axis=1), 1e-300)
    AN = np.matmul(A, N)
    D = (Phi[:, None, :, None] * AN[:, :, None, :]).reshape(
        q.shape[0], A.shape[1], -1) / nrm[:, None, None]
    y = (b - np.einsum("jmk,k->jm", A, w0)) / nrm[:, None]
    return Phi, D.reshape(-1, D.shape[2]), y.reshape(-1)


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
    A = L.blocks(C, Z)
    k = Z.size

    Nz = R.zero_sum_basis(k)
    w0 = np.full(k, float(ne) / k)
    rng = np.random.default_rng(3)
    cen = q[fi[rng.permutation(fi.size)[:N_CENTERS]]]
    r0 = np.median(np.sort(R.cdist(q[fi], cen), axis=1)[:, 0])
    eps = EPS_SCALE / max(r0, 1e-300)

    Phi_f, Df, yf = design(q[fi], A[fi], b[fi], cen, KERNEL, eps, Nz, w0)
    p = Df.shape[1]
    print(f"{RULE}, {k} points, {N_CENTERS} centers, {p} coefficients, "
          f"{fi.size} fit states")
    print(f"  network 100k epochs (softmax, w>=0 by construction) "
          f"{TARGET_ANN:.4e}")
    print(f"  classic ECM, 88 points, w>=0                        "
          f"{REFERENCE_88:.4e}\n", flush=True)

    def report(Cc, tag, extra=""):
        Wr = R.eval_rbf(q[ri], Cc, cen, KERNEL, eps, Nz, w0)
        Ws = R.eval_rbf(q[si], Cc, cen, KERNEL, eps, Nz, w0)
        er = np.median(L.const_err(A[ri], b[ri], Wr))
        es = np.median(L.const_err(A[si], b[si], Ws))
        neg = 100.0 * np.mean(Wr.min(axis=0) < 0.0)
        s = Wr.sum(axis=0)
        print(f"  {tag:<26} sel {es:.4e}  REPORT {er:.4e}  "
              f"min w {Wr.min():+.3e}  neg states {neg:5.1f}%  "
              f"sum {s.min():.4f}..{s.max():.4f} {extra}", flush=True)
        return er

    # --- unconstrained, for reference -------------------------------------
    t0 = time.perf_counter()
    C0 = np.linalg.lstsq(Df, yf, rcond=None)[0].reshape(Phi_f.shape[1], k - 1)
    e_unc = report(C0, "unconstrained", f"[{time.perf_counter() - t0:.0f}s]")

    # --- active-set iteration on w >= 0 at the fit states -----------------
    Cc = C0
    pen = PEN0
    for it in range(1, ITERS + 1):
        Wf = R.eval_rbf(q[fi], Cc, cen, KERNEL, eps, Nz, w0)      # (k, nfit)
        bad = np.argwhere(Wf.T < 0.0)                              # (state, comp)
        if bad.size == 0:
            print(f"  iteration {it}: feasible on all fit states")
            break
        # rows forcing w0 + N c^T phi = 0 at the violated pairs
        rows = (Phi_f[bad[:, 0]][:, :, None]
                * Nz[bad[:, 1]][:, None, :]).reshape(bad.shape[0], -1)
        rhs = -w0[bad[:, 1]]
        sc = pen * np.linalg.norm(Df) / max(np.linalg.norm(rows), 1e-300)
        Cc = np.linalg.lstsq(np.vstack([Df, sc * rows]),
                             np.concatenate([yf, sc * rhs]),
                             rcond=None)[0].reshape(Phi_f.shape[1], k - 1)
        frac = 100.0 * bad.shape[0] / (fi.size * k)
        e = report(Cc, f"iter {it} (pen {pen:.0e})",
                   f"viol {frac:.2f}% of entries")
        pen *= 3.0

    print(f"\n  unconstrained            {e_unc:.4e}")
    print(f"  best with w >= 0 pushed  {e:.4e}")
    print(f"  softmax network          {TARGET_ANN:.4e}")
    print(f"  classic ECM 88 pts       {REFERENCE_88:.4e}")
    np.savez_compressed(L.HERE / f"rbf_pos_{RULE}_{k}.npz",
                        C_unconstrained=C0, C_positive=Cc, centers=cen,
                        N=Nz, w0=w0, eps=eps, kernel=np.array(KERNEL), z=Z)
    print("\nRBF_POSITIVE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
