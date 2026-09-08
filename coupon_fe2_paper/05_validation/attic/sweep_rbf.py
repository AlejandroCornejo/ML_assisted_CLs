#!/usr/bin/env python3
"""Grid search the RBF weight field on the 10-point residual rule.

Scored on the SELECTION half of the held-out states by the median relative
constraint error -- the objective itself, on states not used to fit, and
disjoint from the half the final number is quoted from.

The target to beat is 1.9725e-02, what the network reached on this same support
after 100000 epochs with a working LR schedule. If a linear solve gets under
that, the network's limit was its optimizer.
"""
from __future__ import annotations

import sys
import time

import numpy as np

import maw_lab as L
import rbf_field as R

RULE = "res"
PTS = 10
N_CENTERS = (150, 300, 600)
EPS_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)
LAMS = (1e-10, 1e-8, 1e-6, 1e-4, 1e-2)
TARGET_ANN = 1.9725e-02       # network, 100k epochs, annealed
REFERENCE_88 = 1.5805e-03     # classic ECM, 88 points, fixed weights


def main():
    d = L.load()
    q, ne = d["q"], d["ne"]
    C = d["C_res"] if RULE == "res" else d["C_sig"]
    b = L.targets(C)
    fi, vi = L.split(q.shape[0])
    h = vi.size // 2
    si, ri = vi[:h], vi[h:]           # selection / reporting, disjoint
    ph = np.load(L.HERE / (f"maw_phase2_{RULE}.npz"))
    Z = ph[f"{RULE}_{PTS}_z"]
    A = L.blocks(C, Z)
    k = Z.size

    zc, wc, _ = L.classic_ecm(C, k - 1, U=L.integrand_svd(C, fi)[0])
    e_fix = np.median(L.const_err(L.blocks(C, zc)[ri], b[ri], wc))
    print(f"{RULE}, {k}-point MAW support, {fi.size} fit states")
    print(f"  classic ECM at {k} pts   {e_fix:.4e}")
    print(f"  network, 100k epochs     {TARGET_ANN:.4e}   <- to beat")
    print(f"  classic ECM at 88 pts    {REFERENCE_88:.4e}\n", flush=True)

    Nz = R.zero_sum_basis(k)
    w0 = np.full(k, float(ne) / k)
    rng = np.random.default_rng(3)

    best = None
    print(f"{'kernel':>17} {'M':>5} {'eps_scale':>10} {'lambda':>9} "
          f"{'selection':>11}")
    print("-" * 58)
    for M in N_CENTERS:
        cen = q[fi[rng.permutation(fi.size)[:M]]]
        r0 = np.median(np.sort(
            R.cdist(q[fi], cen), axis=1)[:, 0])
        for kern in R.KERNELS:
            for a in EPS_SCALES:
                eps = a / max(r0, 1e-300)
                t0 = time.perf_counter()
                sols = R.fit_rbf(q[fi], A[fi], b[fi], cen, kern, eps, LAMS,
                                 Nz, w0)
                loc = None
                for lam, Cc in sols.items():
                    Ws = R.eval_rbf(q[si], Cc, cen, kern, eps, Nz, w0)
                    e = np.median(L.const_err(A[si], b[si], Ws))
                    if not np.isfinite(e):
                        continue
                    if loc is None or e < loc[0]:
                        loc = (e, lam)
                    if best is None or e < best["e"]:
                        best = dict(e=e, kern=kern, M=M, a=a, lam=lam, C=Cc,
                                    cen=cen, eps=eps)
                if loc is None:
                    print(f"{kern:>17} {M:>5} {a:>10.2f} {'-':>9} "
                          f"{'no finite':>11}", flush=True)
                    continue
                mark = " *" if (best is not None and best["kern"] == kern
                                and best["M"] == M and best["a"] == a) else ""
                print(f"{kern:>17} {M:>5} {a:>10.2f} {loc[1]:>9.0e} "
                      f"{loc[0]:>11.4e}   [{time.perf_counter() - t0:.0f}s]"
                      f"{mark}", flush=True)

    if best is None:
        print("no configuration produced a finite error")
        return 1

    print(f"\n=== best on selection: {best['kern']}, M={best['M']}, "
          f"eps_scale={best['a']}, lambda={best['lam']:.0e} ===")
    for tag, cl in (("raw", False), ("clip+renorm", True)):
        Wr = R.eval_rbf(q[ri], best["C"], best["cen"], best["kern"],
                        best["eps"], Nz, w0, clip=cl)
        Wf = R.eval_rbf(q[fi], best["C"], best["cen"], best["kern"],
                        best["eps"], Nz, w0, clip=cl)
        er = np.median(L.const_err(A[ri], b[ri], Wr))
        ef = np.median(L.const_err(A[fi], b[fi], Wf))
        s = Wr.sum(axis=0)
        print(f"  {tag:<12} fit {ef:.4e}   REPORT {er:.4e}   "
              f"min w {Wr.min():+.3e}   sum {s.min():.6f}..{s.max():.6f}")
        if not cl:
            neg = float(np.mean(Wr.min(axis=0) < 0.0))
            print(f"  {'':<12} states with a negative weight: {100 * neg:.1f}%")
            e_report = er

    print(f"\n  vs network (100k epochs): {TARGET_ANN / e_report:.2f}x")
    print(f"  vs classic at {k} pts:      {e_fix / e_report:.2f}x")
    print(f"  vs classic at 88 pts:      {REFERENCE_88 / e_report:.2f}x")

    # Normal equations square the condition number. Re-solve the winner with a
    # direct least-squares factorization and compare, rather than assume.
    Phi = R.basis(q[fi], best["cen"], best["kern"], best["eps"])
    nrm = np.maximum(np.linalg.norm(b[fi], axis=1), 1e-300)
    AN = np.matmul(A[fi], Nz)
    D = (Phi[:, None, :, None] * AN[:, :, None, :]).reshape(
        fi.size, A.shape[1], -1) / nrm[:, None, None]
    y = ((b[fi] - np.einsum("jmk,k->jm", A[fi], w0)) / nrm[:, None]).reshape(-1)
    Cd = np.linalg.lstsq(D.reshape(-1, D.shape[2]), y, rcond=None)[0].reshape(
        Phi.shape[1], Nz.shape[1])
    Wd = R.eval_rbf(q[ri], Cd, best["cen"], best["kern"], best["eps"], Nz, w0)
    print(f"\n  direct lstsq (unregularized) REPORT "
          f"{np.median(L.const_err(A[ri], b[ri], Wd)):.4e}")

    np.savez_compressed(L.HERE / f"rbf_{RULE}_{k}.npz",
                        C=best["C"], centers=best["cen"], N=Nz, w0=w0,
                        kernel=np.array(best["kern"]), eps=best["eps"],
                        lam=best["lam"], z=Z)
    print("\nRBF_SWEEP_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
