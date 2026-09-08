#!/usr/bin/env python3
"""Train the deployable MAW-ECM weight fields, with a generous budget.

WHICH RULES, AND WHY THESE. The cancellation measurement decides it:

    integrand                  |sum_e c_e| / sum_e |c_e|
    homogenized stress          9.51e-01     (5% cancellation)
    projected residual          2.05e-02    (98% cancellation)

A cubature constrained to NON-NEGATIVE weights cannot cheaply reproduce an
integral whose value is a near-cancellation of its terms: it has no opposing
signs to work with, so it must instead place enormous weights on a few
elements. Measured at 10 points on the residual rule, one element carries 1330
of the total 1546, non-negativity is active at 98.8% of states, and the exact
weight vector changes by 12.6% between NEAREST NEIGHBOURS in q -- exact at each
state, but not a smooth function of q, hence not regressable.

The stress integrand has no such cancellation, and that is where an adaptive
rule pays. So the stress rule is trained here for deployment, and the residual
is left to the classic fixed-weight ECM -- which is, independently, exactly how
the previous project deployed MAW-ECM: on the homogenization target, not inside
the Newton loop.

The residual rule is still trained at a larger support, to report the honest
accuracy-vs-cost curve for it rather than only its failure at 10 points.
"""
from __future__ import annotations

import sys
import time

import numpy as np

import maw_lab as L

JOBS = (("sig", 10), ("res", 20), ("sig", 15), ("res", 30))
HIDDEN = (128, 128, 128)
WARM_EPOCHS = 6000
EPOCHS = 30000
PATIENCE = 5000
LR = 1.5e-3


def main():
    d = L.load()
    q, ne = d["q"], d["ne"]
    fi, vi = L.split(q.shape[0])
    print(f"states {q.shape[0]} ({fi.size} fit / {vi.size} held out)\n", flush=True)

    Us, store = {}, {}
    for nm, C in (("res", d["C_res"]), ("sig", d["C_sig"])):
        Us[nm] = L.integrand_svd(C, fi)[0]

    for nm, npts in JOBS:
        C = d["C_res"] if nm == "res" else d["C_sig"]
        b = L.targets(C)
        z, w, _ = L.classic_ecm(C, npts - 1, U=Us[nm])
        A = L.blocks(C, z)
        e_fix = np.median(L.const_err(A[vi], b[vi], w))
        Wstar, nc = L.optimal_weights(A, b, ne)
        e_star = np.median(L.const_err(A[vi], b[vi], Wstar[:, vi]))
        print(f"=== {nm}, {z.size} points ===", flush=True)
        print(f"  classic fixed w         {e_fix:.4e}")
        print(f"  exact adaptive (oracle) {e_star:.4e}   "
              f"bound active {100.0 * nc / A.shape[0]:.1f}%", flush=True)

        t0 = time.perf_counter()
        m = L.fit_field(q, A, b, ne, fi, vi, hidden=HIDDEN, epochs=EPOCHS,
                        patience=PATIENCE, lr=LR, warm_target=Wstar,
                        warm_epochs=WARM_EPOCHS, verbose=True,
                        label=f"{nm}{npts}")
        e = L.report("MAW field w(q)", A, b, m["W"], vi,
                     extra=f"[{time.perf_counter() - t0:.0f}s ep{m['best_epoch']}]")
        print(f"  gain over fixed w: {e_fix / e:.2f}x\n", flush=True)

        store[f"{nm}_{npts}_z"] = z
        store[f"{nm}_{npts}_w_fixed"] = w
        store[f"{nm}_{npts}_W_field"] = m["W"]
        store[f"{nm}_{npts}_W_oracle"] = Wstar
        store[f"{nm}_{npts}_mu"] = m["mu"]
        store[f"{nm}_{npts}_sd"] = m["sd"]
        store[f"{nm}_{npts}_act"] = np.array(m["act"])
        for k, v in m["state"].items():
            store[f"{nm}_{npts}_net_{k}"] = v
        np.savez_compressed(L.HERE / "maw_fields.npz", **store)

    print("MAW_FIELDS_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
