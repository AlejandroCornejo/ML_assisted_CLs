#!/usr/bin/env python3
"""Give the residual weight field a budget long enough to actually converge.

WHY. Every residual fit so far stopped at or near its epoch cap with the best
epoch being the LAST one -- the 80-point fit's best was epoch 20000 of 20000.
It was never converged, so "the residual cannot be regressed" was not something
the runs had established. The generalization gap is 1.18x-1.41x throughout, so
the limit is not generalization either; it is optimization on a badly
conditioned loss.

The support is selected on the command line. Measured so far, at 80 points:
20k epochs gave 9.78e-03, 100k gave 2.78e-03 -- a 3.5x gain from budget alone,
which settles that the earlier runs were cut short rather than converged. It
still only ties the classic fixed-weight rule at the same count (0.86x).

At 10 points there is reason to expect better rather than worse: non-negativity
is active at 8.3% of states there against 100% at 80 points, and the network
predicts 10 outputs instead of 80 from the same 4208 training states.
"""
from __future__ import annotations

import sys
import time

import numpy as np

import maw_lab as L

EPOCHS = 100000
# Early stopping is ON, and it is now safe to use because it watches the
# MEDIAN relative constraint error -- the metric the tables quote -- instead of
# the loss. Watching the loss would have stopped the 10-point run at epoch
# 16160 and kept a model 2.2x worse in the reported metric. 20000 is generous
# on purpose: the median improved slowly but MONOTONICALLY all the way to
# 100000 epochs, so a tight patience would cut a run that is still gaining.
PATIENCE = 20000
WARM_EPOCHS = 5000
LOG_EVERY = 5000
HIDDEN = (128, 128, 128)
LR = 1.5e-3
REFERENCE = 1.5805e-03   # classic ECM, 88 points, fixed weights


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pts", type=int, default=80)
    ap.add_argument("--src", default=None)
    ap.add_argument("--out", default=None)
    _a = ap.parse_args()
    src = _a.src or ("maw_phase2_res80.npz" if _a.pts == 80
                     else "maw_phase2_res.npz")
    outname = _a.out or f"maw_res_long{_a.pts}.npz"

    d = L.load()
    q, ne = d["q"], d["ne"]
    C = d["C_res"]
    b = L.targets(C)
    fi, vi = L.split(q.shape[0])
    print(f"classic ECM 88-point reference: {REFERENCE:.4e}", flush=True)

    store = {}
    for npts, src in ((int(_a.pts), src),):
        ph = np.load(L.HERE / src)
        Z = ph[f"res_{npts}_z"]
        Wstar = ph[f"res_{npts}_W_oracle"]
        A = L.blocks(C, Z)
        zc, wc, _ = L.classic_ecm(C, Z.size - 1,
                                  U=L.integrand_svd(C, fi)[0])
        e_fix = np.median(L.const_err(L.blocks(C, zc)[vi], b[vi], wc))
        print(f"\n=== residual, {Z.size} points "
              f"(classic at same count: {e_fix:.4e}) ===", flush=True)
        t0 = time.perf_counter()
        m = L.fit_field(q, A, b, ne, fi, vi, hidden=HIDDEN, epochs=EPOCHS,
                        patience=PATIENCE, lr=LR, warm_target=Wstar,
                        warm_epochs=WARM_EPOCHS, verbose=True,
                        log_every=LOG_EVERY, label=f"res{Z.size}")
        ef = np.median(L.const_err(A[fi], b[fi], m["W"][:, fi]))
        e = m["median_report"]
        print(f"  FINAL {Z.size} pts: fit {ef:.4e}  "
              f"selection half {m['median_select']:.4e}  "
              f"REPORT half {e:.4e}", flush=True)
        print(f"  ({time.perf_counter() - t0:.0f}s, best ep {m['best_epoch']}, "
              f"selected by {m['select_by']})", flush=True)
        print(f"  vs classic at {Z.size} pts: {e_fix / e:.2f}x   "
              f"vs 88-point reference: {REFERENCE / e:.2f}x", flush=True)
        store[f"res_{Z.size}_z"] = Z
        store[f"res_{Z.size}_W_field"] = m["W"]
        store[f"res_{Z.size}_W_oracle"] = Wstar
        store[f"res_{Z.size}_mu"] = m["mu"]
        store[f"res_{Z.size}_sd"] = m["sd"]
        store[f"res_{Z.size}_act"] = np.array(m["act"])
        for k, v in m["state"].items():
            store[f"res_{Z.size}_net_{k}"] = v
        np.savez_compressed(L.HERE / outname, **store)

    print("\nRES_LONG_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
