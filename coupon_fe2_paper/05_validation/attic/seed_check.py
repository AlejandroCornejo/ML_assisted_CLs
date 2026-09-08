#!/usr/bin/env python3
"""Refit the deployed MAW stress fields under different seeds.

TWO CLAIMS TO TEST, both resting on a single fit each so far.

  1. The headline: the 10-point adaptive rule reproduces the classic 75-point
     rule in and out of envelope. If that moves materially with the seed it is
     not a property of the method.
  2. The anomaly: MAW-15 came out 3.4x WORSE than MAW-10 out of envelope
     despite a better in-envelope constraint error. There is no mechanism for
     that, so the first thing to rule out is that it is one bad fit.

The support is held FIXED (the phase-2 pruning is deterministic given its
inputs); only the field's initialization and training seed change. Written to
separate files so `deploy_maw.py --fields` can run each unchanged.
"""
from __future__ import annotations

import sys
import time

import numpy as np

import maw_lab as L

SEEDS = (23, 37)
PTS = (10, 15)
HIDDEN = (128, 128, 128)
WARM_EPOCHS = 5000
EPOCHS = 20000
PATIENCE = 4000
LR = 1.5e-3


def main():
    d = L.load()
    q, ne = d["q"], d["ne"]
    C = d["C_sig"]
    b = L.targets(C)
    fi, vi = L.split(q.shape[0])
    ph = np.load(L.HERE / "maw_phase2.npz")

    for seed in SEEDS:
        store = {}
        for npts in PTS:
            Z = ph[f"sig_{npts}_z"]
            A = L.blocks(C, Z)
            Wstar = ph[f"sig_{npts}_W_oracle"]
            t0 = time.perf_counter()
            m = L.fit_field(q, A, b, ne, fi, vi, hidden=HIDDEN, epochs=EPOCHS,
                            patience=PATIENCE, lr=LR, seed=seed,
                            warm_target=Wstar, warm_epochs=WARM_EPOCHS,
                            label=f"s{seed}p{npts}")
            e = np.median(L.const_err(A[vi], b[vi], m["W"][:, vi]))
            print(f"seed {seed}, {npts} pts: constraint {e:.4e} "
                  f"[{time.perf_counter() - t0:.0f}s ep{m['best_epoch']}]",
                  flush=True)
            store[f"sig_{npts}_z"] = Z
            store[f"sig_{npts}_W_field"] = m["W"]
            store[f"sig_{npts}_W_oracle"] = Wstar
            store[f"sig_{npts}_mu"] = m["mu"]
            store[f"sig_{npts}_sd"] = m["sd"]
            store[f"sig_{npts}_act"] = np.array(m["act"])
            for k, v in m["state"].items():
                store[f"sig_{npts}_net_{k}"] = v
        np.savez_compressed(L.HERE / f"maw_seed{seed}.npz", **store)
    print("SEED_CHECK_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
