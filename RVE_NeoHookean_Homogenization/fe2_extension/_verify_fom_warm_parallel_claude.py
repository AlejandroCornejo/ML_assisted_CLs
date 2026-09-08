#!/usr/bin/env python3
"""End-to-end gate for the warm-started PARALLEL FOM-FE2 law.

Checks the thing that could plausibly break in the parallel version even
though the serial warm-start math is already verified: that each Gauss
point's state actually follows that point across successive calls through
the persistent pool (deterministic chunking + persistent workers), and that
the final answer still matches the existing cold parallel law.

Protocol mirrors what a real macro solve does: the same points are queried
three times at slightly growing strains (as consecutive macro Newton
iterates would be), warm-started throughout; the cold reference is queried
once at the final strain. If state were leaking between points or being
lost between calls, the final values would not match.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Pools first, before anything in this parent process can touch Kratos.
import fom_nested_consistent_warm_law_parallel_claude as warm_par  # noqa: E402
import fom_nested_consistent_law_parallel_claude as cold_par  # noqa: E402

N_WORKERS = 8
warm_par.ensure_persistent_executor(n_workers=N_WORKERS)
cold_par.ensure_persistent_executor(n_workers=N_WORKERS)

# 8 distinct points spanning the dogbone's own final-state range, so a
# state mix-up between points would show up as a visible mismatch.
E_TARGET = np.array([
    [0.2578, -0.0968, 0.0825],
    [0.2400, -0.0900, -0.0700],
    [0.2000, -0.0800, 0.0400],
    [0.1800, -0.0700, -0.0300],
    [0.1700, -0.0600, 0.0100],
    [0.2200, -0.0850, 0.0600],
    [0.2500, -0.0950, -0.0800],
    [0.1600, -0.0550, 0.0050],
])
FRACS = [0.90, 0.95, 1.00]


def rel(a, b):
    den = max(np.linalg.norm(b), 1.0e-30)
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / den)


if __name__ == "__main__":
    print(f"{E_TARGET.shape[0]} points, {N_WORKERS} workers, "
          f"{len(FRACS)} successive warm-started calls at fracs {FRACS}", flush=True)

    t_warm_total = 0.0
    warm_fracs = []
    for f in FRACS:
        t0 = time.perf_counter()
        S_warm, CC_warm = warm_par.fom_nested_consistent_warm_pk2_2d_vectorized_parallel(
            E_TARGET * f, n_workers=N_WORKERS, verbose=False)
        dt = time.perf_counter() - t0
        t_warm_total += dt
        wf = warm_par.last_warm_fraction()
        warm_fracs.append(wf)
        print(f"  [warm] frac={f:.2f}  {dt:.1f}s   warm-started {wf * 100:.0f}% of points", flush=True)

    t0 = time.perf_counter()
    S_cold, CC_cold = cold_par.fom_nested_consistent_pk2_2d_vectorized_parallel(
        E_TARGET, n_workers=N_WORKERS, verbose=False)
    t_cold = time.perf_counter() - t0
    print(f"  [cold] frac=1.00  {t_cold:.1f}s  (single call, from zero)", flush=True)

    err_S = rel(S_warm, S_cold)
    err_CC = rel(CC_warm, CC_cold)
    per_point = [rel(S_warm[i], S_cold[i]) for i in range(E_TARGET.shape[0])]

    print(f"\nfinal-state agreement warm vs cold:  S {err_S:.3e}   CC {err_CC:.3e}", flush=True)
    print(f"worst single point:                  {max(per_point):.3e}", flush=True)
    print(f"last warm call vs cold call:         {t_warm_total / len(FRACS):.1f}s avg warm "
          f"vs {t_cold:.1f}s cold", flush=True)

    # Values agreeing is necessary but NOT sufficient: a path that silently
    # cold-started would agree just as well. The mechanism itself has to be
    # observed -- nothing to continue from on call 1, everything continued
    # from on calls 2+.
    mech_ok = (warm_fracs[0] == 0.0) and all(wf > 0.99 for wf in warm_fracs[1:])
    print(f"warm-start engagement per call:      {[f'{w:.2f}' for w in warm_fracs]} "
          f"(expect first 0.00, rest ~1.00)", flush=True)

    ok = (err_S < 1.0e-5) and (err_CC < 1.0e-5) and (max(per_point) < 1.0e-5) and mech_ok
    print("WARM_PARALLEL_VERIFY_PASS" if ok else "WARM_PARALLEL_VERIFY_FAIL", flush=True)
