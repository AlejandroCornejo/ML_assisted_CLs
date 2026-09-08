#!/usr/bin/env python3
"""Quick, small-scale timing check of the EXISTING fom_nested_pk2_2d_
vectorized_parallel (already built earlier this session, 16-worker
process pool) on a handful of real macro strains, to get a CURRENT,
empirical per-point cost before estimating a full n_body=6/20-step run's
real wall-clock cost. Deliberately small (32 points) since even this is
expected to take a while (7 full 400-substep FOM solves per point)."""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fom_nested_law_parallel_claude as par_module  # noqa: E402
par_module.ensure_persistent_executor(n_workers=16)

d = np.load(HERE / "cruciform_results_hprom_iterative_f64_continuation_claude.npz")
e_gp = d["e_gp"]

N = 32
rng = np.random.default_rng(6)
idx = rng.choice(e_gp.shape[0], size=N, replace=False)
E_call = e_gp[idx]

print(f"[quick_time] calling fom_nested_pk2_2d_vectorized_parallel on {N} real points, "
      f"16 workers...", flush=True)
t0 = time.perf_counter()
S, CC = par_module.fom_nested_pk2_2d_vectorized_parallel(E_call, n_workers=16, verbose=True)
t1 = time.perf_counter() - t0
print(f"[quick_time] {N} points: {t1:.1f}s wall ({t1 / N:.3f}s/point wall-clock, "
      f"{t1 / N / 7:.3f}s/solve-equivalent)", flush=True)
print("QUICK_TIME_DONE_MARKER", flush=True)
