#!/usr/bin/env python3
"""Test LinearHpromParallelContinuationWrapper ALONE in a clean process
(no serial-law Kratos use in this same process before or after pool
creation) -- isolates whether the parallel path itself works and how fast
it is, before attempting any same-process serial-vs-parallel comparison
(which risks the documented fork-after-Kratos-threading hazard if this
process's own prior Kratos use delays the pool's actual worker fork)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import linear_hprom_law_parallel_claude as par_module  # noqa: E402
par_module.ensure_persistent_executor(n_workers=16)

d = np.load(HERE / "cruciform_results_hprom_iterative_f64_continuation_claude.npz")
e_gp = d["e_gp"]

N = 120
rng = np.random.default_rng(3)
idx = rng.choice(e_gp.shape[0], size=N, replace=False)
E_call = e_gp[idx]

wrapper = par_module.LinearHpromParallelContinuationWrapper(n_workers=16)

print(f"[test] calling parallel wrapper on {N} real points, cold start...", flush=True)
t0 = time.perf_counter()
S, CC = wrapper(E_call)
t1 = time.perf_counter() - t0
print(f"[test] call 0: {t1:.3f}s ({t1 / N * 1000:.3f} ms/point wall), "
      f"converged tracked={len(wrapper.q_prev_by_point)}/{N}", flush=True)
print(f"[test] S range: min={S.min():.3e}, max={S.max():.3e}", flush=True)
print(f"[test] CC range: min={CC.min():.3e}, max={CC.max():.3e}", flush=True)

t0 = time.perf_counter()
S2, CC2 = wrapper(E_call * 1.02)
t2 = time.perf_counter() - t0
print(f"[test] call 1 (warm start): {t2:.3f}s ({t2 / N * 1000:.3f} ms/point wall)", flush=True)

np.savez(HERE / "_parallel_only_test_result_claude.npz", S=S, CC=CC, E=E_call, S2=S2, CC2=CC2)
print("TEST_DONE_MARKER", flush=True)
