#!/usr/bin/env python3
"""Test HpromAnnParallelContinuationWrapper ALONE in a clean process (no
serial-law Kratos/torch use in this same process before or after pool
creation) -- isolates whether the two-level (multiprocessing + per-worker
vmap batching) parallel path works and how fast it is."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import hprom_ann_law_parallel_claude as par_module  # noqa: E402
par_module.ensure_persistent_executor(n_workers=16)

HPROMANN_DIR = HERE / "maw_dynamic_reaction_force_hpromann"

d = np.load(HERE / "cruciform_results_hprom_iterative_f64_continuation_claude.npz")
e_gp = d["e_gp"]

N = 600  # full realistic macro-point count
rng = np.random.default_rng(4)
idx = rng.choice(e_gp.shape[0], size=min(N, e_gp.shape[0]), replace=True)
E_call = e_gp[idx]

wrapper = par_module.HpromAnnParallelContinuationWrapper(hprom_ann_dir=str(HPROMANN_DIR), n_workers=16)

print(f"[test] calling parallel wrapper on {len(E_call)} real points, cold start...", flush=True)
t0 = time.perf_counter()
S, CC = wrapper(E_call)
t1 = time.perf_counter() - t0
print(f"[test] call 0: {t1:.3f}s ({t1 / len(E_call) * 1000:.3f} ms/point wall), "
      f"converged tracked={len(wrapper.q_prev_by_point)}/{len(E_call)}", flush=True)

t0 = time.perf_counter()
S2, CC2 = wrapper(E_call * 1.02)
t2 = time.perf_counter() - t0
print(f"[test] call 1 (warm start): {t2:.3f}s ({t2 / len(E_call) * 1000:.3f} ms/point wall)", flush=True)

np.savez(HERE / "_parallel_only_test_hprom_ann_result_claude.npz", S=S, CC=CC, E=E_call, S2=S2, CC2=CC2)
print("TEST_DONE_MARKER", flush=True)
