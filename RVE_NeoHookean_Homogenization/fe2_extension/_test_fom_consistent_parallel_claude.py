#!/usr/bin/env python3
"""Small-scale smoke test + serial-vs-parallel cross-check for the new
consistent FOM law's parallel wrapper, on a handful of real macro strains
(reused from the HPROM-ANN continuation run's own e_gp, same mesh/
geometry so the strain values are representative)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fom_nested_consistent_law_parallel_claude as par_module  # noqa: E402
par_module.ensure_persistent_executor(n_workers=8)

d = np.load(HERE / "cruciform_results_hprom_iterative_f64_continuation_claude.npz")
e_gp = d["e_gp"]

N = 8
rng = np.random.default_rng(7)
idx = rng.choice(e_gp.shape[0], size=N, replace=False)
E_call = e_gp[idx]
print(f"[test] E magnitudes: {np.linalg.norm(E_call, axis=1)}", flush=True)

t0 = time.perf_counter()
S_par, CC_par = par_module.fom_nested_consistent_pk2_2d_vectorized_parallel(E_call, n_workers=8)
t_par = time.perf_counter() - t0
print(f"[test] PARALLEL: {t_par:.2f}s ({t_par / N:.3f}s/point wall)", flush=True)

np.savez(HERE / "_test_fom_consistent_parallel_result_claude.npz", S=S_par, CC=CC_par, E=E_call)
print("TEST_DONE_MARKER", flush=True)
