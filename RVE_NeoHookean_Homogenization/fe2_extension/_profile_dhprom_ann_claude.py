#!/usr/bin/env python3
"""Profile DHpromAnnDirectLawFloat64.evaluate_with_tangent on real macro
strains from the just-completed n_body=6 cruciform run, to find out
empirically where the per-Gauss-point wall time actually goes, rather
than trust a possibly-stale profiling comment in the source."""
from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dhprom_ann_direct_law_float64_claude import DHpromAnnDirectLawFloat64  # noqa: E402

DHPROMANN_DIR = HERE / "maw_dynamic_reaction_force_dhpromann"

d = np.load(HERE / "cruciform_results_dhprom_f64_consistent_claude.npz")
e_gp = d["e_gp"]
print(f"[profile] loaded {e_gp.shape[0]} real macro Gauss-point strains from the n_body=6 run")

law = DHpromAnnDirectLawFloat64(hprom_ann_dir=str(DHPROMANN_DIR))

# Warm-up (first call pays one-time lazy-init costs we don't want polluting the measurement)
_ = law.evaluate_with_tangent(e_gp[0])

N_SAMPLE = 300
rng = np.random.default_rng(0)
sample_idx = rng.choice(e_gp.shape[0], size=min(N_SAMPLE, e_gp.shape[0]), replace=False)

t0 = time.perf_counter()
for i in sample_idx:
    law.evaluate_with_tangent(e_gp[i])
t1 = time.perf_counter()
print(f"[profile] plain loop: {sample_idx.size} calls, {t1-t0:.3f}s total, {(t1-t0)/sample_idx.size*1000:.3f} ms/call")

pr = cProfile.Profile()
pr.enable()
for i in sample_idx:
    law.evaluate_with_tangent(e_gp[i])
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(25)
print(s.getvalue())
print("PROFILE_DONE_MARKER")
