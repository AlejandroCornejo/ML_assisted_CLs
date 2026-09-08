#!/usr/bin/env python3
"""Test the D-HPROM-ANN parallel material function ALONE in a clean
process (no serial-law Kratos/torch use in this same process) -- isolates
whether the parallel path works and how fast it is."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dhprom_ann_law_parallel_claude as par_module  # noqa: E402
par_module.ensure_persistent_executor(n_workers=16)

DHPROMANN_DIR = HERE / "maw_dynamic_reaction_force_dhpromann"
material_func = par_module.make_dhprom_ann_parallel_material_func(DHPROMANN_DIR, n_workers=16)

d = np.load(HERE / "cruciform_results_hprom_iterative_f64_continuation_claude.npz")
e_gp = d["e_gp"]

N = 600
rng = np.random.default_rng(5)
idx = rng.choice(e_gp.shape[0], size=min(N, e_gp.shape[0]), replace=True)
E_call = e_gp[idx]

print(f"[test] calling parallel D-HPROM-ANN material func on {len(E_call)} real points...", flush=True)
t0 = time.perf_counter()
S, CC = material_func(E_call)
t1 = time.perf_counter() - t0
print(f"[test] call 0: {t1:.3f}s ({t1 / len(E_call) * 1000:.3f} ms/point wall)", flush=True)

t0 = time.perf_counter()
S2, CC2 = material_func(E_call * 1.02)
t2 = time.perf_counter() - t0
print(f"[test] call 1 (workers already built): {t2:.3f}s ({t2 / len(E_call) * 1000:.3f} ms/point wall)", flush=True)

np.savez(HERE / "_parallel_only_test_dhprom_ann_result_claude.npz", S=S, CC=CC, E=E_call, S2=S2, CC2=CC2)
print("TEST_DONE_MARKER", flush=True)
