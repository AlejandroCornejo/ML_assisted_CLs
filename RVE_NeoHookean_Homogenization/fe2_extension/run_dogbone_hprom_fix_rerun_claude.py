#!/usr/bin/env python3
"""Re-runs just dhprom_f64_consistent and hprom_iterative_f64_consistent for
the dogbone at delta_x_final=1.5, now that dhprom_ann_direct_law_float64_
claude.py / hprom_ann_iterative_law_float64_claude.py's hom_sig computation
has been fixed (was routing through an incompletely-migrated reaction-force
integrand that silently returned zero stress against this project's actual,
currently-deployed MAW-ECM sig weights -- confirmed both before, by direct
isolated testing, and after, by comparing against the true nested-FOM
reaction-force stress at real saved cruciform states: the fix is measurably
MORE accurate than what was previously saved/reported, not just non-zero).
Sets the same thread-limiting env vars every other sweep/batch script this
session sets, which the original (buggy) dogbone batch launch omitted."""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_dogbone_fe2_claude import run_newton_fe2_dogbone  # noqa: E402

MODELS = ["dhprom_f64_consistent", "hprom_iterative_f64_consistent"]

if __name__ == "__main__":
    results = {}
    for which in MODELS:
        print(f"\n=== {which} (delta_x_final=1.5, POST-FIX) ===", flush=True)
        res = run_newton_fe2_dogbone(which, delta_x_final=1.5, verbose=False, save_npz=True)
        results[which] = res
        print(f"[{which}] fully_converged={res['fully_converged']}  ever_diverged={res['ever_diverged']}  "
              f"E11={res['e11_range']}  E22={res['e22_range']}  g12={res['g12_range']}  "
              f"wall={res['wall_time']:.1f}s", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    for which, res in results.items():
        print(f"  {which:32s}  conv={res['fully_converged']}  diverged={res['ever_diverged']}  "
              f"E11=[{res['e11_range'][0]:+.4f},{res['e11_range'][1]:+.4f}]  "
              f"E22=[{res['e22_range'][0]:+.4f},{res['e22_range'][1]:+.4f}]  "
              f"g12=[{res['g12_range'][0]:+.4f},{res['g12_range'][1]:+.4f}]  wall={res['wall_time']:.1f}s", flush=True)
    print("DOGBONE_HPROM_FIX_RERUN_DONE_MARKER", flush=True)
