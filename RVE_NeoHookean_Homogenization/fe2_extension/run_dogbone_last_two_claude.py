#!/usr/bin/env python3
"""Final two: pann_regression, pann_free at delta_x_final=1.5 for the
dogbone. linear_hprom_continuation already completed successfully and was
saved (dogbone_results_linear_hprom_continuation_claude.npz, fully_converged
=True, in-box E11/E22/g12) despite the process that ran it dying silently
right afterward, for reasons not yet understood -- not pursued further
since the saved data itself is valid and complete."""
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

MODELS = ["pann_regression", "pann_free"]

if __name__ == "__main__":
    results = {}
    for which in MODELS:
        print(f"\n=== {which} (delta_x_final=1.5) ===", flush=True)
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
    print("DOGBONE_LAST_TWO_DONE_MARKER", flush=True)
