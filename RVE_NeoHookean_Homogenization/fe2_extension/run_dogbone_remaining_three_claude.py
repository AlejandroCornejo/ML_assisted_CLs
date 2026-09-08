#!/usr/bin/env python3
"""Runs the 3 models that never got a clean, uncontended run in this
session's earlier dogbone batches: linear_hprom_continuation (the earlier
attempt appeared to hang -- killed by user request -- while running back-
to-back with dhprom/hprom_iterative under the original batch script, which
omitted the thread-limiting env vars every other sweep/batch script this
session sets; this fresh, isolated, properly-limited process may simply not
reproduce that), pann_regression, pann_free (both disqualified by the (C1)/
(C5) certificate violations elsewhere in this project, but requested here
anyway as an honest extra data point: do they converge cleanly when the
visited state genuinely stays inside the trained box, unlike at Cook?)."""
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
from run_cruciform_fe2_claude import MATERIAL_FUNCS, make_linear_hprom_continuation_material_func  # noqa: E402

MATERIAL_FUNCS["linear_hprom_continuation"] = make_linear_hprom_continuation_material_func()

MODELS = ["linear_hprom_continuation", "pann_regression", "pann_free"]

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
    print("DOGBONE_REMAINING_THREE_DONE_MARKER", flush=True)
