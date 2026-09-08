#!/usr/bin/env python3
"""Final dogbone geometry (W_gauge=4.0, W_grip=4.25 i.e. D/d=1.0625, R=2.0
-- the gentle-shoulder design selected after mapping the shoulder-geometry
parameter space this session, ~2.7x the achievable in-box E11 of the
original D/d=2.0 design), displacement-controlled at delta_x_final=4.4
(calibrated: worst|g12|=0.096, worst|E22|=0.100, both just inside the
trained box), all 7 model variants (the 6 survivors + regression/free as an
honest extra data point, matching this session's own established
convention). Runs sequentially in one process (never concurrent timed
jobs), with the thread-limiting env vars every sweep this session sets."""
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

GEOM = dict(W_gauge=4.0, W_grip=4.25, R=2.0)  # L_gauge/L_grip stay at their own defaults (8.0/6.0)
DELTA = 4.4

MODELS = [
    "pann_certified", "pann_ickan",
    "dhprom_f64_consistent", "hprom_iterative_f64_consistent", "linear_hprom_continuation",
    "pann_regression", "pann_free",
]

if __name__ == "__main__":
    results = {}
    for which in MODELS:
        print(f"\n=== {which} (delta_x_final={DELTA}, {GEOM}) ===", flush=True)
        res = run_newton_fe2_dogbone(which, delta_x_final=DELTA, verbose=False, save_npz=True, **GEOM)
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
    print("DOGBONE_FINAL_ALL_SEVEN_DONE_MARKER", flush=True)
