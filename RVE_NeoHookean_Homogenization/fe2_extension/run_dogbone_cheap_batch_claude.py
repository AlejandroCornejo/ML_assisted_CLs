#!/usr/bin/env python3
"""Runs the 5 cheap survivor models (ICNN, ICKAN, D-HPROM-ANN, HPROM-ANN-
iterative, Linear-HPROM) plus, per explicit request, the two disqualified
tiers (Regression, Free) -- all at the calibrated delta_x_final=1.5, where
the ICNN check already confirmed the full macro-strain range stays inside
the trained box. Run sequentially (not concurrently), and BEFORE the
separate, long-running fom_nested job, per this project's own serial-only
benchmarking discipline -- these wall times are still reported, so no
CPU contention with anything else while they run."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_dogbone_fe2_claude import run_newton_fe2_dogbone  # noqa: E402
from run_cruciform_fe2_claude import MATERIAL_FUNCS, make_linear_hprom_continuation_material_func  # noqa: E402

MATERIAL_FUNCS["linear_hprom_continuation"] = make_linear_hprom_continuation_material_func()

MODELS = [
    "pann_certified", "pann_ickan", "dhprom_f64_consistent",
    "hprom_iterative_f64_consistent", "linear_hprom_continuation",
    "pann_regression", "pann_free",
]

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
    print("DOGBONE_CHEAP_BATCH_DONE_MARKER", flush=True)
