#!/usr/bin/env python3
"""Final dogbone setup per explicit user request: FORCE imposed (equal and
opposite consistent nodal tractions) at BOTH ends, not displacement-
controlled -- on the improved, properly-resolved mesh (552 elements,
size_far=2.0/size_near_fillet=0.5, retuned this session after the original
defaults left the gentle shoulder resolved by only 1-2 elements). Calibrated
force level F=1.0e9 (worst|g12|=0.083, worst|E22|=0.097, both safely inside
the trained box; E11 reaches 0.258). All 7 model variants, run sequentially
in one process."""
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

from run_dogbone_force_fe2_claude import run_newton_fe2_dogbone_force  # noqa: E402
from run_cruciform_fe2_claude import MATERIAL_FUNCS, make_linear_hprom_continuation_material_func  # noqa: E402

MATERIAL_FUNCS["linear_hprom_continuation"] = make_linear_hprom_continuation_material_func()

GEOM = dict(W_gauge=4.0, W_grip=4.25, R=2.0)
FORCE = 1.0e9

MODELS = [
    "pann_certified", "pann_ickan",
    "dhprom_f64_consistent", "hprom_iterative_f64_consistent", "linear_hprom_continuation",
    "pann_regression", "pann_free",
]

if __name__ == "__main__":
    results = {}
    for which in MODELS:
        print(f"\n=== {which} (F={FORCE:.2e}, {GEOM}) ===", flush=True)
        res = run_newton_fe2_dogbone_force(which, total_force_final=FORCE, verbose=False, save_npz=True, **GEOM)
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
    print("DOGBONE_FORCE_FINAL_SEVEN_DONE_MARKER", flush=True)
