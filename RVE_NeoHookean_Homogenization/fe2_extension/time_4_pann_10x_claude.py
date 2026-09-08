#!/usr/bin/env python3
"""Time the 4 PANN tiers over 10 independent repetitions each (matching
Table 7's own established convention for these specific rows -- a
single run of something this fast is dominated by incidental overhead,
not real compute, so a mean+std over repetitions is the more honest
number). Saves the .npz only on the LAST repetition of each tier (the
physics/accuracy numbers are deterministic across reps; only wall time
varies)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path("/home/kratos/ML_assisted_CLs_clean/RVE_NeoHookean_Homogenization/fe2_extension")
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_cook_hprom_ann_claude as cook_module  # noqa: E402
import run_cook_pann_w5_refresh_claude as w5_module  # noqa: E402  (registers pann_certified_w5/pann_ickan_w5)

N_REPS = 10
TIERS = ("pann_regression", "pann_free", "pann_certified_w5", "pann_ickan_w5")

if __name__ == "__main__":
    print(f"=== Timing 4 PANN tiers, {N_REPS} reps each, Cook nx=ny=8 ===", flush=True)
    for which in TIERS:
        walls = []
        for rep in range(N_REPS):
            t0 = time.time()
            res = cook_module.run_newton_fe2(
                which, nx=8, ny=8, n_steps=20, verbose=False, save_npz=(rep == N_REPS - 1),
            )
            walls.append(res["wall_time"])
        walls = np.array(walls)
        print(f"[{which}-detail] wall times over {N_REPS} reps: mean={walls.mean():.3f}s, "
              f"std={walls.std():.3f}s, min={walls.min():.3f}s, max={walls.max():.3f}s", flush=True)
        print(f"[{which}] wall={walls.mean():.3f}s", flush=True)
    print("PANN_10X_TIMING_DONE_MARKER", flush=True)
