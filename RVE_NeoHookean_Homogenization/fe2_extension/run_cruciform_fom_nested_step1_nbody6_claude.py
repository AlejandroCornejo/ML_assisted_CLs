#!/usr/bin/env python3
"""FOM-FE2 cruciform, n_body=6 (same mesh/loading as every other model's
own cruciform row this session), STEP 1 ONLY out of the standard 20-step
ramp -- via run_newton_fe2_cruciform's new max_steps_to_run=1, which stops
after step 1 while still using frac=1/20 for that step's own displacement
increment (the SAME small first-step size the other models' full runs
used, not the much larger single-step jump n_steps=1 alone would give).

Real measured cost (this session, 32-point sample, 16-way parallel,
already-existing fom_nested_law_parallel_claude.py): ~3.45s/point wall,
600 points/16 workers -> ~34.5 min per macro-Newton material call. Step 1
typically needs a handful of such calls (matching iters_per_step~4 seen
for every other model this session) -- expect on the order of 2-3 hours
for this single step, not a fast smoke test.

Same fork-after-Kratos/torch-threading discipline as every other parallel
driver this session: create the persistent pool before importing anything
Kratos-related."""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

N_BODY, N_ARM_LEN = 6, 4
DELTA = 1.2
N_STEPS = 20
N_WORKERS = 16

from fom_nested_law_parallel_claude import ensure_persistent_executor  # noqa: E402
ensure_persistent_executor(n_workers=N_WORKERS)

from run_cruciform_fe2_claude import run_newton_fe2_cruciform  # noqa: E402


def main():
    print(f"=== FOM-FE2 (cruciform, n_body={N_BODY}, n_arm_len={N_ARM_LEN}, delta={DELTA}, "
          f"n_steps={N_STEPS} config, STOPPING after step 1, {N_WORKERS}-way parallel FD tangent) ===",
          flush=True)
    t0 = time.time()
    res = run_newton_fe2_cruciform(
        "fom_nested", n_body=N_BODY, n_arm_len=N_ARM_LEN, n_steps=N_STEPS, max_steps_to_run=1,
        delta_x_final=DELTA, delta_y_final=DELTA, verbose=True, use_line_search=True, save_npz=True,
    )
    d = np.load(HERE / "cruciform_results_fom_nested_claude.npz")
    g12 = np.abs(d["e_gp"][:, 2])
    print(f"[fom_nested_step1] fully_converged={res['fully_converged']}, ever_diverged={res['ever_diverged']}, "
          f"E11={res['e11_range']}, E22={res['e22_range']}, g12={res['g12_range']}, "
          f"n_material_calls={res['n_material_calls']}, "
          f"wall={res['wall_time']:.1f}s (elapsed {time.time() - t0:.1f}s)", flush=True)
    print("FOM_NESTED_STEP1_DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
