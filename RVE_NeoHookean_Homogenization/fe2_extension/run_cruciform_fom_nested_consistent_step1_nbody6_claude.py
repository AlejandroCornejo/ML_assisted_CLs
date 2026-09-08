#!/usr/bin/env python3
"""FOM-FE2 cruciform, n_body=6, STEP 1 ONLY (via max_steps_to_run=1, same
small first-step increment as the standard 20-step ramp), now using the
CORRECTED, consistent law (fom_nested_consistent_law_claude.py: reaction-
force stress + analytic tangent) with process parallelism -- replacing
the earlier FD-tangent/naive-average-stress attempt that was stopped.

Expected to be dramatically faster than the earlier attempt: ~1s/point
(one solve + one cheap sparse linear solve) instead of ~3.45s/point
(seven full nonlinear solves for the FD tangent alone)."""
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

from fom_nested_consistent_law_parallel_claude import ensure_persistent_executor  # noqa: E402
ensure_persistent_executor(n_workers=N_WORKERS)

from run_cruciform_fe2_claude import run_newton_fe2_cruciform, make_fom_nested_consistent_parallel_material_func  # noqa: E402
import run_cruciform_fe2_claude as rc  # noqa: E402

rc.MATERIAL_FUNCS["fom_nested_consistent_parallel"] = make_fom_nested_consistent_parallel_material_func(
    n_workers=N_WORKERS,
)


def main():
    print(f"=== FOM-FE2 CONSISTENT (cruciform, n_body={N_BODY}, n_arm_len={N_ARM_LEN}, delta={DELTA}, "
          f"n_steps={N_STEPS} config, STOPPING after step 1, {N_WORKERS}-way parallel) ===", flush=True)
    t0 = time.time()
    res = run_newton_fe2_cruciform(
        "fom_nested_consistent_parallel", n_body=N_BODY, n_arm_len=N_ARM_LEN, n_steps=N_STEPS,
        max_steps_to_run=1, delta_x_final=DELTA, delta_y_final=DELTA,
        verbose=True, use_line_search=True, save_npz=True,
    )
    d = np.load(HERE / "cruciform_results_fom_nested_consistent_parallel_claude.npz")
    g12 = np.abs(d["e_gp"][:, 2])
    print(f"[fom_nested_consistent_step1] fully_converged={res['fully_converged']}, "
          f"ever_diverged={res['ever_diverged']}, E11={res['e11_range']}, E22={res['e22_range']}, "
          f"g12={res['g12_range']}, n_material_calls={res['n_material_calls']}, "
          f"wall={res['wall_time']:.1f}s (elapsed {time.time() - t0:.1f}s)", flush=True)
    print("FOM_NESTED_CONSISTENT_STEP1_DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
