#!/usr/bin/env python3
"""HPROM-ANN-iterative on the final dogbone geometry with 30 steps instead
of 20, to test whether finer load increments avoid the isolated step-10/20
divergence found with the coarser 20-step ramp (same total delta_x_final,
same geometry)."""
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

if __name__ == "__main__":
    res = run_newton_fe2_dogbone(
        "hprom_iterative_f64_consistent", W_gauge=4.0, W_grip=4.25, R=2.0,
        delta_x_final=4.4, n_steps=30, verbose=True, save_npz=True,
    )
    print(f"\n[hprom_iterative_f64_consistent] fully_converged={res['fully_converged']}  "
          f"ever_diverged={res['ever_diverged']}  E11={res['e11_range']}  E22={res['e22_range']}  "
          f"g12={res['g12_range']}  wall={res['wall_time']:.1f}s", flush=True)
    print("HPROM_ITER_30STEPS_DONE_MARKER", flush=True)
