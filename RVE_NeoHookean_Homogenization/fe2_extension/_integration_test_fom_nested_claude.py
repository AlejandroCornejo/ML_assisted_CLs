#!/usr/bin/env python3
"""Tiny, cheap end-to-end integration test before trusting the real
n_body=12 run: does the persistent-pool-based fom_nested material law
still work correctly once the PARENT process has also done its own
macro-level Kratos work (building the cruciform mesh, running the macro
Newton strategy)? The earlier validation only tested calling the pool
directly, with no macro-level Kratos activity in the parent at all --
this is the actually-untested, actually-relevant scenario.

Tiny mesh (n_body=6, n_arm_len=2, 88 elements -> 264 Gauss points),
n_steps=2, small delta -- just needs to prove the chain completes
without hanging and gives a sane, converged result.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fom_nested_law_parallel_claude import ensure_persistent_executor  # noqa: E402
ensure_persistent_executor(n_workers=4)

from run_cruciform_fe2_claude import run_newton_fe2_cruciform  # noqa: E402

print("=== integration test: tiny fom_nested cruciform run ===", flush=True)
t0 = time.time()
res = run_newton_fe2_cruciform(
    "fom_nested", n_body=6, n_arm_len=2, n_steps=2,
    delta_x_final=0.05, delta_y_final=0.05, verbose=True, use_line_search=True, save_npz=False,
)
print(f"[integration] fully_converged={res['fully_converged']}, ever_diverged={res['ever_diverged']}, "
      f"wall={res['wall_time']:.1f}s (elapsed {time.time() - t0:.1f}s)", flush=True)
print("INTEGRATION_TEST_DONE_MARKER", flush=True)
