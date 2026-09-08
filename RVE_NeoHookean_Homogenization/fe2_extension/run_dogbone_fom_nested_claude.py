#!/usr/bin/env python3
"""FOM-FE^2 for the dogbone at delta_x_final=1.5 -- the true, non-reduced
990-element RVE nested at every one of the 606 macro Gauss points, the
actual multiscale ground truth the other six rows approximate. Expensive:
Cruciform's own analogous run (600 Gauss points) took 40,093.7s (11.1h);
this problem has 606, so expect a comparable order of magnitude. Uses
MATERIAL_FUNCS["fom_nested"], which already runs its own 16-worker
process pool internally (fom_nested_law_parallel_claude.py) -- the
thread-limiting env vars below are essential here, not optional, since
each of the 16 worker processes would otherwise also spawn its own
multi-threaded BLAS, causing severe oversubscription (matching this
project's own established practice for every other 16-worker sweep this
session)."""
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
    print("=== fom_nested (delta_x_final=1.5) ===", flush=True)
    res = run_newton_fe2_dogbone("fom_nested", delta_x_final=1.5, verbose=True, save_npz=True)
    print(f"\n[fom_nested] fully_converged={res['fully_converged']}  ever_diverged={res['ever_diverged']}  "
          f"E11={res['e11_range']}  E22={res['e22_range']}  g12={res['g12_range']}  "
          f"wall={res['wall_time']:.1f}s", flush=True)
    print("DOGBONE_FOM_NESTED_DONE_MARKER", flush=True)
