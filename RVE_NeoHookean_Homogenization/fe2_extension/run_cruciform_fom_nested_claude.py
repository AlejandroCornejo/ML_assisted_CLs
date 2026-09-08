#!/usr/bin/env python3
"""The cruciform's true FOM-FE2 reference row (n_body=12, n_arm_len=8 --
same production resolution and loading as the just-completed PANN-tier
and D-HPROM-ANN batch in run_cruciform_overnight_claude.py, for a direct,
apples-to-apples comparison): no surrogate, no reduced-order model, the
full 990-element RVE solved from a cold start at every one of the
n_body=12 mesh's 2400 Gauss points, every macro Newton iteration.

Tangent is central finite differences (fom_nested_law_claude.py, h=1e-4),
matching exactly what already produced the paper's own reported Cook
FOM-FE2 row (76,992.2s) -- not a new, unvalidated analytic tangent for
the non-reduced FOM, which does not exist anywhere in this project and
would be reckless to derive and trust for the first time on an unattended
multi-hour-to-multi-day run. This keeps the FOM-FE2 methodology identical
across both geometries reported in the paper.

The only change from a plain reuse of Cook's own fom_nested_pk2_2d_vectorized
is wall-clock: fom_nested_law_parallel_claude.py runs the (already fully
independent) 7 solve_at_strain calls per Gauss point across a 16-way
process pool instead of one after another.

Import order in this file is deliberate and load-bearing, not cosmetic:
the persistent worker pool is created before this process imports
anything Kratos-related (KratosMultiphysics itself, fom_solver_rve, or
run_cruciform_fe2_claude, which imports both). Confirmed directly this
session: forking worker processes AFTER the parent has already used
Kratos (which spawns its own internal OpenMP thread pool on first
SOLVE, not just on import) leaves the first forked task hanging
indefinitely -- fork() only duplicates the calling thread, not whatever
worker threads Kratos already believes it has. Creating the pool first,
then reusing that same pool for every one of the run's macro Newton
calls (never recreating it per call), avoids the hazard for the whole
run, not just its first call.
"""
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

N_BODY, N_ARM_LEN = 12, 8
DELTA = 1.2
N_STEPS = 20
N_WORKERS = 16

# Pool created HERE, before any Kratos-touching import below.
from fom_nested_law_parallel_claude import ensure_persistent_executor  # noqa: E402
ensure_persistent_executor(n_workers=N_WORKERS)

# Only now is it safe to pull in Kratos.
from run_cruciform_fe2_claude import run_newton_fe2_cruciform  # noqa: E402


def main():
    print(f"=== FOM-FE2 (cruciform, n_body={N_BODY}, n_arm_len={N_ARM_LEN}, "
          f"delta={DELTA}, n_steps={N_STEPS}, {N_WORKERS}-way parallel FD tangent, "
          f"persistent pool) ===", flush=True)
    t0 = time.time()
    res = run_newton_fe2_cruciform(
        "fom_nested", n_body=N_BODY, n_arm_len=N_ARM_LEN, n_steps=N_STEPS,
        delta_x_final=DELTA, delta_y_final=DELTA, verbose=True, use_line_search=True, save_npz=True,
    )
    d = np.load(HERE / "cruciform_results_fom_nested_claude.npz")
    g12 = np.abs(d["e_gp"][:, 2])
    print(f"[fom_nested] fully_converged={res['fully_converged']}, ever_diverged={res['ever_diverged']}, "
          f"E11={res['e11_range']}, E22={res['e22_range']}, g12={res['g12_range']}, "
          f"median|g12|={np.median(g12):.4f}, frac>0.1={np.mean(g12 > 0.1):.3f}, "
          f"wall={res['wall_time']:.1f}s (elapsed {time.time() - t0:.1f}s)", flush=True)
    print("FOM_NESTED_DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
