#!/usr/bin/env python3
"""FOM-FE^2 reference for the final dogbone setup, using the warm-started
consistent law -- the same converged answer as the cold law, ~90x cheaper
than what was originally launched for this problem.

Where that factor comes from (both fixes are independent and multiply):

1. The CONSISTENT law (analytic implicit-function-theorem tangent, 1 solve
   per Gauss point) instead of fom_nested_law_claude's central-FD tangent
   (7 solves per Gauss point). The FD variant was used by mistake in the
   first dogbone FOM launch; this project's own published Cook/cruciform
   FOM-FE^2 rows already use the consistent one.

2. WARM-STARTING per Gauss point. Substeps are allocated proportional to
   the strain path's length, so a from-zero ramp costs ~200*|E| substeps
   (about 55 at this problem's full load) while continuing from the same
   point's previous converged state costs 1. Free of approximation because
   the RVE is Neo-Hookean hyperelastic with no history, so its converged
   state is a pure function of E -- verified to 1e-15 relative on both S
   and CC (_verify_fom_warm_vs_cold_claude.py, _verify_fom_warm_parallel_
   claude.py, the latter also confirming the mechanism actually engages,
   which a values-only check cannot see).

Also silences the solver's per-substep chatter at the file-descriptor level:
the first attempt at this run wrote a 4.2 GB log of it.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

N_WORKERS = 16

# Pool BEFORE the parent touches Kratos (fork-after-threading discipline),
# and persistence is what keeps the run cheap across macro iterations.
import fom_nested_consistent_warm_law_parallel_claude as warm_par  # noqa: E402
warm_par.ensure_persistent_executor(n_workers=N_WORKERS)

from run_dogbone_force_fe2_claude import run_newton_fe2_dogbone_force  # noqa: E402
from run_cruciform_fe2_claude import MATERIAL_FUNCS  # noqa: E402

GEOM = dict(W_gauge=4.0, W_grip=4.25, R=2.0)
FORCE = 1.0e9
WHICH = "fom_nested_consistent_warm_parallel"


def _material_func(e_voigt, young=None, poisson=None):
    return warm_par.fom_nested_consistent_warm_pk2_2d_vectorized_parallel(
        e_voigt, n_workers=N_WORKERS, verbose=True, silence=True)


MATERIAL_FUNCS[WHICH] = _material_func

if __name__ == "__main__":
    warm_par.reset_state()  # independent macro problem: start with no history
    print(f"=== {WHICH} (F={FORCE:.2e}, {GEOM}, improved mesh) ===", flush=True)
    t0 = time.perf_counter()
    # No line search for the FOM row, matching this project's own published
    # Cook FOM-FE^2 row ("run without line search (plain Newton already
    # converges it, and line search would triple that row's own cost)").
    # _line_search_alpha costs 2 extra material-law evaluations per Newton
    # iteration, and for THIS law one evaluation is a full 1656-Gauss-point
    # batch of nonlinear RVE solves -- measured directly here as 9 law calls
    # for a 3-iteration step before this was switched off. The surrogate rows
    # all converge on this problem in 3 iterations with clean ~4-orders-per-
    # iteration residual drops, i.e. alpha=1 behaviour, so the search was
    # buying nothing on it.
    res = run_newton_fe2_dogbone_force(
        WHICH, total_force_final=FORCE, verbose=True, save_npz=True,
        use_line_search=False, **GEOM)
    print(f"\n[{WHICH}] fully_converged={res['fully_converged']}  "
          f"ever_diverged={res['ever_diverged']}  E11={res['e11_range']}  "
          f"E22={res['e22_range']}  g12={res['g12_range']}  "
          f"wall={res['wall_time']:.1f}s ({(time.perf_counter() - t0) / 60.0:.1f} min total)", flush=True)
    print("DOGBONE_FOM_WARM_DONE_MARKER", flush=True)
