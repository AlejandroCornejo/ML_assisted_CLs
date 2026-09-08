#!/usr/bin/env python3
"""FOM-FE2 (fom_nested_consistent_law_claude.py: reaction-force stress +
analytic implicit-function-theorem tangent), Cook's membrane, nx=ny=8,
full 20-step ramp -- the true, non-reduced multiscale reference for
Table 7, redone with this session's analytic tangent + 16-worker
parallelization (replacing the old central-finite-difference-tangent
version this row previously used). Run standalone, nothing else
concurrent, so the wall-clock number stays clean.

Per Table 7's own established convention for this specific row (unlike
every other row): run WITHOUT line search. Plain Newton already
converges the true FOM here; line search would triple this row's own
already-considerable cost for no benefit (the converged fixed point is
determined by the true stress law at zero residual, not by how exactly
the tangent gets there)."""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
import time
from pathlib import Path

HERE = Path("/home/kratos/ML_assisted_CLs_clean/RVE_NeoHookean_Homogenization/fe2_extension")
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
os.chdir(str(HERE))

NX = NY = 8
N_STEPS = 20
N_WORKERS = 16

import fom_nested_consistent_law_parallel_claude as fom_par_module  # noqa: E402
fom_par_module.ensure_persistent_executor(n_workers=N_WORKERS)

import run_cruciform_fe2_claude as rc  # noqa: E402 (reused ONLY for its generic factory function)
import run_cook_hprom_ann_claude as cook_module  # noqa: E402


def main():
    which = "fom_nested_consistent_parallel"
    cook_module.MATERIAL_FUNCS[which] = rc.make_fom_nested_consistent_parallel_material_func(n_workers=N_WORKERS)

    print(f"=== FOM-FE2 (consistent, parallel), Cook's membrane, nx=ny={NX}, n_steps={N_STEPS}, "
          f"use_line_search=False ===", flush=True)
    t0 = time.time()
    res = cook_module.run_newton_fe2(
        which, nx=NX, ny=NY, n_steps=N_STEPS, verbose=True, use_line_search=False, save_npz=True,
    )
    print(f"[{which}] fully_converged={res['fully_converged']}, ever_diverged={res['ever_diverged']}, "
          f"tip_uy={res['tip_uy_range']}, n_material_calls={res['n_material_calls']}, "
          f"wall={res['wall_time']:.1f}s (elapsed {time.time() - t0:.1f}s)", flush=True)
    print("FOM_COOK_NX8_FULL_DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
