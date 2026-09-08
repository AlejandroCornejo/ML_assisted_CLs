#!/usr/bin/env python3
"""FOM-FE2 (fom_nested_consistent_law_claude.py: reaction-force stress +
analytic implicit-function-theorem tangent), n_body=6, n_arm_len=4, full
20-step ramp -- the true, non-reduced multiscale reference for the
Cruciform table. Run standalone, with nothing else running concurrently,
so its wall-clock number is clean/uncontended (this is the ground-truth
row every speedup number in the table is measured against). Step 1 alone
already measured at 514.3s at this mesh; a full 20-step run is expected
to take on the order of hours."""
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

N_BODY, N_ARM_LEN = 6, 4
DELTA = 1.2
N_STEPS = 20
N_WORKERS = 16

import fom_nested_consistent_law_parallel_claude as fom_par_module  # noqa: E402
fom_par_module.ensure_persistent_executor(n_workers=N_WORKERS)

import run_cruciform_fe2_claude as rc  # noqa: E402


def main():
    which = "fom_nested_consistent_parallel"
    rc.MATERIAL_FUNCS[which] = rc.make_fom_nested_consistent_parallel_material_func(n_workers=N_WORKERS)

    print(f"=== FOM-FE2 (consistent, parallel), n_body={N_BODY}, n_arm_len={N_ARM_LEN}, "
          f"delta={DELTA}, n_steps={N_STEPS} ===", flush=True)
    t0 = time.time()
    res = rc.run_newton_fe2_cruciform(
        which, n_body=N_BODY, n_arm_len=N_ARM_LEN, n_steps=N_STEPS,
        delta_x_final=DELTA, delta_y_final=DELTA, verbose=True, use_line_search=True, save_npz=True,
    )
    import numpy as np
    d = np.load(HERE / f"cruciform_results_{which}_claude.npz")
    g12 = np.abs(d["e_gp"][:, 2])
    print(f"[{which}] fully_converged={res['fully_converged']}, ever_diverged={res['ever_diverged']}, "
          f"E11={res['e11_range']}, E22={res['e22_range']}, g12={res['g12_range']}, "
          f"median|g12|={np.median(g12):.4f}, n_material_calls={res['n_material_calls']}, "
          f"reaction_px={res['reaction_px']:.4e}, "
          f"wall={res['wall_time']:.1f}s (elapsed {time.time() - t0:.1f}s)", flush=True)
    print("FOM_NBODY6_FULL_DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
