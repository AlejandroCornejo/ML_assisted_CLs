#!/usr/bin/env python3
"""HPROM-ANN, force-controlled Cruciform, F=1.0e10 N (10 GN), 60 steps
instead of 20 -- tests whether finer load increments avoid the "invalid
Green-Lagrange strain" crash seen at 20 steps. Unlike D-HPROM-ANN (which
still crashed at 60 steps, via a frozen-displacement stall from ~33% load
onward rather than a discretization artifact), HPROM-ANN's iterative
correction carries continuation/warm-start state, so its behavior under
finer stepping is less predictable a priori."""
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

N_STEPS = 60
N_WORKERS = 16
TOTAL_FORCE_FINAL = 1.0e10

HPROMANN_DIR = HERE / "maw_dynamic_reaction_force_hpromann"

import hprom_ann_law_parallel_claude as hprom_par_module  # noqa: E402
hprom_par_module.ensure_persistent_executor(n_workers=N_WORKERS)

import run_cruciform_fe2_claude as rc  # noqa: E402
from run_cruciform_fe2_force_controlled_claude import run_newton_fe2_cruciform_force  # noqa: E402

which = "hprom_ann_parallel_continuation"
rc.MATERIAL_FUNCS[which] = rc.make_hprom_ann_parallel_continuation_material_func(str(HPROMANN_DIR), n_workers=N_WORKERS)


def main():
    print(f"=== {which}, force-controlled, F={TOTAL_FORCE_FINAL:.3e} N (10 GN), n_steps={N_STEPS} ===", flush=True)
    t0 = time.time()
    try:
        res = run_newton_fe2_cruciform_force(
            which, n_body=6, n_arm_len=4, n_steps=N_STEPS, total_force_final=TOTAL_FORCE_FINAL,
            verbose=True, use_line_search=True, save_npz=True,
        )
        last = res["step_log"][-1]
        print(f"\n[{which}] fully_converged={res['fully_converged']}, ever_diverged={res['ever_diverged']}, "
              f"final tip_ux={last['tip_ux_px_mean']:.4f}, tip_uy={last['tip_uy_py_mean']:.4f}, "
              f"wall={res['wall_time']:.1f}s (elapsed {time.time() - t0:.1f}s)", flush=True)
        print("HPROMANN_60STEP_SUCCESS_MARKER", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[{which}] CRASHED after {time.time() - t0:.1f}s: {type(exc).__name__}: {exc}", flush=True)
        print("HPROMANN_60STEP_CRASHED_MARKER", flush=True)


if __name__ == "__main__":
    main()
