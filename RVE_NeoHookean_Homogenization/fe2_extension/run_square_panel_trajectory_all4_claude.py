#!/usr/bin/env python3
"""Run the purpose-built square-panel trajectory (large equibiaxial tension
+ full trained shear sweep, cyclic back to origin -- run_square_panel_
trajectory_claude.py) across Regression, Free, D-HPROM-ANN, and HPROM-ANN.
Registers the parallel D-HPROM-ANN/HPROM-ANN material funcs BEFORE importing
anything that touches Kratos/torch, matching this session's own established
fork-after-init discipline."""
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

N_WORKERS = 16
DHPROMANN_DIR = HERE / "maw_dynamic_reaction_force_dhpromann"
HPROMANN_DIR = HERE / "maw_dynamic_reaction_force_hpromann"

import dhprom_ann_law_parallel_claude as dhprom_par_module  # noqa: E402
dhprom_par_module.ensure_persistent_executor(n_workers=N_WORKERS)
import hprom_ann_law_parallel_claude as hprom_par_module  # noqa: E402
hprom_par_module.ensure_persistent_executor(n_workers=N_WORKERS)

from run_square_panel_trajectory_claude import run_trajectory  # noqa: E402
import run_cruciform_fe2_claude as rc  # noqa: E402

which_dhprom = "dhprom_ann_parallel"
which_hprom_ann = "hprom_ann_parallel_continuation"
rc.MATERIAL_FUNCS[which_dhprom] = rc.make_dhprom_ann_parallel_material_func(str(DHPROMANN_DIR), n_workers=N_WORKERS)
rc.MATERIAL_FUNCS[which_hprom_ann] = rc.make_hprom_ann_parallel_continuation_material_func(str(HPROMANN_DIR), n_workers=N_WORKERS)

MODELS = ["pann_regression", "pann_free", which_dhprom, which_hprom_ann]


def main():
    results = {}
    for which in MODELS:
        print(f"\n=== {which}, square panel, purpose-built trajectory ===", flush=True)
        t0 = time.time()
        try:
            res = run_trajectory(which, verbose=False)
            print(f"[{which}] fully_converged={res['fully_converged']}  ever_diverged={res['ever_diverged']}  "
                  f"wall={res['wall_time']:.1f}s  ({time.time() - t0:.1f}s elapsed)", flush=True)
            n_stalled = sum(1 for s in res["step_log"] if s["status"] == "stalled")
            n_diverged = sum(1 for s in res["step_log"] if s["status"] == "diverged")
            if n_stalled or n_diverged:
                print(f"[{which}] {n_stalled} stalled steps, {n_diverged} diverged steps -- "
                      f"first trouble at step {next(s['step'] for s in res['step_log'] if s['status'] != 'converged')}",
                      flush=True)
            results[which] = res
        except Exception as exc:  # noqa: BLE001
            print(f"[{which}] CRASHED after {time.time() - t0:.1f}s: {type(exc).__name__}: {exc}", flush=True)
            results[which] = None

    print("\n=== SUMMARY ===", flush=True)
    for which in MODELS:
        r = results[which]
        if r is None:
            print(f"  {which:32s}: CRASHED", flush=True)
        else:
            print(f"  {which:32s}: fully_converged={r['fully_converged']}, ever_diverged={r['ever_diverged']}, "
                  f"wall={r['wall_time']:.1f}s", flush=True)
    print("TRAJECTORY_ALL4_DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
