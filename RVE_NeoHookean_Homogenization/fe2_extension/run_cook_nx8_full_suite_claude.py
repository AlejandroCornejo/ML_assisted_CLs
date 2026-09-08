#!/usr/bin/env python3
"""Cook's membrane, full 8-model FE2 suite redone with this session's
complete optimization stack (the same parallel law wrappers verified on
the Cruciform demo), at the paper's own established resolution
(nx=ny=8, 128 elements, 384 Gauss points) and loading (Table 7's own
LINE_LOAD_MODULUS_FINAL/EDGE_LENGTH, 20 steps). run_cook_hprom_ann_
claude.py's own run_newton_fe2 is material-law-agnostic (same
MATERIAL_FUNCS-keyed pattern already used for every existing Cook
variant, e.g. run_cook_linear_hprom_claude.py's own (serial)
registration) -- this script just registers the PARALLEL versions of
Linear-HPROM/HPROM-ANN/D-HPROM-ANN under new keys, reusing run_
cruciform_fe2_claude.py's own factory functions verbatim (they wrap
*_law_parallel_claude.py modules that are pure functions of macro
strain, entirely agnostic to which macro geometry calls them).

FOM-FE2 (the slow, multi-hour reference) is deliberately NOT included
here -- run standalone afterward (run_cook_fom_nx8_full_claude.py), same
reasoning as the Cruciform split: keep its wall-clock number
uncontended, and per Table 7's own established convention, run it
WITHOUT line search (plain Newton already converges it there; line
search would triple that row's already-considerable cost) -- unlike
every other row here, which uses line search like the Cruciform suite
did.

Runs strictly sequentially, one model at a time, per this project's own
serial-only-benchmarking rule.
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

HERE = Path("/home/kratos/ML_assisted_CLs_clean/RVE_NeoHookean_Homogenization/fe2_extension")
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
os.chdir(str(HERE))

NX = NY = 8
N_STEPS = 20
N_WORKERS = 16

DHPROMANN_DIR = HERE / "maw_dynamic_reaction_force_dhpromann"
HPROMANN_DIR = HERE / "maw_dynamic_reaction_force_hpromann"

# Initialize ALL parallel pools' persistent executors before any Kratos/
# torch-touching import happens in this (parent) process -- same fork-
# after-CUDA-init discipline as every batch script this session.
import dhprom_ann_law_parallel_claude as dhprom_par_module  # noqa: E402
dhprom_par_module.ensure_persistent_executor(n_workers=N_WORKERS)
import hprom_ann_law_parallel_claude as hprom_par_module  # noqa: E402
hprom_par_module.ensure_persistent_executor(n_workers=N_WORKERS)
import linear_hprom_law_parallel_claude as linear_hprom_par_module  # noqa: E402
linear_hprom_par_module.ensure_persistent_executor(n_workers=N_WORKERS)

import run_cruciform_fe2_claude as rc  # noqa: E402 (reused ONLY for its generic factory functions)
import run_cook_hprom_ann_claude as cook_module  # noqa: E402


def run_and_report(which, t0_all):
    t0 = time.time()
    res = cook_module.run_newton_fe2(
        which, nx=NX, ny=NY, n_steps=N_STEPS, verbose=False, use_line_search=True, save_npz=True,
    )
    d = np.load(HERE / f"cook_results_{which}_claude.npz")
    print(f"[{which}] fully_converged={res['fully_converged']}, ever_diverged={res['ever_diverged']}, "
          f"tip_uy={res['tip_uy_range']}, n_material_calls={res['n_material_calls']}, "
          f"wall={res['wall_time']:.1f}s (elapsed this model {time.time() - t0:.1f}s, "
          f"total elapsed {time.time() - t0_all:.1f}s)", flush=True)
    return res


def main():
    t0_all = time.time()
    results = {}

    print(f"=== 4 PANN tiers (nx=ny={NX}, n_steps={N_STEPS}) ===", flush=True)
    for which in ("pann_certified", "pann_free", "pann_ickan", "pann_regression"):
        results[which] = run_and_report(which, t0_all)

    print("\n=== Linear-HPROM-FE2 (parallel+continuation) ===", flush=True)
    which = "linear_hprom_parallel_continuation"
    cook_module.MATERIAL_FUNCS[which] = rc.make_linear_hprom_parallel_continuation_material_func(n_workers=N_WORKERS)
    results[which] = run_and_report(which, t0_all)

    print("\n=== HPROM-ANN-FE2 (parallel+continuation) ===", flush=True)
    which = "hprom_ann_parallel_continuation"
    cook_module.MATERIAL_FUNCS[which] = rc.make_hprom_ann_parallel_continuation_material_func(str(HPROMANN_DIR), n_workers=N_WORKERS)
    results[which] = run_and_report(which, t0_all)

    print("\n=== D-HPROM-ANN-FE2 (parallel) ===", flush=True)
    which = "dhprom_ann_parallel"
    cook_module.MATERIAL_FUNCS[which] = rc.make_dhprom_ann_parallel_material_func(str(DHPROMANN_DIR), n_workers=N_WORKERS)
    results[which] = run_and_report(which, t0_all)

    print("\n=== BATCH SUMMARY (Cook's membrane, nx=ny=8, all 7 fast tiers; FOM-FE2 run separately) ===", flush=True)
    for which, res in results.items():
        print(f"  {which:32s}: fully_converged={res['fully_converged']}, ever_diverged={res['ever_diverged']}, "
              f"wall={res['wall_time']:.1f}s", flush=True)
    print(f"TOTAL elapsed: {time.time() - t0_all:.1f}s", flush=True)
    print("COOK_NX8_FULL_SUITE_DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
