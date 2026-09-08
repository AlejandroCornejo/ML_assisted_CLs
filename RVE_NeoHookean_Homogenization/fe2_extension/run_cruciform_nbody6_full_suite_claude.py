#!/usr/bin/env python3
"""Full n_body=6 cruciform FE2 suite, redone from scratch with this
session's complete optimization stack (all parallel law wrappers), for
the paper's actual Cruciform results: the 4 PANN tiers, then HPROM-FE2
(Linear-HPROM, parallel+continuation), HPROM-ANN-FE2 (parallel+
continuation), D-HPROM-ANN-FE2 (parallel). FOM-FE2 (the slow, multi-hour
reference) is deliberately NOT included here -- it's run standalone,
afterward, with nothing else running concurrently, so its wall-clock
number for the paper isn't contended.

Saves each result under its established canonical (non-suffixed) npz
name, overwriting any stale prior copy -- this run is the source of
truth going forward. Everything here runs strictly sequentially, never
concurrently, per this project's own serial-only-benchmarking rule.
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

N_BODY, N_ARM_LEN = 6, 4
DELTA = 1.2
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

import run_cruciform_fe2_claude as rc  # noqa: E402


def run_and_report(which, t0_all):
    t0 = time.time()
    res = rc.run_newton_fe2_cruciform(
        which, n_body=N_BODY, n_arm_len=N_ARM_LEN, n_steps=N_STEPS,
        delta_x_final=DELTA, delta_y_final=DELTA, verbose=False, use_line_search=True, save_npz=True,
    )
    d = np.load(HERE / f"cruciform_results_{which}_claude.npz")
    g12 = np.abs(d["e_gp"][:, 2])
    print(f"[{which}] fully_converged={res['fully_converged']}, ever_diverged={res['ever_diverged']}, "
          f"E11={res['e11_range']}, E22={res['e22_range']}, g12={res['g12_range']}, "
          f"median|g12|={np.median(g12):.4f}, n_material_calls={res['n_material_calls']}, "
          f"reaction_px={res['reaction_px']:.4e}, "
          f"wall={res['wall_time']:.1f}s (elapsed this model {time.time() - t0:.1f}s, "
          f"total elapsed {time.time() - t0_all:.1f}s)", flush=True)
    return res


def main():
    t0_all = time.time()
    results = {}

    print(f"=== 4 PANN tiers (n_body={N_BODY}, n_arm_len={N_ARM_LEN}, delta={DELTA}, n_steps={N_STEPS}) ===",
          flush=True)
    for which in ("pann_certified", "pann_free", "pann_ickan", "pann_regression"):
        results[which] = run_and_report(which, t0_all)

    print("\n=== HPROM-FE2 (Linear-HPROM, parallel+continuation) ===", flush=True)
    which = "linear_hprom_parallel_continuation"
    rc.MATERIAL_FUNCS[which] = rc.make_linear_hprom_parallel_continuation_material_func(n_workers=N_WORKERS)
    results[which] = run_and_report(which, t0_all)

    print("\n=== HPROM-ANN-FE2 (parallel+continuation) ===", flush=True)
    which = "hprom_ann_parallel_continuation"
    rc.MATERIAL_FUNCS[which] = rc.make_hprom_ann_parallel_continuation_material_func(str(HPROMANN_DIR), n_workers=N_WORKERS)
    results[which] = run_and_report(which, t0_all)

    print("\n=== D-HPROM-ANN-FE2 (parallel) ===", flush=True)
    which = "dhprom_ann_parallel"
    rc.MATERIAL_FUNCS[which] = rc.make_dhprom_ann_parallel_material_func(str(DHPROMANN_DIR), n_workers=N_WORKERS)
    results[which] = run_and_report(which, t0_all)

    print("\n=== BATCH SUMMARY (n_body=6, all 6 fast tiers; FOM-FE2 run separately) ===", flush=True)
    for which, res in results.items():
        print(f"  {which:32s}: fully_converged={res['fully_converged']}, ever_diverged={res['ever_diverged']}, "
              f"wall={res['wall_time']:.1f}s", flush=True)
    print(f"TOTAL elapsed: {time.time() - t0_all:.1f}s", flush=True)
    print("NBODY6_FULL_SUITE_DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
