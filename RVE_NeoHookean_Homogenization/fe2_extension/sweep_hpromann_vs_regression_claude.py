#!/usr/bin/env python3
"""Systematic sweep looking specifically for states within the RVE's own
trained box where HPROM-ANN converges cleanly but pure Regression does
NOT -- the opposite of every comparison so far this session, where
Regression has (uncomfortably) come out looking more robust. Covers
regions never stress-tested this session: the compression corner
(training data DOES include a compression branch, E22->-0.1, per the
paper's own training-trajectory description), strongly asymmetric/
uniaxial-like states (matching the held-out test trajectory's own
character: E11 large, E22~0, not equibiaxial), and near-max-shear at
modest (not extreme) tension -- deliberately NOT re-testing the already-
known (1.9,1.9,0.1) corner where the result (HPROM-ANN fails, Regression
doesn't) is already established.

Reuses run_square_panel_trajectory_claude.py's run_trajectory verbatim
(single-segment "keyframes" from the origin to each target state) --
zero new Newton-loop code, same analytic-seeding + noise-floor-aware
convergence fix already verified this session."""
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
HPROMANN_DIR = HERE / "maw_dynamic_reaction_force_hpromann"

import hprom_ann_law_parallel_claude as hprom_par_module  # noqa: E402
hprom_par_module.ensure_persistent_executor(n_workers=N_WORKERS)

from run_square_panel_trajectory_claude import run_trajectory  # noqa: E402
import run_cruciform_fe2_claude as rc  # noqa: E402

which_hprom_ann = "hprom_ann_parallel_continuation"
rc.MATERIAL_FUNCS[which_hprom_ann] = rc.make_hprom_ann_parallel_continuation_material_func(
    str(HPROMANN_DIR), n_workers=N_WORKERS)

MODELS = ["pann_regression", which_hprom_ann]

# (label, (E11, E22, gamma12)) -- deliberately NOT re-testing the known
# (1.9,1.9,0.1) corner. Covers: compression-compression, compression+shear,
# compression-tension mixed, strongly asymmetric/uniaxial tension (matching
# the held-out test trajectory's own character), asymmetric tension+shear,
# and near-max-shear at modest/near-zero tension.
STATES = [
    ("compression-compression", (-0.10, -0.10, 0.00)),
    ("compression-compression-shear", (-0.08, -0.08, 0.09)),
    ("compression-tension mixed", (-0.10, 1.50, 0.05)),
    ("asymmetric uniaxial tension (test-traj-like)", (1.80, 0.00, 0.00)),
    ("asymmetric uniaxial tension + shear", (1.80, 0.00, 0.09)),
    ("near-max shear, modest tension", (0.30, 0.30, 0.095)),
    ("near-max shear, near-zero tension", (0.05, 0.05, 0.095)),
    ("near-max shear, asymmetric modest tension", (0.30, -0.05, 0.095)),
]


def main():
    results = {}
    for label, target in STATES:
        for which in MODELS:
            key = (label, which)
            print(f"\n=== {label}  target=({target[0]:+.2f},{target[1]:+.2f},{target[2]:+.3f})  {which} ===",
                  flush=True)
            t0 = time.time()
            try:
                res = run_trajectory(which, verbose=False, keyframes=[(0.0, 0.0, 0.0), target],
                                      steps_per_segment=15)
                print(f"[{label}|{which}] fully_converged={res['fully_converged']}  "
                      f"ever_diverged={res['ever_diverged']}  wall={res['wall_time']:.1f}s  "
                      f"({time.time() - t0:.1f}s elapsed)", flush=True)
                results[key] = res["fully_converged"] and not res["ever_diverged"]
            except Exception as exc:  # noqa: BLE001
                print(f"[{label}|{which}] CRASHED after {time.time() - t0:.1f}s: {type(exc).__name__}: {exc}",
                      flush=True)
                results[key] = False

    print("\n=== SUMMARY: OK means fully converged, never diverged, never crashed ===", flush=True)
    print(f"  {'state':45s}  {'regression':>12s}  {'hprom_ann':>12s}  interesting?", flush=True)
    for label, target in STATES:
        r_ok = results[(label, "pann_regression")]
        h_ok = results[(label, which_hprom_ann)]
        interesting = "<<< YES (hprom ok, regression fails)" if (h_ok and not r_ok) else ""
        print(f"  {label:45s}  {'OK' if r_ok else 'FAIL':>12s}  {'OK' if h_ok else 'FAIL':>12s}  {interesting}",
              flush=True)
    print("SWEEP_HPROMANN_VS_REGRESSION_DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
