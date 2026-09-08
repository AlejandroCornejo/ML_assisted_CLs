#!/usr/bin/env python3
"""Follow-up to sweep_cruciform_delta_uncertified_claude.py: scaling the
SAME equibiaxial direction up to 4x found no trouble for Regression/Free,
so direction (not magnitude) is the more likely lever, mirroring how
Cook's own training-coverage gap is a specific combination the training
trajectories never visit together, not just "large strain." Holds
delta_x fixed at 2.4 (already confirmed safe in the equibiaxial case)
and sweeps delta_y across a wide range, including asymmetric tension,
pure uniaxial (delta_y=0), and mixed tension/compression (delta_y<0) --
pushing E11 away from E22 rather than scaling both together."""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path("/home/kratos/ML_assisted_CLs_clean/RVE_NeoHookean_Homogenization/fe2_extension")
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_cruciform_fe2_claude as rc  # noqa: E402

N_BODY, N_ARM_LEN = 6, 4
N_STEPS = 20
DELTA_X = 2.4
DELTA_YS = [2.4, 1.8, 1.2, 0.6, 0.0, -0.6, -1.2, -1.8, -2.4]
UNCERTIFIED = ("pann_regression", "pann_free")
CERTIFIED = ("pann_certified", "pann_ickan")


def run_one(which, delta_x, delta_y):
    t0 = time.time()
    res = rc.run_newton_fe2_cruciform(
        which, n_body=N_BODY, n_arm_len=N_ARM_LEN, n_steps=N_STEPS,
        delta_x_final=delta_x, delta_y_final=delta_y, verbose=False, use_line_search=True, save_npz=False,
    )
    dt = time.time() - t0
    print(f"  [{which}] delta_x={delta_x:.1f} delta_y={delta_y:.1f}: fully_converged={res['fully_converged']}, "
          f"ever_diverged={res['ever_diverged']}, iters={[s['iters'] for s in res['step_log']]}, "
          f"status={[s['status'][0] for s in res['step_log']]}, "
          f"E11={res['e11_range']}, E22={res['e22_range']}, g12={res['g12_range']}, wall={dt:.2f}s", flush=True)
    return res


def main():
    print("=== Sweep: uncertified tiers, delta_x=2.4 fixed, delta_y varying (asymmetric/uniaxial/mixed) ===",
          flush=True)
    trouble = []
    for which in UNCERTIFIED:
        print(f"-- {which} --", flush=True)
        for delta_y in DELTA_YS:
            res = run_one(which, DELTA_X, delta_y)
            if res["ever_diverged"] or not res["fully_converged"]:
                trouble.append((which, delta_y))
                print(f"    ^^^ TROUBLE FOUND: {which} at delta_y={delta_y}", flush=True)

    print(f"\n=== trouble found at: {trouble} ===", flush=True)

    if trouble:
        print("\n=== Contrast: certified tiers at the SAME (delta_x, delta_y) that caused trouble ===", flush=True)
        seen = set()
        for which, delta_y in trouble:
            if delta_y in seen:
                continue
            seen.add(delta_y)
            for cwhich in CERTIFIED:
                run_one(cwhich, DELTA_X, delta_y)

    print("ASYM_SWEEP_DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
