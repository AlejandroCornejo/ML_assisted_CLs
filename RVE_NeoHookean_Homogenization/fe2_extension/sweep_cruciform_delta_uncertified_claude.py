#!/usr/bin/env python3
"""Exploratory (not paper-bound): does pushing the Cruciform's own
equibiaxial pull harder (larger delta, same direction as the
established delta=1.2 protocol) expose the two uncertified PANN tiers
(Regression, Free hyperelastic) to convergence trouble, the way Cook's
membrane already does at its own (very different) loading? Sweeps delta
for regression/free AND, at any delta where either struggles, also runs
the two certified tiers (ICNN, ICKAN) at the identical delta as a direct
contrast -- same geometry, same load, does the polyconvexity certificate
actually buy robustness here. All 4 models are cheap (sub-few-second
full 20-step runs), so this whole sweep is cheap too. Does not touch or
overwrite any of the established delta=1.2 result files (different
MATERIAL_FUNCS keys, save_npz=False)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path("/home/kratos/ML_assisted_CLs_clean/RVE_NeoHookean_Homogenization/fe2_extension")
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_cruciform_fe2_claude as rc  # noqa: E402

N_BODY, N_ARM_LEN = 6, 4
N_STEPS = 20
DELTAS = [1.2, 1.8, 2.4, 3.0, 3.6, 4.2, 4.8]
UNCERTIFIED = ("pann_regression", "pann_free")
CERTIFIED = ("pann_certified", "pann_ickan")


def run_one(which, delta):
    t0 = time.time()
    res = rc.run_newton_fe2_cruciform(
        which, n_body=N_BODY, n_arm_len=N_ARM_LEN, n_steps=N_STEPS,
        delta_x_final=delta, delta_y_final=delta, verbose=False, use_line_search=True, save_npz=False,
    )
    dt = time.time() - t0
    print(f"  [{which}] delta={delta:.1f}: fully_converged={res['fully_converged']}, "
          f"ever_diverged={res['ever_diverged']}, iters={[s['iters'] for s in res['step_log']]}, "
          f"status={[s['status'][0] for s in res['step_log']]}, "
          f"g12_range={res['g12_range']}, wall={dt:.2f}s", flush=True)
    return res


def main():
    print("=== Sweep 1: uncertified tiers (Regression, Free), equibiaxial delta scan ===", flush=True)
    trouble_at = {}
    for which in UNCERTIFIED:
        print(f"-- {which} --", flush=True)
        for delta in DELTAS:
            res = run_one(which, delta)
            if res["ever_diverged"] or not res["fully_converged"]:
                trouble_at.setdefault(which, delta)

    print("\n=== Sweep 2: certified tiers (ICNN, ICKAN), SAME deltas for direct contrast ===", flush=True)
    for which in CERTIFIED:
        print(f"-- {which} --", flush=True)
        for delta in DELTAS:
            run_one(which, delta)

    print("\n=== SUMMARY: first delta where each uncertified tier showed trouble ===", flush=True)
    for which in UNCERTIFIED:
        print(f"  {which}: {trouble_at.get(which, 'no trouble found in this sweep')}", flush=True)
    print("SWEEP_DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
