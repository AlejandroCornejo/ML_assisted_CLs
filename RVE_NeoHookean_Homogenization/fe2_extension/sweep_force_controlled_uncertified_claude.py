#!/usr/bin/env python3
"""Force-controlled Cruciform sweep for Regression/Free (and, for
contrast, ICNN/ICKAN at the same force levels): ramps total_force_final
well past the reference level (1.2551e9 N, the ICNN reaction force at
the standard delta=1.2 displacement-controlled protocol, already
verified self-consistent -- see run_cruciform_fe2_force_controlled_claude.py's
own __main__ check) to look for a limit point (force-displacement curve
turning over) -- the one mechanism not yet ruled out for why Cook's
membrane makes these two tiers stall while nothing tried on the
Cruciform geometry under displacement control has."""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path("/home/kratos/ML_assisted_CLs_clean/RVE_NeoHookean_Homogenization/fe2_extension")
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_cruciform_fe2_force_controlled_claude import run_newton_fe2_cruciform_force  # noqa: E402

N_STEPS = 20
REF_FORCE = 1.2551e9
MULTIPLES = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0]
UNCERTIFIED = ("pann_regression", "pann_free")
CERTIFIED = ("pann_certified", "pann_ickan")


def run_one(which, total_force_final):
    t0 = time.time()
    res = run_newton_fe2_cruciform_force(
        which, n_body=6, n_arm_len=4, n_steps=N_STEPS, total_force_final=total_force_final,
        verbose=False, use_line_search=True, save_npz=False,
    )
    dt = time.time() - t0
    last = res["step_log"][-1]
    print(f"  [{which}] F={total_force_final:.3e}: fully_converged={res['fully_converged']}, "
          f"ever_diverged={res['ever_diverged']}, iters={[s['iters'] for s in res['step_log']]}, "
          f"status={[s['status'][0] for s in res['step_log']]}, "
          f"final tip_ux={last['tip_ux_px_mean']:.4f}, tip_uy={last['tip_uy_py_mean']:.4f}, wall={dt:.2f}s",
          flush=True)
    return res


def main():
    print("=== Force-controlled sweep: uncertified tiers ===", flush=True)
    trouble = []
    for which in UNCERTIFIED:
        print(f"-- {which} --", flush=True)
        for mult in MULTIPLES:
            try:
                res = run_one(which, REF_FORCE * mult)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{which}] F={REF_FORCE * mult:.3e}: CRASHED (hard, invalid state mid-line-search): "
                      f"{type(exc).__name__}: {exc}", flush=True)
                trouble.append((which, mult))
                print(f"    ^^^ TROUBLE at {mult}x reference force", flush=True)
                continue
            if res["ever_diverged"] or not res["fully_converged"]:
                trouble.append((which, mult))
                print(f"    ^^^ TROUBLE at {mult}x reference force", flush=True)

    print(f"\n=== trouble found at: {trouble} ===", flush=True)

    # Contrast at the two most meaningful levels: 1x (where Free already
    # stalls) and the LOWEST multiplier where anything went wrong (where
    # Regression first diverges) -- not the higher multipliers, which are
    # likely just "too extreme for anything" (confirmed separately: 10x
    # already pushes even ICNN into an invalid deformation state) and
    # would not give a meaningful contrast either way.
    contrast_mults = sorted({1.0} | {m for _, m in trouble})[:2]
    print(f"\n=== Contrast: certified tiers at multiplier(s) {contrast_mults} ===", flush=True)
    for mult in contrast_mults:
        for cwhich in CERTIFIED:
            try:
                run_one(cwhich, REF_FORCE * mult)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{cwhich}] F={REF_FORCE * mult:.3e}: CRASHED: {type(exc).__name__}: {exc}", flush=True)

    print("FORCE_SWEEP_DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
