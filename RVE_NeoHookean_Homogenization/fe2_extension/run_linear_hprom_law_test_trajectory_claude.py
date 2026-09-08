#!/usr/bin/env python3
"""Table-6-style test of LinearHpromIterativeLawFloat64 (linear_hprom_
iterative_law_float64_claude.py) on the held-out test trajectory -- calls
ONLY the law's own evaluate() per step (native, online reaction-force
stress, q_prev-threaded continuation across steps) -- NO post-hoc
correction, matching exactly what Cook's own FE2 driver will do per macro
Gauss point.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
CORE_DIR = REPO_ROOT / "core"
PANN_DATA = REPO_ROOT / "pann" / "data" / "alltraj_stage10_direct_energy.npz"

if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

from fom_solver_rve import BuildDynamicSegmentSteps, REFERENCE_STEPS_FOR_UNIT_AMPLITUDE, MIN_STEPS_PER_SEGMENT  # noqa: E402
from stage6_test_hprom import generate_safe_test_path  # noqa: E402
from linear_hprom_iterative_law_float64_claude import LinearHpromIterativeLawFloat64  # noqa: E402


def relative_l2(pred, ref):
    return float(np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), 1.0e-30))


def main() -> None:
    bundle_path = REPO_ROOT / "trajectories" / "stage_0_trajectory" / "stage_0_trajectories.npz"
    data = np.load(bundle_path, allow_pickle=True)
    rel6 = list(data["relative_boundary"])
    emax = float(np.ravel(data["emax"])[0]) if "emax" in data else float(np.ravel(data["reference_amplitude"])[0])
    domain_type = str(data["domain_type"][0]) if "domain_type" in data else "box"
    _, waypoints = generate_safe_test_path(emax, rel6, domain_type)
    E_wp = np.array(waypoints, dtype=float)
    n_seg = len(E_wp) - 1
    seg_steps, _ = BuildDynamicSegmentSteps(
        E_wp, reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
        min_steps=MIN_STEPS_PER_SEGMENT, reference_amplitude=emax,
    )
    step_offsets = np.concatenate(([0], np.cumsum(seg_steps)))
    n_steps_total = int(step_offsets[-1])
    print(f"[test-law] test trajectory: {len(E_wp)} waypoints, {n_steps_total} dynamic steps")

    law = LinearHpromIterativeLawFloat64(verbose=False)

    E_hist = np.zeros((n_steps_total + 1, 3), dtype=float)
    sig_hist = np.zeros((n_steps_total + 1, 3), dtype=float)
    iters_hist = np.zeros(n_steps_total + 1, dtype=int)
    conv_hist = np.ones(n_steps_total + 1, dtype=bool)

    q_prev = None
    t0 = time.perf_counter()
    for step in range(1, n_steps_total + 1):
        s = int(np.searchsorted(step_offsets, step, side="left") - 1)
        s = max(0, min(s, n_seg - 1))
        xi = float(step - step_offsets[s]) / float(max(seg_steps[s], 1))
        E_t = (1.0 - xi) * E_wp[s, :] + xi * E_wp[s + 1, :]

        _hom_eps, hom_sig, q_p, n_it, converged = law.evaluate(E_t, q_prev=q_prev, step_index=step)
        E_hist[step] = E_t
        sig_hist[step] = hom_sig
        iters_hist[step] = n_it
        conv_hist[step] = converged
        q_prev = q_p

        if step % 100 == 0 or step == n_steps_total:
            dt = time.perf_counter() - t0
            print(f"  step {step}/{n_steps_total}: sig={hom_sig}, iters={n_it}, converged={converged}, "
                  f"{dt:.1f}s elapsed", flush=True)

    n_failed = int(np.sum(~conv_hist))
    print(f"[test-law] total wall time: {time.perf_counter() - t0:.1f}s")
    print(f"[test-law] failed-to-converge steps: {n_failed}/{n_steps_total}")
    print(f"[test-law] iteration histogram: "
          f"{ {int(k): int(np.sum(iters_hist[1:] == k)) for k in sorted(set(iters_hist[1:].tolist()))} }")

    pann = np.load(PANN_DATA)
    ref_stress = np.asarray(pann["stage10_stress"], dtype=float)
    n = min(sig_hist.shape[0], ref_stress.shape[0])
    err = relative_l2(sig_hist[:n], ref_stress[:n])
    print(f"[test-law] relative L2 reaction-force stress error vs Table-6 ground truth = {err:.4%}")
    for k, comp in enumerate(["xx", "yy", "xy"]):
        e = relative_l2(sig_hist[:n, k], ref_stress[:n, k])
        print(f"  sigma_{comp} relative L2 error = {e:.4%}")

    np.savez(HERE / "linear_hprom_law_table6_result_claude.npz",
             E_hist=E_hist[:n], sig_hist=sig_hist[:n], ref_stress=ref_stress[:n],
             iters_hist=iters_hist[:n], conv_hist=conv_hist[:n], err=err)
    print(f"[test-law] saved to linear_hprom_law_table6_result_claude.npz")


if __name__ == "__main__":
    main()
