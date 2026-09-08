#!/usr/bin/env python3
"""Stage A, HPROM-ANN (iterative) variant. See
stress_correction_stage_a_dhprom_claude.py's module docstring for the full
rationale; this file mirrors it for the iterative model.

Unlike D-HPROM-ANN, q_p here is NOT a pure closed-form function of E -- it
is found by the inner Newton correction (HpromAnnIterativeLawFloat64's own
evaluate()), which originally ran warm-started (qp_init_mode="continuation")
across Cook's 20 load steps. This script instead calls evaluate(E,
q_prev=None, step_index=1) at each of the 384 already-converged final
strains: with step_index=1, _initial_qp_guess falls through to the
mu_affine closed-form guess regardless of qp_init_mode="continuation"
(confirmed by direct reading of hprom_ann_iterative_law_float64_claude.py's
_initial_qp_guess), i.e. a cold start, matching D-HPROM-ANN's own initial
guess. Newton's fixed point should be the same regardless of path to it
for a well-posed problem; the cross-check against the already-saved s_gp
below (same relative-L2 discipline as Stage A/dhprom) verifies this
assumption rather than trusting it blindly, and the returned convergence
flag is checked for every single state.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

from hprom_ann_iterative_law_float64_claude import HpromAnnIterativeLawFloat64  # noqa: E402

COOK_NPZ = HERE / "cook_results_hprom_iterative_f64_consistent_claude.npz"
OUT_NPZ = HERE / "stress_correction_stage_a_hprom_result_claude.npz"


def main() -> None:
    d = np.load(COOK_NPZ)
    e_gp = np.asarray(d["e_gp"], dtype=float)
    s_gp_saved = np.asarray(d["s_gp"], dtype=float)
    n = e_gp.shape[0]
    print(f"[stage-a-hprom] loaded e_gp {e_gp.shape}, s_gp {s_gp_saved.shape} from {COOK_NPZ.name}")

    law = HpromAnnIterativeLawFloat64()

    q_p_all = np.zeros((n, law.n_primary), dtype=float)
    hom_sig_check = np.zeros((n, 3), dtype=float)
    n_iters_all = np.zeros(n, dtype=int)
    converged_all = np.zeros(n, dtype=bool)

    t0 = time.perf_counter()
    for i in range(n):
        E = e_gp[i]
        hom_eps, hom_sig, q_p, it, converged = law.evaluate(E, q_prev=None, step_index=1)
        q_p_all[i] = q_p
        hom_sig_check[i] = hom_sig
        n_iters_all[i] = it
        converged_all[i] = converged
        if (i + 1) % 32 == 0:
            dt = time.perf_counter() - t0
            print(f"  [{i + 1}/{n}] {dt:.1f}s elapsed ({dt / (i + 1):.4f}s/call), "
                  f"last it={it}, converged={converged}", flush=True)

    dt = time.perf_counter() - t0
    n_not_converged = int(np.sum(~converged_all))
    print(f"[stage-a-hprom] DONE: {n} calls, {dt:.1f}s total ({dt / n:.4f}s/call), "
          f"{n_not_converged} did not converge, iters min/mean/max="
          f"{n_iters_all.min()}/{n_iters_all.mean():.2f}/{n_iters_all.max()}")

    err = np.linalg.norm(hom_sig_check - s_gp_saved) / max(np.linalg.norm(s_gp_saved), 1.0e-30)
    print(f"[stage-a-hprom] cross-check: fresh cold-start evaluate() hom_sig vs saved s_gp, "
          f"relative L2 = {err:.6e}")
    if err > 1.0e-3:
        print("[stage-a-hprom] WARNING: cross-check error is larger than expected for a "
              "re-convergence to what should be the same Newton fixed point -- investigate "
              "before trusting q_p_all downstream (could be genuine multiple-fixed-point "
              "sensitivity to the initial guess, or a bug).")

    np.savez(OUT_NPZ, e_gp=e_gp, q_p_all=q_p_all, hom_sig_check=hom_sig_check,
              s_gp_saved=s_gp_saved, cross_check_relative_l2=err,
              n_iters_all=n_iters_all, converged_all=converged_all)
    print(f"[stage-a-hprom] saved to {OUT_NPZ}")


if __name__ == "__main__":
    main()
