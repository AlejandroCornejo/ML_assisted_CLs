#!/usr/bin/env python3
"""Consistency check: is Table 7's "S err." for the PANN tiers (which output
the energy-conjugate stress directly, by training construction) comparable
to HPROM-ANN/D-HPROM-ANN's own "S err." (native volume-average vs FOM-FE2's
own volume-average, same family, hyperreduction-fidelity-only question)?

A PANN has no inner RVE Gauss-point field at all -- it is an end-to-end
network trained directly on (E, S) pairs where S is already the energy-
conjugate quantity (verified earlier this session, full-coverage, all 10
training trajectories + held-out). So a PANN's own Cook output should
approximate the CONJUGATE stress, not the naive volume average that FOM-
FE2's own s_gp (the reference Table 7 compares every row against) reports.
If these differ meaningfully at the states the ICNN's own Cook run visits,
Table 7's reported ICNN "S err." (0.29%) is being measured against the
wrong-convention reference, same issue as HPROM-ANN/D-HPROM-ANN, just for
a structurally different reason (PANN has no ECM/MAW sparsity choice to
make at all -- it never had access to the naive-average convention).

This re-solves the true FOM from a cold start (same pattern as
stress_correction_stage_a_fom_claude.py) at ICNN's OWN 384 saved final-
state macro strains (cook_results_icnn_w5_final_claude.npz's e_gp, NOT
FOM-FE2's own e_gp -- different converged states, since ICNN's own
stiffness differs slightly from the true FOM's), and applies the reaction-
force correction (via Stage B's pattern) to get the true conjugate
reference AT THOSE SPECIFIC STATES, for a direct three-way comparison:
  (1) ICNN's own reported s_gp (conjugate-approximating, by training)
  (2) fresh cold-start FOM native volume-average at the same e_gp (should
      closely reproduce the currently-reported 0.29% error against (1),
      as a sanity check that this script targets the right states)
  (3) fresh cold-start FOM reaction-force-corrected stress at the same e_gp
      (the TRUE conjugate reference -- the real apples-to-apples comparison
      for a PANN)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CORE_DIR = HERE.parent / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import fom_solver_rve as fom  # noqa: E402
from _material_law_guard_claude import true_neo_hookean_active  # noqa: E402
from time_fom_single_query_claude import make_parameters  # noqa: E402

ICNN_NPZ = HERE / "cook_results_icnn_w5_final_claude.npz"
OUT_NPZ = HERE / "stress_correction_stage_a_icnn_check_result_claude.npz"
SCRATCH_DIR = HERE / "fom_query_scratch_icnn_check"


def solve_at_strain_full_field(E, reference_amplitude=2.0, reference_steps=400):
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    with true_neo_hookean_active():
        strain_hist, stress_hist = fom.RunFomBatchSimulation(
            parameters=make_parameters(), out_dir=str(SCRATCH_DIR),
            save_results=True, save_plot=False,
            strain_path=np.vstack(([0.0, 0.0, 0.0], E)), trajectory_index=None,
            reference_amplitude=reference_amplitude, reference_steps=reference_steps,
        )
    U_hist = np.load(SCRATCH_DIR / "single_run_U.npy")
    return np.asarray(strain_hist[-1]), np.asarray(stress_hist[-1]), U_hist[-1]


def relative_l2(prediction, reference):
    return float(np.linalg.norm(prediction - reference) / max(np.linalg.norm(reference), 1.0e-30))


def main() -> None:
    d = np.load(ICNN_NPZ)
    e_gp = np.asarray(d["e_gp"], dtype=float)
    s_gp_icnn = np.asarray(d["s_gp"], dtype=float)
    n = e_gp.shape[0]
    print(f"[icnn-check] loaded e_gp {e_gp.shape}, s_gp (ICNN's own) {s_gp_icnn.shape}")

    _eps0, _sig0, u0 = solve_at_strain_full_field(np.zeros(3))
    n_dof = u0.shape[0]
    U_final_all = np.zeros((n, n_dof), dtype=float)
    sig_naive_fresh = np.zeros((n, 3), dtype=float)

    t0 = time.perf_counter()
    for i in range(n):
        E = e_gp[i]
        _eps_final, sig_final, u_final = solve_at_strain_full_field(E)
        sig_naive_fresh[i] = sig_final
        U_final_all[i] = u_final
        if (i + 1) % 32 == 0:
            dt = time.perf_counter() - t0
            print(f"  [{i + 1}/{n}] {dt:.1f}s elapsed ({dt / (i + 1):.3f}s/state)", flush=True)

    dt = time.perf_counter() - t0
    print(f"[icnn-check] DONE: {n} fresh cold-start solves, {dt:.1f}s total ({dt / n:.3f}s/state)")

    err_icnn_vs_naive_fresh = relative_l2(s_gp_icnn, sig_naive_fresh)
    print(f"[icnn-check] ICNN's own s_gp vs FRESH cold-start naive-average FOM at ICNN's own "
          f"e_gp: relative L2 = {err_icnn_vs_naive_fresh:.6e} "
          f"(sanity check -- should be close to the already-reported 0.29% if this targets "
          f"the right states)")

    np.savez(OUT_NPZ, e_gp=e_gp, s_gp_icnn=s_gp_icnn, U_final_all=U_final_all,
             sig_naive_fresh=sig_naive_fresh, err_icnn_vs_naive_fresh=err_icnn_vs_naive_fresh)
    print(f"[icnn-check] saved to {OUT_NPZ}")


if __name__ == "__main__":
    main()
