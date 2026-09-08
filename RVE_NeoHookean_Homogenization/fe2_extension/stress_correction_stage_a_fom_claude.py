#!/usr/bin/env python3
"""Stage A, FOM-FE2 variant. See stress_correction_stage_a_dhprom_claude.py's
module docstring for the general rationale.

FOM-FE2 has no reduced order/decoder at all, so there is no q_p to extract
-- instead, this re-solves the true, non-reduced RVE FOM from a cold start
(reusing the already-validated time_fom_single_query_claude.solve_at_strain
pattern, via RunFomBatchSimulation directly with save_results=True) at each
of Cook's 384 already-converged final-state macro strains (e_gp, saved in
cook_results_fom_nested_full_claude.npz), reading back the converged full
displacement field immediately after each call (save_results=True writes
it to out_dir, overwritten every call, so it must be read back before the
next iteration overwrites it).

This is NOT re-running the expensive 20-step-ramp full Cook FE2 solve --
it is a single, direct 2-point ramp (0 -> E) to the SAME final macro strain
Cook's own nested FOM already converged to. Since Neo-Hookean hyperelastic
equilibrium is a pure function of state (no history/path dependence), this
must reach the same equilibrium regardless of how E was originally arrived
at inside Cook's own load-stepping -- the cross-check against the already-
saved s_gp/e_gp below verifies this rather than assuming it.
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

COOK_NPZ = HERE / "cook_results_fom_nested_full_claude.npz"
OUT_NPZ = HERE / "stress_correction_stage_a_fom_result_claude.npz"
SCRATCH_DIR = HERE / "fom_query_scratch_correction"


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


def main() -> None:
    d = np.load(COOK_NPZ)
    e_gp = np.asarray(d["e_gp"], dtype=float)
    s_gp_saved = np.asarray(d["s_gp"], dtype=float)
    n = e_gp.shape[0]
    print(f"[stage-a-fom] loaded e_gp {e_gp.shape}, s_gp {s_gp_saved.shape} from {COOK_NPZ.name}")

    # Probe n_dof from a throwaway zero-strain solve before allocating.
    _eps0, _sig0, u0 = solve_at_strain_full_field(np.zeros(3))
    n_dof = u0.shape[0]
    print(f"[stage-a-fom] n_dof={n_dof}")

    U_final_all = np.zeros((n, n_dof), dtype=float)
    eps_check = np.zeros((n, 3), dtype=float)
    sig_check = np.zeros((n, 3), dtype=float)

    t0 = time.perf_counter()
    for i in range(n):
        E = e_gp[i]
        eps_final, sig_final, u_final = solve_at_strain_full_field(E)
        eps_check[i] = eps_final
        sig_check[i] = sig_final
        U_final_all[i] = u_final
        dt = time.perf_counter() - t0
        print(f"  [{i + 1}/{n}] {dt:.1f}s elapsed ({dt / (i + 1):.3f}s/state)", flush=True)

    dt = time.perf_counter() - t0
    print(f"[stage-a-fom] DONE: {n} fresh cold-start solves, {dt:.1f}s total ({dt / n:.3f}s/state)")

    err = np.linalg.norm(sig_check - s_gp_saved) / max(np.linalg.norm(s_gp_saved), 1.0e-30)
    print(f"[stage-a-fom] cross-check: fresh cold-start native (unweighted volume-average) "
          f"stress vs saved s_gp, relative L2 = {err:.6e}")
    if err > 1.0e-3:
        print("[stage-a-fom] WARNING: cross-check error is larger than expected for a "
              "re-solve at the same imposed E -- investigate before trusting U_final_all "
              "downstream.")

    np.savez(OUT_NPZ, e_gp=e_gp, U_final_all=U_final_all, eps_check=eps_check,
              sig_check=sig_check, s_gp_saved=s_gp_saved, cross_check_relative_l2=err)
    print(f"[stage-a-fom] saved to {OUT_NPZ}")


if __name__ == "__main__":
    main()
