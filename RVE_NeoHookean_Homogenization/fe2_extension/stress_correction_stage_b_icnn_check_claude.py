#!/usr/bin/env python3
"""Stage B of the ICNN consistency check. Takes Stage A's saved
U_final_all (384, n_dof) -- the fresh, cold-start FOM displacement field at
each of ICNN's own 384 converged Cook states -- and applies the reaction-
force correction (DirectStressGenerator, unmodified) to get the TRUE
energy-conjugate reference at those exact states, for a direct comparison
against ICNN's own (conjugate-approximating, by training) output.

Three numbers reported together for a complete picture:
  (1) ICNN's own s_gp vs fresh naive-average FOM at the same e_gp (Stage
      A's own result, 1.26% -- isolates state-matched value error only,
      under the WRONG reference convention)
  (2) ICNN's own s_gp vs fresh reaction-force-corrected FOM at the same
      e_gp (THIS script's new result -- the correct, apples-to-apples
      comparison, since ICNN was only ever trained to approximate this
      quantity)
  (3) for context, the already-reported Table 7 number (0.29%, ICNN's own
      state vs FOM-FE2's own, different, state, wrong reference convention)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DIRECT_ENERGY_DIR = REPO_ROOT / "pann" / "direct_energy"
if str(DIRECT_ENERGY_DIR) not in sys.path:
    sys.path.insert(0, str(DIRECT_ENERGY_DIR))

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

from reaction_force_direct_stress import DirectStressGenerator  # noqa: E402

STAGE_A_NPZ = HERE / "stress_correction_stage_a_icnn_check_result_claude.npz"
OUT_NPZ = HERE / "stress_correction_stage_b_icnn_check_result_claude.npz"


def relative_l2(prediction, reference):
    return float(np.linalg.norm(prediction - reference) / max(np.linalg.norm(reference), 1.0e-30))


def main() -> None:
    stage_a = np.load(STAGE_A_NPZ)
    e_gp = np.asarray(stage_a["e_gp"], dtype=float)
    U_final_all = np.asarray(stage_a["U_final_all"], dtype=float)
    s_gp_icnn = np.asarray(stage_a["s_gp_icnn"], dtype=float)
    sig_naive_fresh = np.asarray(stage_a["sig_naive_fresh"], dtype=float)
    n = e_gp.shape[0]
    print(f"[icnn-check-b] loaded U_final_all {U_final_all.shape}, e_gp {e_gp.shape}")

    gen = DirectStressGenerator()
    stress_rf = gen.direct_stress_history(U_final_all, e_gp)
    gen.close()

    err_vs_naive = relative_l2(s_gp_icnn, sig_naive_fresh)
    err_vs_conjugate = relative_l2(s_gp_icnn, stress_rf)
    err_naive_vs_conjugate = relative_l2(sig_naive_fresh, stress_rf)

    print(f"[icnn-check-b] (1) ICNN vs fresh naive-average FOM, same e_gp: {err_vs_naive:.6e}")
    print(f"[icnn-check-b] (2) ICNN vs fresh reaction-force-corrected FOM, same e_gp: {err_vs_conjugate:.6e}")
    print(f"[icnn-check-b] (for context) naive-average vs reaction-force gap at ICNN's own states: "
          f"{err_naive_vs_conjugate:.6e}")
    print(f"[icnn-check-b] (for context) Table 7's own reported ICNN S err. (different-state "
          f"comparison, vs FOM-FE2's own state): 0.29%")

    np.savez(OUT_NPZ, e_gp=e_gp, s_gp_icnn=s_gp_icnn, sig_naive_fresh=sig_naive_fresh,
             stress_rf=stress_rf, err_vs_naive=err_vs_naive, err_vs_conjugate=err_vs_conjugate,
             err_naive_vs_conjugate=err_naive_vs_conjugate)
    print(f"[icnn-check-b] saved to {OUT_NPZ}")


if __name__ == "__main__":
    main()
