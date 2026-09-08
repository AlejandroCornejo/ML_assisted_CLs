#!/usr/bin/env python3
"""Same as run_classic_ecm_reaction_force_claude.py (37-point classic ECM)
but with a tighter RSVD_TOL to keep more modes, giving classic ECM a
higher-rank basis to reproduce -- per the user's own request: does using
more than 37 points (fixed weights, no MAW/ANN adaptivity) improve D-HPROM
-ANN's Cook accuracy further? RSVD_TOL=1e-6 gives rank~82 (found by direct
probe: 1e-4->37, 1e-5->61, 1e-6->82, 1e-8->100, out of 156 candidates).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CORE_DIR = HERE.parent / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from empirical_cubature_method import EmpiricalCubatureMethod  # noqa: E402
from randomized_singular_value_decomposition import RandomizedSingularValueDecomposition  # noqa: E402
from reaction_force_ecm_target_claude import (  # noqa: E402
    DirectStressGenerator,
    build_dof_to_dirpos,
    per_element_reaction_force_contribution,
)

DATASET_NPZ = HERE / "reaction_force_ecm_dataset_claude.npz"
STAGE10_DIR = HERE.parent / "hprom" / "ann" / "stage_10_results_maw_dynamic"
PANN_DATA = HERE.parent / "pann" / "data" / "alltraj_stage10_direct_energy.npz"
FOM_STAGE_A_NPZ = HERE / "stress_correction_stage_a_fom_result_claude.npz"
FOM_STAGE_B_NPZ = HERE / "stress_correction_stage_b_fom_result_claude.npz"
OUT_NPZ = HERE / "classic_ecm_reaction_force_result_n82_claude.npz"

RSVD_TOL = 1.0e-6
ECM_TOL = 1.0e-8  # tight enough to force ECM to use the full RSVD rank


def relative_l2(pred, ref):
    return float(np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), 1.0e-30))


def main() -> None:
    d = np.load(DATASET_NPZ)
    z_ini = d["z_ini"]
    n_cand = z_ini.size
    A_blocks = d["A_blocks"]
    n_states = A_blocks.shape[0]
    C = A_blocks.reshape(n_states * 3, n_cand)
    print(f"[classic-ecm-n82] snapshot matrix C shape={C.shape}")

    rsvd = RandomizedSingularValueDecomposition(
        COMPUTE_U=True, COMPUTE_V=False, RELATIVE_SVD=True, USE_RANDOMIZATION=False,
    )
    U, s, _, eSVD = rsvd.Calculate(np.ascontiguousarray(C.T), truncation_tolerance=float(RSVD_TOL))
    print(f"[classic-ecm-n82] classic (non-randomized) SVD done: U.shape={U.shape}")

    ecm = EmpiricalCubatureMethod(ECM_tolerance=float(ECM_TOL), Filter_tolerance=0.0, Plotting=False,
                                  MaximumNumberUnsuccesfulIterations=300)
    ecm.SetUp(ResidualsBasis=U, InitialCandidatesSet=None, constrain_sum_of_weights=False,
              constrain_conditions=False, number_of_conditions=0)
    ecm.Run()
    Z_local = np.array(ecm.z, dtype=int).ravel()
    w_sel = np.array(ecm.w, dtype=float).ravel()
    Z_elements = z_ini[Z_local]
    print(f"[classic-ecm-n82] SELECTED |Z| = {Z_local.size} (of {n_cand} candidates)")

    A_support = A_blocks[:, :, Z_local]
    b_full = d["b_blocks"]
    pred_train = np.einsum("skz,z->sk", A_support, w_sel)
    train_err = relative_l2(pred_train, b_full)
    print(f"[classic-ecm-n82] fixed-weight reproduction over all {n_states} training states: {train_err:.4%}")

    gen = DirectStressGenerator()
    dof_to_dirpos = build_dof_to_dirpos(gen)

    def evaluate_fixed(e_gp_held, U_held, stress_true, label):
        n_held = e_gp_held.shape[0]
        pred = np.zeros((n_held, 3), dtype=float)
        for i in range(n_held):
            c_e = per_element_reaction_force_contribution(gen, dof_to_dirpos, U_held[i], e_gp_held[i])
            pred[i] = c_e[Z_elements, :].T @ w_sel
        err = relative_l2(pred, stress_true)
        print(f"[classic-ecm-n82] HELD-OUT ({label}, n={n_held}): relative L2 = {err:.4%}")
        return err

    U_stage10 = np.load(STAGE10_DIR / "single_run_U.npy")
    e_stage10 = np.load(STAGE10_DIR / "single_run_applied_strain.npy")
    pann = np.load(PANN_DATA)
    stress_stage10_true = np.asarray(pann["stage10_stress"], dtype=float)
    err_stage10 = evaluate_fixed(e_stage10, U_stage10, stress_stage10_true, "Table-6 test trajectory")

    stage_a = np.load(FOM_STAGE_A_NPZ)
    stage_b = np.load(FOM_STAGE_B_NPZ)
    err_cook = evaluate_fixed(stage_a["e_gp"], stage_a["U_final_all"], stage_b["stress_rf"], "FOM-FE2 Cook states")
    gen.close()

    np.savez(OUT_NPZ, Z_elements=Z_elements, w_sel=w_sel, train_err=train_err,
             err_stage10=err_stage10, err_cook=err_cook)
    print(f"[classic-ecm-n82] saved to {OUT_NPZ}")


if __name__ == "__main__":
    main()
