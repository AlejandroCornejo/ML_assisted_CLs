#!/usr/bin/env python3
"""Track B, alternative experiment: classic (non-adaptive, fixed-weight) ECM
on the reaction-force target, instead of MAW's per-state-adaptive pruning +
ANN regression. One weight vector, found once via the standard RSVD+greedy-
NNLS ECM recipe (core/empirical_cubature_method.py,
core/randomized_singular_value_decomposition.py -- both already
self-contained in this project, already used for this project's own actual
Z/Z_eps/Z_sig construction via hprom/ann/stage9_compute_ecm_weights_ann.py's
own run_ecm/run_rsvd_on_transpose pattern, reused verbatim here).

No phase-1/phase-2 reduction at all -- ECM's own greedy algorithm decides
how many points it needs to reach the requested tolerance, then stops.
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
OUT_NPZ = HERE / "classic_ecm_reaction_force_result_claude.npz"

RSVD_TOL = 1.0e-4
ECM_TOL = 1.0e-4


def relative_l2(pred, ref):
    return float(np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), 1.0e-30))


def main() -> None:
    d = np.load(DATASET_NPZ)
    z_ini = d["z_ini"]  # (156,) candidate element indices
    n_cand = z_ini.size
    A_blocks = d["A_blocks"]  # (n_states, 3, n_cand)
    n_states = A_blocks.shape[0]
    C = A_blocks.reshape(n_states * 3, n_cand)  # (3*n_states, n_cand), matches C_sig's own convention
    print(f"[classic-ecm] snapshot matrix C shape={C.shape} (rows=3 x {n_states} states, cols={n_cand} candidates)")

    print(f"[classic-ecm] running RSVD on C.T (shape {C.T.shape}), tol={RSVD_TOL}")
    rsvd = RandomizedSingularValueDecomposition(
        COMPUTE_U=True, COMPUTE_V=False, RELATIVE_SVD=True, USE_RANDOMIZATION=True,
    )
    U, s, _, eSVD = rsvd.Calculate(np.ascontiguousarray(C.T), truncation_tolerance=float(RSVD_TOL))
    print(f"[classic-ecm] RSVD done: U.shape={U.shape}, kept modes={s.size}, eSVD={eSVD:.3e}")

    print(f"[classic-ecm] running classic ECM, tol={ECM_TOL}")
    ecm = EmpiricalCubatureMethod(ECM_tolerance=float(ECM_TOL), Filter_tolerance=0.0, Plotting=False,
                                  MaximumNumberUnsuccesfulIterations=200)
    ecm.SetUp(ResidualsBasis=U, InitialCandidatesSet=None, constrain_sum_of_weights=False,
              constrain_conditions=False, number_of_conditions=0)
    ecm.Run()
    Z_local = np.array(ecm.z, dtype=int).ravel()
    w_sel = np.array(ecm.w, dtype=float).ravel()
    Z_elements = z_ini[Z_local]
    print(f"[classic-ecm] SELECTED |Z| = {Z_local.size} (of {n_cand} candidates, "
          f"{100.0 * Z_local.size / n_cand:.1f}%)")
    print(f"[classic-ecm] element indices: {Z_elements}")
    print(f"[classic-ecm] weights: {w_sel}")

    # Discrete training-state reproduction check (fixed weights, not per-state).
    A_support = A_blocks[:, :, Z_local]  # (n_states, 3, |Z|)
    b_full = d["b_blocks"]
    pred_train = np.einsum("skz,z->sk", A_support, w_sel)
    train_err = relative_l2(pred_train, b_full)
    print(f"[classic-ecm] fixed-weight reproduction over all {n_states} training states: "
          f"relative L2 = {train_err:.4%} (a single compromise weight vector, NOT expected "
          f"to be machine-precision like the adaptive per-state table)")

    # Held-out checks, same two sets as the MAW experiments.
    gen = DirectStressGenerator()
    dof_to_dirpos = build_dof_to_dirpos(gen)

    def evaluate_fixed(e_gp_held, U_held, stress_true, label):
        n_held = e_gp_held.shape[0]
        pred = np.zeros((n_held, 3), dtype=float)
        for i in range(n_held):
            c_e = per_element_reaction_force_contribution(gen, dof_to_dirpos, U_held[i], e_gp_held[i])
            pred[i] = c_e[Z_elements, :].T @ w_sel
        err = relative_l2(pred, stress_true)
        print(f"[classic-ecm] HELD-OUT ACCURACY ({label}, n={n_held}): "
              f"fixed {Z_local.size}-point rule vs true reaction-force stress: relative L2 = {err:.4%}")
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
    print(f"[classic-ecm] saved to {OUT_NPZ}")


if __name__ == "__main__":
    main()
