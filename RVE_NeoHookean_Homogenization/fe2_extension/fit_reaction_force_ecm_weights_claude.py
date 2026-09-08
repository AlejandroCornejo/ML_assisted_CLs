#!/usr/bin/env python3
"""Track B, Stage 4: fit the continuous weight-field regressor (ANN) on
Stage 3's pruned support + per-training-state weight table, matching the
existing sig/eps targets' own choice of regressor (per this session's
earlier finding: "both eps and sig use ANN").

Verification gate, using an already-available held-out set: FOM-FE2's own
384 Cook states (never used in the 490-state training set above, which came
from the training trajectories' interior states, not Cook's own solve).
stress_correction_stage_a_fom_result_claude.npz already has the full
reconstructed displacement field U_final_all at those 384 states;
stress_correction_stage_b_fom_result_claude.npz already has the true
reaction-force stress_rf at those same states. This script computes c_e at
those states restricted to the pruned 6-point support, combines with the
newly-fitted ANN's predicted weights, and compares the resulting predicted
stress against the true stress_rf -- a genuine, already-in-hand held-out
accuracy check, not just a weight-sanity check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MAWECM_DIR = HERE.parent / "mawecm"
if str(MAWECM_DIR) not in sys.path:
    sys.path.insert(0, str(MAWECM_DIR))

from mawecm_ann_weights import fit_mawecm_ann, eval_mawecm_ann  # noqa: E402
from reaction_force_ecm_target_claude import (  # noqa: E402
    DirectStressGenerator,
    build_dof_to_dirpos,
    per_element_reaction_force_contribution,
)

REPO_ROOT = HERE.parent
N_STOP = 20
DATASET_NPZ = HERE / "reaction_force_ecm_dataset_claude.npz"
PRUNING_NPZ = HERE / f"reaction_force_ecm_pruning_result_nstop{N_STOP}_claude.npz"
FOM_STAGE_A_NPZ = HERE / "stress_correction_stage_a_fom_result_claude.npz"
FOM_STAGE_B_NPZ = HERE / "stress_correction_stage_b_fom_result_claude.npz"
STAGE10_DIR = REPO_ROOT / "hprom" / "ann" / "stage_10_results_maw_dynamic"
PANN_DATA = REPO_ROOT / "pann" / "data" / "alltraj_stage10_direct_energy.npz"
OUT_NPZ = HERE / f"reaction_force_ecm_ann_model_nstop{N_STOP}_claude.npz"


def relative_l2(pred, ref):
    return float(np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), 1.0e-30))


def evaluate_held_out(gen, dof_to_dirpos, ann, Z_support, e_gp_held, U_held, stress_true, label):
    w_pred = eval_mawecm_ann(e_gp_held, ann)  # (n_support, n_held)
    assert np.all(w_pred >= -1.0e-9), f"{label}: predicted weights went negative"
    n_held = e_gp_held.shape[0]
    stress_pred = np.zeros((n_held, 3), dtype=float)
    for i in range(n_held):
        c_e = per_element_reaction_force_contribution(gen, dof_to_dirpos, U_held[i], e_gp_held[i])
        stress_pred[i] = c_e[Z_support, :].T @ w_pred[:, i]
    err = relative_l2(stress_pred, stress_true)
    print(f"[fit-rf-ann] HELD-OUT ACCURACY ({label}, n={n_held}): "
          f"new {Z_support.size}-point rule vs true reaction-force stress: "
          f"relative L2 = {err:.4%}")
    return err, stress_pred


def main() -> None:
    dataset = np.load(DATASET_NPZ)
    pruning = np.load(PRUNING_NPZ)

    z_ini = dataset["z_ini"]
    Z_support = pruning["Z_support"]
    W_support = pruning["W_support"]  # (n_support, n_states)
    q_train = pruning["q_train"]  # (n_states, 3), macro-strain mu

    support_pos = np.searchsorted(z_ini, Z_support)
    A_full = dataset["A_blocks"]  # (n_states, 3, n_cand)
    b_full = dataset["b_blocks"]  # (n_states, 3)
    n_states = A_full.shape[0]
    A_support = [A_full[s][:, support_pos] for s in range(n_states)]
    b_list = [b_full[s] for s in range(n_states)]

    print(f"[fit-rf-ann] fitting ANN over {Z_support.size}-point support, {n_states} training states")

    ann = fit_mawecm_ann(
        q_train=q_train,
        W_train=W_support,
        target_sum=0.0,  # let the function derive a sensible target from the pruned table itself
        constraint_A_blocks=A_support,
        constraint_b_blocks=b_list,
        physics_q_train=q_train,
        physics_constraint_A_blocks=A_support,
        physics_constraint_b_blocks=b_list,
        hidden_dims=(128, 128, 128),
        activation="gelu",
        epochs=4000,
        physics_weight=1.0,
        verbose=True,
        label="MAW-RF-ANN",
    )
    print(f"[fit-rf-ann] training done, train_rel_error={ann.get('train_rel_error')}")

    gen = DirectStressGenerator()
    dof_to_dirpos = build_dof_to_dirpos(gen)

    # Verification gate 1: the CANONICAL held-out set this project already
    # uses for Table 6 (the Stage-10 test trajectory, 1151 states, never used
    # in training above) -- explicit user instruction (this session): confirm
    # against this specific, already-established trajectory, not just an
    # improvised one.
    U_stage10 = np.load(STAGE10_DIR / "single_run_U.npy")
    e_stage10 = np.load(STAGE10_DIR / "single_run_applied_strain.npy")
    pann = np.load(PANN_DATA)
    stress_stage10_true = np.asarray(pann["stage10_stress"], dtype=float)
    assert np.allclose(e_stage10, pann["stage10_strain"]), "stage10 strain mismatch"
    err_stage10, _ = evaluate_held_out(
        gen, dof_to_dirpos, ann, Z_support, e_stage10, U_stage10, stress_stage10_true,
        label="Table-6 canonical test trajectory",
    )

    # Verification gate 2: FOM-FE2's own 384 held-out Cook states (a second,
    # independent held-out set, structurally different from a trajectory --
    # nested-FE2 macro-strain states, not a single continuous path).
    stage_a = np.load(FOM_STAGE_A_NPZ)
    e_gp_held = stage_a["e_gp"]
    U_held = stage_a["U_final_all"]
    stage_b = np.load(FOM_STAGE_B_NPZ)
    stress_rf_true = stage_b["stress_rf"]
    err_cook, _ = evaluate_held_out(
        gen, dof_to_dirpos, ann, Z_support, e_gp_held, U_held, stress_rf_true,
        label="FOM-FE2 Cook states",
    )
    gen.close()

    print(f"[fit-rf-ann] for comparison, this session's existing findings at the Cook states: "
          f"native Z_sigma (10 pts, naive-average target) vs true conjugate was ~1.8-3.6% "
          f"for the ROM rows.")

    np.savez(OUT_NPZ, Z_support=Z_support, err_stage10=err_stage10, err_cook=err_cook,
             **{k: v for k, v in ann.items() if isinstance(v, np.ndarray)},
             **{f"scalar_{k}": v for k, v in ann.items() if not isinstance(v, np.ndarray)})
    print(f"[fit-rf-ann] saved to {OUT_NPZ}")


if __name__ == "__main__":
    main()
