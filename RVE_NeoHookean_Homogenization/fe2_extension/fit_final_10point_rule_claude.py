#!/usr/bin/env python3
"""Fit a continuous ANN weight-field regressor on the best Track B result so
far: classic ECM (5%, FPS/param-aware sampling, 37 points) refined via
phase-2-only graph-regularized pruning to 10 points
(reaction_force_ecm_pruning_from_classic_nstop10_5pct_fps_claude.npz,
already validated via nearest-state lookup at 0.156%/0.151%).

This replaces that crude nearest-neighbor proxy with a real, smooth,
continuously-evaluable function of macro-strain (matching the exact format
this project's existing online evaluator already expects --
hprom_ann_solver_rve.py's _build_maw_hom_target_model/eval via
mawecm/mawecm_ann_weights.py's eval_mawecm_ann), needed before this can
actually be wired into DHpromAnnDirectLawFloat64/HpromAnnIterativeLawFloat64.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
MAWECM_DIR = REPO_ROOT / "mawecm"
if str(MAWECM_DIR) not in sys.path:
    sys.path.insert(0, str(MAWECM_DIR))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mawecm_ann_weights import fit_mawecm_ann, eval_mawecm_ann  # noqa: E402
from reaction_force_ecm_target_claude import (  # noqa: E402
    DirectStressGenerator,
    build_dof_to_dirpos,
    per_element_reaction_force_contribution,
)

DATASET_NPZ = HERE / "reaction_force_ecm_dataset_claude.npz"
PRUNING_NPZ = HERE / "reaction_force_ecm_pruning_from_classic_nstop10_5pct_fps_claude.npz"
STAGE10_DIR = REPO_ROOT / "hprom" / "ann" / "stage_10_results_maw_dynamic"
PANN_DATA = REPO_ROOT / "pann" / "data" / "alltraj_stage10_direct_energy.npz"
FOM_STAGE_A_NPZ = HERE / "stress_correction_stage_a_fom_result_claude.npz"
FOM_STAGE_B_NPZ = HERE / "stress_correction_stage_b_fom_result_claude.npz"
GRAPH_SUBSAMPLE_SIZE = 500  # must match run_maw_from_classic_ecm_claude.py's own choice exactly
OUT_NPZ = HERE / "reaction_force_ecm_ann_model_FINAL_claude.npz"


def relative_l2(pred, ref):
    return float(np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), 1.0e-30))


def evaluate_held_out(gen, dof_to_dirpos, ann, Z_support, e_gp_held, U_held, stress_true, label):
    w_pred = eval_mawecm_ann(e_gp_held, ann)
    print(f"[fit-final] {label}: predicted weight range=[{w_pred.min():.4f}, {w_pred.max():.4f}]")
    n_held = e_gp_held.shape[0]
    stress_pred = np.zeros((n_held, 3), dtype=float)
    for i in range(n_held):
        c_e = per_element_reaction_force_contribution(gen, dof_to_dirpos, U_held[i], e_gp_held[i])
        stress_pred[i] = c_e[Z_support, :].T @ w_pred[:, i]
    err = relative_l2(stress_pred, stress_true)
    print(f"[fit-final] HELD-OUT ACCURACY ({label}, n={n_held}): relative L2 = {err:.4%}")
    return err


def main() -> None:
    dataset = np.load(DATASET_NPZ)
    pruning = np.load(PRUNING_NPZ)

    z_ini_full = dataset["z_ini"]  # (156,)
    Z_support = pruning["Z_support"]  # (10,) final mesh element indices
    W_support = pruning["W_support"]  # (10, 500)
    q_train = pruning["q_train"]  # (500, 3) -- the exact 500-state subsample used by pruning

    # Reproduce the exact same 500-state subsample of the full dataset (same
    # np.linspace over the full, non-subsampled A_blocks/b_blocks) so the
    # physics-constraint terms fed to fit_mawecm_ann match q_train exactly.
    A_full_all = dataset["A_blocks"]
    b_full_all = dataset["b_blocks"]
    n_states_all = A_full_all.shape[0]
    n_sub = min(n_states_all, GRAPH_SUBSAMPLE_SIZE)
    sub_idx = np.linspace(0, n_states_all - 1, n_sub, dtype=int)
    assert np.allclose(dataset["mu_train"][sub_idx], q_train), "subsample index mismatch -- check GRAPH_SUBSAMPLE_SIZE"

    support_pos = np.searchsorted(z_ini_full, Z_support)
    assert np.array_equal(z_ini_full[support_pos], Z_support), "element index mapping mismatch"

    A_sub = A_full_all[sub_idx][:, :, support_pos]  # (500, 3, 10)
    b_sub = b_full_all[sub_idx]  # (500, 3)
    A_blocks = [A_sub[s] for s in range(n_sub)]
    b_blocks = [b_sub[s] for s in range(n_sub)]

    print(f"[fit-final] fitting ANN over {Z_support.size}-point support, {n_sub} training states")

    ann = fit_mawecm_ann(
        q_train=q_train,
        W_train=W_support,
        target_sum=0.0,
        constraint_A_blocks=A_blocks,
        constraint_b_blocks=b_blocks,
        physics_q_train=q_train,
        physics_constraint_A_blocks=A_blocks,
        physics_constraint_b_blocks=b_blocks,
        hidden_dims=(256, 256, 256),
        activation="gelu",
        epochs=40000,
        patience=3000,
        physics_weight=1.0,
        verbose=True,
        label="MAW-RF-FINAL",
    )
    print(f"[fit-final] training done, train_rel_error={ann.get('train_rel_error')}, "
          f"val_rel_error={ann.get('val_rel_error')}")

    gen = DirectStressGenerator()
    dof_to_dirpos = build_dof_to_dirpos(gen)

    U_stage10 = np.load(STAGE10_DIR / "single_run_U.npy")
    e_stage10 = np.load(STAGE10_DIR / "single_run_applied_strain.npy")
    pann = np.load(PANN_DATA)
    stress_stage10_true = np.asarray(pann["stage10_stress"], dtype=float)
    err_stage10 = evaluate_held_out(gen, dof_to_dirpos, ann, Z_support, e_stage10, U_stage10,
                                     stress_stage10_true, "Table-6 test trajectory")

    stage_a = np.load(FOM_STAGE_A_NPZ)
    stage_b = np.load(FOM_STAGE_B_NPZ)
    err_cook = evaluate_held_out(gen, dof_to_dirpos, ann, Z_support, stage_a["e_gp"],
                                  stage_a["U_final_all"], stage_b["stress_rf"], "FOM-FE2 Cook states")
    gen.close()

    print(f"[fit-final] for comparison, nearest-state-lookup validation (not a real deployable "
          f"function): 0.1564% (Table-6), 0.1510% (Cook)")

    np.savez(OUT_NPZ, Z_support=Z_support, err_stage10=err_stage10, err_cook=err_cook,
             **{k: v for k, v in ann.items() if isinstance(v, np.ndarray)},
             **{f"scalar_{k}": v for k, v in ann.items() if not isinstance(v, np.ndarray)})
    print(f"[fit-final] saved to {OUT_NPZ}")


if __name__ == "__main__":
    main()
