#!/usr/bin/env python3
"""Answers a precise question: if the existing Z_sigma (naive-average
target) inside HPROM-ANN/D-HPROM-ANN's own online reconstruction is
replaced by the new 10-point reaction-force-targeting rule, what test-
trajectory stress error results -- directly comparable to the existing
\\HpromIterStressError (0.174%) / \\DHpromStressError (0.202%) macros?

This is DIFFERENT from evaluate_maw_from_classic_claude.py's own check,
which applies the new rule to the TRUE FOM's own field (isolating the new
rule's own hyperreduction quality, decoupled from HPROM-ANN's manifold-
regression quality). Here, the new rule is applied to HPROM-ANN's/D-HPROM-
ANN's OWN reconstructed field (same LS-decoder reconstruction already
validated in pann/direct_energy/hprom_ann_direct_stress.py), so the result
combines both error sources -- the real, honest number for the table.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DIRECT_ENERGY_DIR = REPO_ROOT / "pann" / "direct_energy"
PROM_ANN_DIR = REPO_ROOT / "prom" / "ann"
POD_DIR = REPO_ROOT / "pod" / "stage_2_pod_rve"
ANN_MODEL_DIR = PROM_ANN_DIR / "stage_7_ann_model_ls"
STAGE10_DIR = REPO_ROOT / "hprom" / "ann" / "stage_10_results_maw_dynamic"
PANN_DATA = REPO_ROOT / "pann" / "data" / "alltraj_stage10_direct_energy.npz"

for p in (DIRECT_ENERGY_DIR, PROM_ANN_DIR, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

from reaction_force_direct_stress import DirectStressGenerator  # noqa: E402
from prom_ann_solver_rve import LoadPromAnnModel  # noqa: E402
from hprom_ann_direct_stress import (  # noqa: E402
    reconstruct_free_fluctuation,
    free_dof_reference_coordinates,
    assemble_full_displacement,
)
from reaction_force_ecm_target_claude import build_dof_to_dirpos, per_element_reaction_force_contribution  # noqa: E402

NEW_RULE_NPZ = HERE / "reaction_force_ecm_pruning_from_classic_nstop10_claude.npz"


def relative_l2(pred, ref):
    return float(np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), 1.0e-30))


def main() -> None:
    rule = np.load(NEW_RULE_NPZ)
    Z_support = rule["Z_support"]
    W_support = rule["W_support"]
    q_train = rule["q_train"]

    gen = DirectStressGenerator()
    free_dofs_pod = np.load(POD_DIR / "free_dofs.npy")
    dir_dofs_pod = np.load(POD_DIR / "dirichlet_dofs.npy")
    phi_p, phi_s, free_dofs_ref, _dir_dofs_ref, _eq_map_ref, ann_model, device, include_macro = LoadPromAnnModel(
        basis_dir=str(POD_DIR), ann_data_dir=str(ANN_MODEL_DIR)
    )
    a_m = np.asarray(ann_model.a_m_np, dtype=float)
    x_free, y_free, is_x_free = free_dof_reference_coordinates(free_dofs_pod, gen)

    applied_strain = np.load(STAGE10_DIR / "single_run_applied_strain.npy")
    pann = np.load(PANN_DATA)
    stage10_stress_true = np.asarray(pann["stage10_stress"], dtype=float)

    dof_to_dirpos = build_dof_to_dirpos(gen)

    def evaluate_mode(q_p_path, label):
        q_p_hist = np.load(q_p_path)
        n = q_p_hist.shape[0]
        u_fluc_hist = reconstruct_free_fluctuation(q_p_hist, phi_p, phi_s, a_m, ann_model, device)
        U_full = assemble_full_displacement(u_fluc_hist, applied_strain[:n], free_dofs_pod,
                                             x_free, y_free, is_x_free, gen)
        pred = np.zeros((n, 3), dtype=float)
        for i in range(n):
            nearest = np.argmin(np.sum((q_train - applied_strain[i]) ** 2, axis=1))
            w = W_support[:, nearest]
            c_e = per_element_reaction_force_contribution(gen, dof_to_dirpos, U_full[i], applied_strain[i])
            pred[i] = c_e[Z_support, :].T @ w
        err = relative_l2(pred, stage10_stress_true[:n])
        print(f"[new-rule-on-hprom-field] {label}: new 10-point rule on the model's OWN "
              f"reconstructed field vs true stage10_stress, n={n}: relative L2 = {err:.4%}")
        return err

    err_hprom = evaluate_mode(STAGE10_DIR / "hprom_ann_run_q_p.npy", "HPROM-ANN (iterative)")
    err_dhprom = evaluate_mode(STAGE10_DIR / "trajectory_direct_hprom_ann_q_p.npy", "D-HPROM-ANN (direct)")
    gen.close()

    print(f"[new-rule-on-hprom-field] for comparison, OLD (Z_sigma native + full reaction-force "
          f"correction) numbers already in the paper: HPROM-ANN 0.174%, D-HPROM-ANN 0.202%")


if __name__ == "__main__":
    main()
