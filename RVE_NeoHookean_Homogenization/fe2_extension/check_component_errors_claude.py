#!/usr/bin/env python3
"""Break down the final ANN model's held-out accuracy by stress component
(S11, S22, S12) instead of the combined relative L2, to see whether one
component dominates the error."""
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

from mawecm_ann_weights import eval_mawecm_ann  # noqa: E402
from reaction_force_ecm_target_claude import (  # noqa: E402
    DirectStressGenerator,
    build_dof_to_dirpos,
    per_element_reaction_force_contribution,
)

MODEL_NPZ = HERE / "reaction_force_ecm_ann_model_FINAL_claude.npz"
STAGE10_DIR = REPO_ROOT / "hprom" / "ann" / "stage_10_results_maw_dynamic"
PANN_DATA = REPO_ROOT / "pann" / "data" / "alltraj_stage10_direct_energy.npz"
FOM_STAGE_A_NPZ = HERE / "stress_correction_stage_a_fom_result_claude.npz"
FOM_STAGE_B_NPZ = HERE / "stress_correction_stage_b_fom_result_claude.npz"


def component_relative_l2(pred, ref):
    out = []
    for k in range(3):
        num = np.linalg.norm(pred[:, k] - ref[:, k])
        den = max(np.linalg.norm(ref[:, k]), 1.0e-30)
        out.append(num / den)
    return out


def main() -> None:
    model = np.load(MODEL_NPZ)
    Z_support = model["Z_support"]
    ann = {k: model[k] for k in model.files if not k.startswith("scalar_") and k not in ("Z_support", "err_stage10", "err_cook")}
    ann["activation"] = str(model["scalar_activation"])
    ann["target_sum"] = float(model["scalar_target_sum"])
    ann["n_layers"] = int(model["scalar_n_layers"])

    gen = DirectStressGenerator()
    dof_to_dirpos = build_dof_to_dirpos(gen)

    def predict(e_gp_held, U_held):
        w_pred = eval_mawecm_ann(e_gp_held, ann)
        n_held = e_gp_held.shape[0]
        pred = np.zeros((n_held, 3), dtype=float)
        for i in range(n_held):
            c_e = per_element_reaction_force_contribution(gen, dof_to_dirpos, U_held[i], e_gp_held[i])
            pred[i] = c_e[Z_support, :].T @ w_pred[:, i]
        return pred

    U_stage10 = np.load(STAGE10_DIR / "single_run_U.npy")
    e_stage10 = np.load(STAGE10_DIR / "single_run_applied_strain.npy")
    pann = np.load(PANN_DATA)
    stress_stage10_true = np.asarray(pann["stage10_stress"], dtype=float)
    pred10 = predict(e_stage10, U_stage10)
    comp10 = component_relative_l2(pred10, stress_stage10_true)
    print(f"[component-check] Table-6 test trajectory: S11={comp10[0]:.4%}, S22={comp10[1]:.4%}, S12={comp10[2]:.4%}")

    stage_a = np.load(FOM_STAGE_A_NPZ)
    stage_b = np.load(FOM_STAGE_B_NPZ)
    pred_cook = predict(stage_a["e_gp"], stage_a["U_final_all"])
    comp_cook = component_relative_l2(pred_cook, stage_b["stress_rf"])
    print(f"[component-check] FOM-FE2 Cook states: S11={comp_cook[0]:.4%}, S22={comp_cook[1]:.4%}, S12={comp_cook[2]:.4%}")
    gen.close()


if __name__ == "__main__":
    main()
