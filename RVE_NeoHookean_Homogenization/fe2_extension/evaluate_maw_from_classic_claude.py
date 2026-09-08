#!/usr/bin/env python3
"""Held-out validation of run_maw_from_classic_ecm_claude.py's 15-point
result (classic ECM's 27 points, refined via phase-2-only graph-regularized
pruning), using the same fixed per-state weight table's own final row
(the LAST training state's weights are NOT representative -- this rule is
per-state ADAPTIVE, unlike the classic ECM's fixed vector, so held-out
evaluation needs the same "reconstruct via nearest/interpolated weights"
question Stage 4 already handles. Since no ANN was fit here (this is a
direct phase-2 pruning result, a discrete per-training-state table, not a
continuous function), the fairest simple check is: does using the CLOSEST
training state's own weight vector generalize acceptably at held-out
states close to it? Reported alongside the fixed classic-ECM baseline for
direct comparison.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from reaction_force_ecm_target_claude import (  # noqa: E402
    DirectStressGenerator,
    build_dof_to_dirpos,
    per_element_reaction_force_contribution,
)

RESULT_NPZ = HERE / "reaction_force_ecm_pruning_from_classic_nstop10_5pct_fps_claude.npz"
STAGE10_DIR = HERE.parent / "hprom" / "ann" / "stage_10_results_maw_dynamic"
PANN_DATA = HERE.parent / "pann" / "data" / "alltraj_stage10_direct_energy.npz"
FOM_STAGE_A_NPZ = HERE / "stress_correction_stage_a_fom_result_claude.npz"
FOM_STAGE_B_NPZ = HERE / "stress_correction_stage_b_fom_result_claude.npz"


def relative_l2(pred, ref):
    return float(np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), 1.0e-30))


def main() -> None:
    d = np.load(RESULT_NPZ)
    Z_support = d["Z_support"]
    W_support = d["W_support"]  # (15, 490)
    q_train = d["q_train"]  # (490, 3)

    gen = DirectStressGenerator()
    dof_to_dirpos = build_dof_to_dirpos(gen)

    def evaluate_nearest(e_gp_held, U_held, stress_true, label):
        n_held = e_gp_held.shape[0]
        pred = np.zeros((n_held, 3), dtype=float)
        for i in range(n_held):
            nearest = np.argmin(np.sum((q_train - e_gp_held[i]) ** 2, axis=1))
            w = W_support[:, nearest]
            c_e = per_element_reaction_force_contribution(gen, dof_to_dirpos, U_held[i], e_gp_held[i])
            pred[i] = c_e[Z_support, :].T @ w
        err = relative_l2(pred, stress_true)
        print(f"[eval-maw-from-classic] HELD-OUT ({label}, n={n_held}, nearest-state weights): "
              f"relative L2 = {err:.4%}")
        return err

    U_stage10 = np.load(STAGE10_DIR / "single_run_U.npy")
    e_stage10 = np.load(STAGE10_DIR / "single_run_applied_strain.npy")
    pann = np.load(PANN_DATA)
    stress_stage10_true = np.asarray(pann["stage10_stress"], dtype=float)
    evaluate_nearest(e_stage10, U_stage10, stress_stage10_true, "Table-6 test trajectory")

    stage_a = np.load(FOM_STAGE_A_NPZ)
    stage_b = np.load(FOM_STAGE_B_NPZ)
    evaluate_nearest(stage_a["e_gp"], stage_a["U_final_all"], stage_b["stress_rf"], "FOM-FE2 Cook states")
    gen.close()

    print("[eval-maw-from-classic] for comparison, classic ECM's own fixed 27-point rule: "
          "0.16% (Table-6), 0.0071% (Cook)")


if __name__ == "__main__":
    main()
