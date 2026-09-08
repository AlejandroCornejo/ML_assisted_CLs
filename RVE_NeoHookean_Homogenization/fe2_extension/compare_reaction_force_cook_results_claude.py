#!/usr/bin/env python3
"""Track B, Stage 6 (final comparison): honest S-err for the new,
unified reaction-force rule's Cook results, against
stress_correction_stage_b_fom_result_claude.npz's stress_rf -- the TRUE
energy-conjugate reaction-force reference at Cook's own 384 macro Gauss
points (built earlier this session via Track A), NOT
cook_results_fom_nested_full_claude.npz's own s_gp (confirmed this
session to be the naive-average convention natively -- though, at Cook's
own narrow-strain-range states specifically, the naive-vs-conjugate gap
there is tiny, 0.0066%, so this mostly changes methodology/rigor, not
the resulting numbers by much).

Reports both the NEW rows (this session's reaction-force rule + super-
reduced mesh) and the OLD rows (naive-average rule, backed up before
being overwritten as cook_results_*_OLD_naive_average_claude.npz) against
the SAME true reference, for a direct before/after comparison.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TRUE_REF_NPZ = HERE / "stress_correction_stage_b_fom_result_claude.npz"
FOM_NESTED_NPZ = HERE / "cook_results_fom_nested_full_claude.npz"

ROWS = [
    ("D-HPROM-ANN-FE2 (37pt classic, fixed, no ANN)",
     "cook_results_dhprom_f64_consistent_claude.npz", 296.8),
    ("D-HPROM-ANN-FE2 (10pt ANN)",
     "cook_results_dhprom_f64_consistent_NEW_10point_ann_claude.npz", 282.7),
    ("D-HPROM-ANN-FE2 (OLD: naive-average, 29-elem)",
     "cook_results_dhprom_f64_consistent_OLD_naive_average_claude.npz", 386.4),
    ("HPROM-ANN-FE2 (47pt classic, fixed, no ANN)",
     "cook_results_hprom_iterative_f64_consistent_claude.npz", 1356.7),
    ("HPROM-ANN-FE2 (20pt ANN)",
     "cook_results_hprom_iterative_f64_consistent_NEW_10point_ann_claude.npz", 1311.9),
    ("HPROM-ANN-FE2 (OLD: naive-average, 29-elem)",
     "cook_results_hprom_iterative_f64_consistent_OLD_naive_average_claude.npz", 1979.8),
    ("Linear-HPROM-FE2 (237pt res + 37pt sig, no ANN)",
     "cook_results_linear_hprom_claude.npz", 2017.6),
]


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - reference) / max(np.linalg.norm(reference), 1.0e-30))


def main() -> None:
    ref = np.load(TRUE_REF_NPZ)
    stress_rf = np.asarray(ref["stress_rf"], dtype=np.float64)
    fom = np.load(FOM_NESTED_NPZ)
    true_coords, true_tris = np.asarray(fom["coords"]), np.asarray(fom["tris"])
    true_tip = 0.5 * (float(fom["tip_uy_min_per_step"][-1]) + float(fom["tip_uy_max_per_step"][-1]))
    true_wall = None  # not re-run this session; only relative speedups vs each row's OWN paired OLD wall matter here

    print(f"True reaction-force reference: stress_rf shape={stress_rf.shape}, "
          f"true FOM tip_uy midpoint={true_tip:.4f}\n")
    print(f"{'Row':<58}{'wall(s)':>9}{'tip u_y err':>13}{'S err. (vs true RF)':>22}")

    for label, fname, wall in ROWS:
        path = HERE / fname
        if not path.exists():
            print(f"{label:<58}  [missing: {fname}]")
            continue
        d = np.load(path)
        assert np.allclose(d["coords"], true_coords), f"{label}: mesh coords mismatch"
        assert np.array_equal(d["tris"], true_tris), f"{label}: mesh connectivity mismatch"
        assert len(d["load_per_step"]) == 20, f"{label}: not a full 20-step run"
        s_gp = np.asarray(d["s_gp"], dtype=np.float64)
        row_tip = 0.5 * (float(d["tip_uy_min_per_step"][-1]) + float(d["tip_uy_max_per_step"][-1]))
        err_tip = abs(row_tip - true_tip) / abs(true_tip)
        s_err = relative_l2(s_gp, stress_rf)
        print(f"{label:<58}{wall:>8.1f}{err_tip:>12.4%}{s_err:>21.4%}")


if __name__ == "__main__":
    main()
