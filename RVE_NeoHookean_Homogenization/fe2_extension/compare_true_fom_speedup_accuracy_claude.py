#!/usr/bin/env python3
"""Once the genuine, full 20-step, nx=8 FOM-nested FE^2 run
(cook_results_fom_nested_full_claude.npz) exists, compute, against it as
the true multiscale reference, for all six other rows of Table 7:
  - speedup = true_FOM_wall_time / row_wall_time
  - tip u_y relative error, using the midpoint of the (min, max) tip-edge
    range Table 7 itself reports as "the" tip displacement (a single
    number, not min/max separately -- the two are close enough that
    reporting both is redundant precision)
  - full 384-Gauss-point final-state stress relative L2 error

Wall times and tip_uy ranges for the six rows are the already-verified
numbers already in Table 7 (tab:cook-fe2) -- not re-derived here, since
they are fixed, previously-checked results. Only the true FOM's own
number (this run's own wall time, printed by run_newton_fe2 as part of
its return dict / the launch command's own final print) is new.

IMPORTANT data-provenance note: for regression and free hyperelastic,
use cook_results_pann_{regression,free}_claude.npz, NOT
cook_results_{regression,free}_nx8_claude.npz -- the latter turned out
to be a stale, single-load-step (5%) diagnostic leftover mislabeled
with the canonical filename (load_per_step had length 1, not 20;
already deleted from fe2_extension/ once found). Using it here silently
compared the true FOM's full-load stress field against regression/free's
barely-deformed 5%-load state, producing a spurious ~95% "error" that
had nothing to do with either model's actual accuracy. Confirmed via
tip_uy_min_per_step[-1]/tip_uy_max_per_step[-1] and load_per_step's
length before trusting any npz here -- do the same for any new file
added to ROWS below.

Usage: python3 compare_true_fom_speedup_accuracy_claude.py [true_fom_wall_time_seconds]
If the wall time isn't passed on the command line, it's parsed from the
launch log's own "DONE ..." line.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LOG_PATH = Path("/tmp/claude-1000/-home-kratos-ML-assisted-CLs-clean/4a3da423-8dfc-40e5-a7fa-8fa40d3c8b2f/scratchpad/fom_nested_full20_claude.log")
TRUE_FOM_NPZ = HERE / "cook_results_fom_nested_full_claude.npz"

# (label, npz filename, wall_time_s, tip_uy_step1, tip_uy_final) -- wall
# times and tip_uy values are Table 7's own already-verified numbers.
ROWS = [
    # wall times for the four PANN tiers are the mean of 10 independent
    # repetitions (see time_4_ann_10x.py), not a single run
    ("Regression (tier 1)", "cook_results_pann_regression_claude.npz", 0.405, 3.070, 3.202),
    ("Free hyperelastic (tier 2)", "cook_results_pann_free_claude.npz", 1.295, 3.114, 3.251),
    ("Polyconvex ICNN (tier 3a)", "cook_results_icnn_w5_final_claude.npz", 2.586, 3.0979, 3.2330),
    ("Polyconvex ICKAN (tier 3b)", "cook_results_ickan_w5_final_claude.npz", 8.690, 3.1678, 3.3019),
    ("HPROM--ANN-FE$^2$", "cook_results_hprom_iterative_f64_consistent_claude.npz", 3995.1, 3.0700, 3.2019),
    ("D-HPROM--ANN-FE$^2$", "cook_results_dhprom_f64_consistent_claude.npz", 1230.0, 3.0958, 3.2284),
]


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - reference) / max(np.linalg.norm(reference), 1.0e-30))


def parse_wall_time_from_log() -> float:
    text = LOG_PATH.read_text()
    for line in reversed(text.splitlines()):
        if line.startswith("DONE"):
            # DONE <fully_converged> (<tip_lo>, <tip_hi>) <wall_time>
            return float(line.strip().split()[-1])
    raise RuntimeError(f"No 'DONE' line found yet in {LOG_PATH} -- run not finished.")


def main() -> None:
    if not TRUE_FOM_NPZ.exists():
        raise SystemExit(f"{TRUE_FOM_NPZ} does not exist yet -- the true FOM run has not finished/saved.")

    true = np.load(TRUE_FOM_NPZ)
    true_s_gp = np.asarray(true["s_gp"], dtype=np.float64)
    true_coords = np.asarray(true["coords"])
    true_tris = np.asarray(true["tris"])
    if "tip_uy_min_per_step" in true:
        tip_min = float(true["tip_uy_min_per_step"][-1])
        tip_max = float(true["tip_uy_max_per_step"][-1])
    else:
        # fall back to the DONE line's own (min, max), parsed from the log
        text = LOG_PATH.read_text()
        done_line = next(l for l in reversed(text.splitlines()) if l.startswith("DONE"))
        # DONE True (min, max) wall
        paren = done_line[done_line.index("(") + 1: done_line.index(")")]
        tip_min, tip_max = (float(x) for x in paren.split(","))

    true_wall = float(sys.argv[1]) if len(sys.argv) > 1 else parse_wall_time_from_log()
    true_tip = 0.5 * (tip_min + tip_max)

    print(f"True FOM (genuine FE2): wall={true_wall:.1f}s, tip_uy midpoint={true_tip:.4f} (range {tip_min:.4f}-{tip_max:.4f})\n")
    print(f"{'Model':<28}{'Speedup':>10}{'Tip u_y err':>14}{'Final-state S rel.L2':>22}")

    for label, fname, wall, row_min, row_max in ROWS:
        d = np.load(HERE / fname)
        assert np.allclose(d["coords"], true_coords), f"{label}: mesh coords mismatch vs true FOM"
        assert np.array_equal(d["tris"], true_tris), f"{label}: mesh connectivity mismatch vs true FOM"
        assert len(d["load_per_step"]) == 20, \
            f"{label}: {fname} has {len(d['load_per_step'])} load steps, not 20 -- stale/partial file, do not use"
        s_gp = np.asarray(d["s_gp"], dtype=np.float64)

        row_tip = 0.5 * (row_min + row_max)
        speedup = true_wall / wall
        err_tip = abs(row_tip - true_tip) / abs(true_tip)
        s_err = relative_l2(s_gp, true_s_gp)

        print(f"{label:<28}{speedup:>9.1f}x{err_tip:>13.4%}{s_err:>21.4%}")


if __name__ == "__main__":
    main()
