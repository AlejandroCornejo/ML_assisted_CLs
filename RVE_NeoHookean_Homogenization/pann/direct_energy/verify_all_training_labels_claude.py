#!/usr/bin/env python3
"""Full-coverage provenance check for the PANN training/eval labels
(pann/data/alltraj_stage10_direct_energy.npz's train_stress/stage10_stress).

Motivation: DATA_DICTIONARY.md states this data was "historically produced
by a generator script that no longer exists in this repository." Only
trajectories 1/2/6 (out of 10) and the held-out Stage-10 path had ever been
spot-checked against a fresh, independent reconstruction
(reaction_force_direct_stress.py's own main(), hprom_ann_direct_stress.py's
control). This script closes that gap: it reruns the SAME already-validated
reaction-force formula (DirectStressGenerator, imported unmodified) against
EVERY ONE of the 10 training trajectories in full (not a sample), plus the
held-out Stage-10 path, so the paper's core training data has a complete,
current, re-runnable provenance check rather than a partial one plus a
"trust the historical file" leap of faith.

Uses only data already on disk: trajectory_N_U.npy / trajectory_N_applied_
strain.npy (the raw, already-solved FOM displacement histories, saved by
the original Stage-1 generation runs and never touched since) and the
training npz's own train_stress column, as the reference to reproduce.
Nothing about the training data itself is regenerated or modified -- this
is strictly a verification pass.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
TRAJ_DIR = REPO_ROOT / "trajectories" / "stage_1_training_set_fom"
PANN_DATA = REPO_ROOT / "pann" / "data" / "alltraj_stage10_direct_energy.npz"
RESULTS_DIR = REPO_ROOT / "hprom" / "ann" / "stage_10_results_maw_dynamic"
OUT_JSON = HERE / "all_training_labels_verification_claude.json"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

from reaction_force_direct_stress import DirectStressGenerator  # noqa: E402


def relative_l2_error(computed: np.ndarray, reference: np.ndarray) -> float:
    num = np.linalg.norm(computed - reference)
    den = np.linalg.norm(reference)
    return float(num / den) if den > 0.0 else float(num)


def component_relative_l2(computed: np.ndarray, reference: np.ndarray) -> list[float]:
    out = []
    for k in range(3):
        num = np.linalg.norm(computed[:, k] - reference[:, k])
        den = np.linalg.norm(reference[:, k])
        out.append(float(num / den) if den > 0.0 else float(num))
    return out


def main() -> None:
    data = np.load(PANN_DATA)
    tid_all = data["train_trajectory_id"]
    sidx_all = data["train_source_index"]
    train_stress_all = data["train_stress"]
    train_strain_all = data["train_strain"]

    gen = DirectStressGenerator()
    results = {}
    t_start = time.perf_counter()

    for traj in range(1, 11):
        root = TRAJ_DIR / f"trajectory_{traj}"
        U = np.load(root / f"trajectory_{traj}_U.npy")
        applied_strain = np.load(root / f"trajectory_{traj}_applied_strain.npy")

        mask = tid_all == traj
        sidx = sidx_all[mask]
        order = np.argsort(sidx)
        sidx_sorted = sidx[order]
        gt_strain = train_strain_all[mask][order]
        gt_stress = train_stress_all[mask][order]

        assert np.array_equal(sidx_sorted, np.arange(U.shape[0])), (
            f"trajectory {traj}: source_index does not cover the full saved "
            f"U.npy range 0..{U.shape[0] - 1} exactly -- cannot assume full coverage."
        )
        assert np.allclose(applied_strain[sidx_sorted], gt_strain), (
            f"trajectory {traj}: ground-truth strain mismatch at mapped source index."
        )

        t0 = time.perf_counter()
        computed = gen.direct_stress_history(U, applied_strain)
        dt = time.perf_counter() - t0

        err = relative_l2_error(computed, gt_stress)
        comp_err = component_relative_l2(computed, gt_stress)
        results[f"trajectory_{traj}"] = {
            "n_states": int(U.shape[0]),
            "relative_l2": err,
            "component_relative_l2": comp_err,
            "wall_time_s": dt,
        }
        print(f"[verify-all] trajectory {traj}: n={U.shape[0]}, relative_l2={err:.6e}, "
              f"component={[f'{v:.3e}' for v in comp_err]}, {dt:.1f}s "
              f"(total elapsed {time.perf_counter() - t_start:.1f}s)", flush=True)

    # Held-out Stage-10 path (same check hprom_ann_direct_stress.py's own
    # "control" already performs, reproduced here for a single unified report).
    single_U = np.load(RESULTS_DIR / "single_run_U.npy")
    single_applied_strain = np.load(RESULTS_DIR / "single_run_applied_strain.npy")
    stage10_strain = np.asarray(data["stage10_strain"], dtype=float)
    stage10_stress = np.asarray(data["stage10_stress"], dtype=float)
    assert np.allclose(single_applied_strain, stage10_strain), (
        "single_run_applied_strain.npy does not match stage10_strain."
    )
    t0 = time.perf_counter()
    computed_stage10 = gen.direct_stress_history(single_U, single_applied_strain)
    dt = time.perf_counter() - t0
    err10 = relative_l2_error(computed_stage10, stage10_stress)
    comp_err10 = component_relative_l2(computed_stage10, stage10_stress)
    results["stage10_held_out"] = {
        "n_states": int(single_U.shape[0]),
        "relative_l2": err10,
        "component_relative_l2": comp_err10,
        "wall_time_s": dt,
    }
    print(f"[verify-all] stage10 (held-out): n={single_U.shape[0]}, relative_l2={err10:.6e}, "
          f"component={[f'{v:.3e}' for v in comp_err10]}, {dt:.1f}s", flush=True)

    gen.close()

    worst = max(results.items(), key=lambda kv: kv[1]["relative_l2"])
    print(f"\n[verify-all] DONE. Worst case: {worst[0]} at relative_l2={worst[1]['relative_l2']:.6e}. "
          f"Total wall time {time.perf_counter() - t_start:.1f}s.")

    OUT_JSON.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"[verify-all] saved to {OUT_JSON}")


if __name__ == "__main__":
    main()
