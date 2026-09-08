#!/usr/bin/env python3
"""Track B, Stage 2: build the A_blocks/b_blocks training dataset for the new
reaction-force-targeted MAW-ECM rule.

Uses Stage 1's verified per-element integrand (reaction_force_ecm_target_
claude.py). Reuses this project's own already-solved, already-saved FOM
trajectories (trajectories/stage_1_training_set_fom/) -- no new FOM solves.

Key simplification enabled by Stage 1's exact identity (sum_e c_e == the
already-validated global reaction-force stress): with w_ini = all-ones over
the candidate set, b_blocks[s] is simply DirectStressGenerator's own already-
validated direct_stress_history output at that state -- no separate target
computation needed, and w_ini trivially satisfies every constraint by
construction (same structure as the Stage 0 smoke test).

Candidate set z_ini is restricted to only the elements that touch at least
one Dirichlet dof (~156 of 990) -- every other element's c_e is identically
zero (confirmed by Stage 1's own derivation), so including them would only
waste pruning effort on candidates that can never matter.

Sampling: 50 states per training trajectory (10 trajectories -> 500 states
total, matching the ~500-state scale the paper already describes for the
existing Z_sigma construction), evenly spaced within each trajectory's own
saved history (mirrors ickan_workflow.py's sample_equally_spaced pattern
from the old codebase, reimplemented here rather than imported, since that
file lives in the old, non-self-contained directory).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
CORE_DIR = REPO_ROOT / "core"
TRAJ_DIR = REPO_ROOT / "trajectories" / "stage_1_training_set_fom"
OUT_NPZ = HERE / "reaction_force_ecm_dataset_claude.npz"

for p in (HERE, CORE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from reaction_force_ecm_target_claude import (  # noqa: E402
    DirectStressGenerator,
    build_dof_to_dirpos,
    per_element_reaction_force_contribution,
)
from ecm_sampling_utils import get_param_aware_indices  # noqa: E402

N_TRAJECTORIES = 10
# Matches this project's own original MAW-ECM dataset-construction convention
# (RVE_homogenization_NeoHookean_using_Kratos/stage12a_build_mawecm_dataset_
# gpr.py's --snapshot-percent-hom, confirmed by direct code reading:
# n_pick_hom = ceil(frac_hom * n_steps), a PERCENTAGE OF EACH TRAJECTORY'S
# OWN FULL LENGTH, with NO strain-magnitude filtering at all) and its own
# "param_aware" sampling mode (core/ecm_sampling_utils.py's
# get_param_aware_indices -- farthest-point sampling in normalized
# [E11,E22,gamma12, time_weight*step] feature space, not a plain uniform-
# in-index pick), confirmed against the original's own meta.npz
# (snapshot_percent_hom=5.0, N_s_hom=5482, sampling_mode='param_aware',
# param_aware_time_weight=0.2).
SNAPSHOT_PERCENT_HOM = 5.0
PARAM_AWARE_TIME_WEIGHT = 0.2
PARAM_AWARE_SEED = 42


def sample_equally_spaced_indices(n_total: int, n_samples: int) -> np.ndarray:
    if n_samples >= n_total:
        return np.arange(n_total, dtype=int)
    return np.linspace(0, n_total - 1, n_samples, dtype=int)


def main() -> None:
    gen = DirectStressGenerator()
    dof_to_dirpos = build_dof_to_dirpos(gen)

    # Candidate set: elements touching >=1 Dirichlet dof (every other element's
    # c_e is identically zero, per Stage 1's own derivation).
    local_dirpos = dof_to_dirpos[gen.assembler.local_eq_ids]
    touches_dirichlet = np.any(local_dirpos >= 0, axis=1)
    z_ini = np.flatnonzero(touches_dirichlet).astype(np.int64)
    n_cand = z_ini.size
    print(f"[build-rf-dataset] candidate set: {n_cand}/{gen.assembler.n_elems} elements "
          f"touch the Dirichlet boundary")

    A_blocks, b_blocks, q_train, mu_train, traj_ids, src_idx = [], [], [], [], [], []

    t0 = time.perf_counter()
    for traj in range(1, N_TRAJECTORIES + 1):
        root = TRAJ_DIR / f"trajectory_{traj}"
        U = np.load(root / f"trajectory_{traj}_U.npy")
        applied_strain = np.load(root / f"trajectory_{traj}_applied_strain.npy")
        n_steps = U.shape[0]
        n_pick = int(np.ceil((SNAPSHOT_PERCENT_HOM / 100.0) * n_steps))
        idx = get_param_aware_indices(
            applied_strain[:n_steps], n_pick,
            seed=PARAM_AWARE_SEED, time_weight=PARAM_AWARE_TIME_WEIGHT,
        )

        # Batch call reuses the already-validated global formula directly for
        # b_blocks (equal, by Stage 1's exact identity, to c_e[z_ini].sum(axis=0)).
        b_global = gen.direct_stress_history(U[idx], applied_strain[idx])

        for k, i in enumerate(idx):
            c_e = per_element_reaction_force_contribution(
                gen, dof_to_dirpos, U[i], applied_strain[i]
            )
            A_blocks.append(c_e[z_ini, :].T)  # (3, n_cand)
            b_blocks.append(b_global[k])       # (3,)
            q_train.append(applied_strain[i].copy())  # macro-strain coordinate (mu)
            mu_train.append(applied_strain[i].copy())
            traj_ids.append(traj)
            src_idx.append(int(i))

        dt = time.perf_counter() - t0
        print(f"  [trajectory {traj}] {idx.size} states sampled, {dt:.1f}s elapsed", flush=True)

    gen.close()

    q_train = np.asarray(q_train, dtype=float)
    b_blocks_arr = np.stack(b_blocks, axis=0)
    b_norms = np.linalg.norm(b_blocks_arr, axis=1)
    median_norm = float(np.median(b_norms))
    keep = b_norms >= 1.0e-3 * median_norm
    n_dropped = int(np.sum(~keep))
    if n_dropped:
        print(f"[build-rf-dataset] dropping {n_dropped} near-zero-target state(s) "
              f"(b-norm < 1e-3*median={median_norm:.3e}) -- these are the trivial, "
              f"near-zero-strain starting states of each trajectory; a target of ~0 is "
              f"satisfied by any weight choice and only breaks the pruning algorithm's "
              f"own rank-based feasibility check, contributing no useful constraint.")
        A_blocks = [A_blocks[i] for i in range(len(A_blocks)) if keep[i]]
        b_blocks = [b_blocks[i] for i in range(len(b_blocks)) if keep[i]]
        q_train = q_train[keep]
        mu_train = [mu_train[i] for i in range(len(mu_train)) if keep[i]]
        traj_ids = [traj_ids[i] for i in range(len(traj_ids)) if keep[i]]
        src_idx = [src_idx[i] for i in range(len(src_idx)) if keep[i]]

    n_nodes = q_train.shape[0]
    w_ini = np.ones(n_cand, dtype=float)

    # Verification gate: with w_ini=ones, every constraint must already be
    # satisfied exactly (Stage 1's own identity) -- spot check a handful.
    max_err = 0.0
    for j in range(0, n_nodes, max(1, n_nodes // 20)):
        pred = A_blocks[j] @ w_ini
        err = np.linalg.norm(pred - b_blocks[j]) / max(np.linalg.norm(b_blocks[j]), 1.0e-30)
        max_err = max(max_err, err)
    print(f"[build-rf-dataset] w_ini=ones constraint check, max relative error over spot "
          f"checks = {max_err:.3e}")
    assert max_err < 1.0e-9, "w_ini=ones does not satisfy the constructed constraints -- bug"

    np.savez(
        OUT_NPZ,
        z_ini=z_ini, w_ini=w_ini, q_train=q_train, mu_train=np.asarray(mu_train, dtype=float),
        traj_ids=np.asarray(traj_ids, dtype=np.int64), src_idx=np.asarray(src_idx, dtype=np.int64),
        A_blocks=np.stack(A_blocks, axis=0), b_blocks=np.stack(b_blocks, axis=0),
        n_elems_total=gen.assembler.n_elems,
    )
    print(f"[build-rf-dataset] saved {n_nodes} training states, {n_cand} candidates, to {OUT_NPZ}")


if __name__ == "__main__":
    main()
