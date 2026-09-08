#!/usr/bin/env python3
"""Track B, corrected architecture: classic ECM ALWAYS runs first (already
done, run_classic_ecm_reaction_force_claude.py's own 27-point, already-
excellent result: 0.16% Table-6, 0.0071% Cook). MAW-ECM's own two phases
then start FROM that ECM output, not from the full 156-candidate pool.

This script: phase 2 only (graph-regularized, fully -- no phase-1 greedy
step at all, per explicit user instruction), pruning the classic ECM's own
27-point support down to 15, with a much stronger smoothing regularization
(alpha_smooth=1e4, vs. the earlier, much weaker 0.1) since we are now
refining an already-good starting point rather than searching from scratch.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MAWECM_DIR = HERE.parent / "mawecm"
if str(MAWECM_DIR) not in sys.path:
    sys.path.insert(0, str(MAWECM_DIR))

from mawecm_pruning_claude import run_mawecm_pruning  # noqa: E402
from mawecm_graph_utils_claude import build_knn_graph_laplacian  # noqa: E402

DATASET_NPZ = HERE / "reaction_force_ecm_dataset_claude.npz"
CLASSIC_ECM_NPZ = HERE / "classic_ecm_reaction_force_result_claude.npz"
N_STOP = 10
GRAPH_SUBSAMPLE_SIZE = 500  # fixed count, matching this project's own original
                            # "--maw-prune-graph-subsample-size 500" convention --
                            # a FRACTION of the (now much larger, 20%-of-full-length)
                            # dataset would make the graph step slow again.
ALPHA_SMOOTH = 1.0e4
N_CANDIDATES_TO_TRY = 20
OUT_NPZ = HERE / f"reaction_force_ecm_pruning_from_classic_nstop{N_STOP}_20pct_claude.npz"


def main() -> None:
    dataset = np.load(DATASET_NPZ)
    classic = np.load(CLASSIC_ECM_NPZ)

    z_ini_full = dataset["z_ini"]  # (156,) full candidate pool, actual element indices
    Z_elements = classic["Z_elements"]  # (27,) classic ECM's own selected element indices
    w_sel = classic["w_sel"]  # (27,) classic ECM's own weights

    # Map the 27 selected elements to their column positions in the full
    # 156-candidate A_blocks array.
    local_pos = np.searchsorted(z_ini_full, Z_elements)
    assert np.array_equal(z_ini_full[local_pos], Z_elements), "element index mapping mismatch"

    A_full_all = dataset["A_blocks"]  # (n_states, 3, 156)
    b_full_all = dataset["b_blocks"]  # (n_states, 3)
    q_train_all = dataset["mu_train"]
    n_states_all = A_full_all.shape[0]

    # Uniform (evenly-spaced) fixed-size subsample for the graph-regularized
    # pruning step specifically -- matches this project's own original recipe
    # (--maw-prune-graph-subsample-size 500 / -mode uniform), and directly
    # cuts free_dim = r * n_nodes (the dominant cost driver in the CG solve).
    n_sub = min(int(n_states_all), int(GRAPH_SUBSAMPLE_SIZE))
    sub_idx = np.linspace(0, n_states_all - 1, n_sub, dtype=int)
    print(f"[maw-from-classic] subsampling graph-pruning states: {n_states_all} -> {n_sub} "
          f"({100.0 * n_sub / n_states_all:.1f}%), uniform/evenly-spaced")

    A_full = A_full_all[sub_idx]
    b_full = b_full_all[sub_idx]
    q_train = q_train_all[sub_idx]
    A_blocks = [A_full[s][:, local_pos] for s in range(A_full.shape[0])]
    b_blocks = [b_full[s] for s in range(b_full.shape[0])]

    print(f"[maw-from-classic] starting from classic ECM's own {Z_elements.size}-point "
          f"support, pruning to {N_STOP} via phase-2-only (graph-regularized, no phase 1), "
          f"alpha_smooth={ALPHA_SMOOTH}, n_candidates_to_try={N_CANDIDATES_TO_TRY}")

    k_graph = build_knn_graph_laplacian(q_train, knn=8, kernel="gaussian")

    t0 = time.perf_counter()
    result = run_mawecm_pruning(
        A_blocks=A_blocks, b_blocks=b_blocks, z_ini=Z_elements, w_ini=w_sel, q_train=q_train,
        options={
            "verbose": True, "n_stop": N_STOP,
            "smooth_laplacian_all_iterations": True,
            "use_global_graph_2ndstage": True,
            "K_graph": k_graph,
            "alpha_smooth": ALPHA_SMOOTH,
            "number_of_candidates_to_try": N_CANDIDATES_TO_TRY,
        },
    )
    dt = time.perf_counter() - t0
    print(f"[maw-from-classic] DONE in {dt:.1f}s")

    Z_support = result["Z_support"]
    W_support = result["W_support"]
    print(f"[maw-from-classic] final support size = {Z_support.size}")
    print(f"[maw-from-classic] support element indices: {Z_support}")

    # Save immediately, before any further post-processing that could fail
    # and lose this (expensive, ~9 min) result.
    np.savez(OUT_NPZ, z_ini=Z_elements, Z_support=Z_support, W_support=W_support, q_train=q_train)
    print(f"[maw-from-classic] saved to {OUT_NPZ}")

    # Z_elements (classic ECM's own output) is not guaranteed sorted --
    # match by value, not by searchsorted (which requires a sorted array).
    elem_to_local = {int(z): i for i, z in enumerate(Z_elements)}
    support_local = np.array([elem_to_local[int(z)] for z in Z_support], dtype=int)
    errs = np.zeros(len(A_blocks), dtype=float)
    for s in range(len(A_blocks)):
        pred = A_blocks[s][:, support_local] @ W_support[:, s]
        errs[s] = np.linalg.norm(pred - b_blocks[s]) / max(np.linalg.norm(b_blocks[s]), 1.0e-30)
    print(f"[maw-from-classic] discrete per-state reproduction: mean={errs.mean():.3e}, "
          f"max={errs.max():.3e} relative L2 (across {len(A_blocks)} training states)")


if __name__ == "__main__":
    main()
