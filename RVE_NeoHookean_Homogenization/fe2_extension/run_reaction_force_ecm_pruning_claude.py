#!/usr/bin/env python3
"""Track B, Stage 3: run the ported, unmodified run_mawecm_pruning on the new
reaction-force target dataset (Stage 2's output).

Verification gate: using the pruned support's own per-training-state weight
table directly (no regression yet), confirm it reproduces the reaction-force
stress accurately at those training states.
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
N_STOP = 15
OUT_NPZ = HERE / f"reaction_force_ecm_pruning_result_phase2only_nstop{N_STOP}_claude.npz"


def main() -> None:
    d = np.load(DATASET_NPZ)
    z_ini = d["z_ini"]
    w_ini = d["w_ini"]
    q_train = d["mu_train"]  # regress over macro-strain mu, matching the existing sig/eps targets
    A_blocks = [d["A_blocks"][i] for i in range(d["A_blocks"].shape[0])]
    b_blocks = [d["b_blocks"][i] for i in range(d["b_blocks"].shape[0])]

    print(f"[prune-rf] loaded {len(A_blocks)} states, {z_ini.size} candidates")

    # Graph-regularized phase 2 (Option B), matching the paper's own MAW-ECM
    # description exactly (k=8 kNN graph over the training states) -- without
    # this, phase 2 falls back to the LOCAL active-set variant (no smoothness
    # coupling across neighboring states at all), which fits training states
    # exactly but leaves the resulting per-state weight table free to jump
    # around between neighbors, making it hard for any continuous regressor
    # to generalize well. This was missed in the first pass (caught by the
    # held-out accuracy check in Stage 4, exactly the discipline that check
    # exists for).
    k_graph = build_knn_graph_laplacian(q_train, knn=8, kernel="gaussian")

    # phase1_stop_size caps how far the ungraph-aware phase-1 greedy pass is
    # allowed to go on its own before forcibly handing off to the graph-
    # regularized phase 2 -- explicit user instruction (this session): do not
    # trust phase 1's own choices for the bulk of the pruning, since it has
    # no smoothness-across-neighboring-states awareness at all; force most
    # of the actual candidate elimination through phase 2 instead, even
    # though that is slower.
    t0 = time.perf_counter()
    result = run_mawecm_pruning(
        A_blocks=A_blocks, b_blocks=b_blocks, z_ini=z_ini, w_ini=w_ini, q_train=q_train,
        options={
            "verbose": True, "n_stop": N_STOP,
            "smooth_laplacian_all_iterations": True,  # forces phase 2 from the very first removal, no phase 1 at all
            "use_global_graph_2ndstage": True,
            "K_graph": k_graph,
            "alpha_smooth": 0.1,
        },
    )
    dt = time.perf_counter() - t0
    print(f"[prune-rf] DONE in {dt:.1f}s")

    Z_support = result["Z_support"]
    W_support = result["W_support"]
    print(f"[prune-rf] final support size = {Z_support.size} (of {z_ini.size} candidates, "
          f"{d['n_elems_total']} total mesh elements)")
    print(f"[prune-rf] support element indices: {Z_support}")

    # Verification gate: reproduce every training state's target using the
    # pruned support's own per-state weight table directly.
    errs = np.zeros(len(A_blocks), dtype=float)
    for s in range(len(A_blocks)):
        A_full = np.zeros(z_ini.size, dtype=float)
        support_pos = np.searchsorted(z_ini, Z_support)
        pred = A_blocks[s][:, support_pos] @ W_support[:, s]
        errs[s] = np.linalg.norm(pred - b_blocks[s]) / max(np.linalg.norm(b_blocks[s]), 1.0e-30)
    print(f"[prune-rf] discrete per-state reproduction: mean={errs.mean():.3e}, "
          f"max={errs.max():.3e} relative L2 (across {len(A_blocks)} training states)")

    np.savez(OUT_NPZ, z_ini=z_ini, Z_support=Z_support, W_support=W_support,
             q_train=q_train, active_counts=result["active_counts"],
             discrete_reproduction_errs=errs)
    print(f"[prune-rf] saved to {OUT_NPZ}")


if __name__ == "__main__":
    main()
