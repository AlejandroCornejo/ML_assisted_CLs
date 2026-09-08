#!/usr/bin/env python3
"""Two-phase MAW-ECM pruning for both rules, per the recipe:

    classic ECM  ->  phase 1 (no regularization) down to 30
                 ->  phase 2 (graph-regularized, alpha = 1e4) down to 10

Phase 1 has no awareness of smoothness across neighbouring states, so it is
deliberately stopped early and most of the elimination is forced through the
graph-regularized phase 2, which is slower but sees the manifold structure.

Graph: k-NN (k=8, gaussian) over q_train, and the pruning states are
subsampled to 500 for the regularized step, since the CG cost scales with
r * n_nodes.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJ = ROOT.parent / "RVE_NeoHookean_Homogenization"
for p in (str(ROOT), str(HERE), str(PROJ / "mawecm")):
    if p not in sys.path:
        sys.path.insert(0, p)

from mawecm_pruning_claude import run_mawecm_pruning          # noqa: E402
from mawecm_graph_utils_claude import build_knn_graph_laplacian  # noqa: E402

PHASE1_STOP = 30
N_STOP = 10
ALPHA_SMOOTH = 1.0e4
N_CAND_TRY = 20
GRAPH_SUB = 500

d = np.load(HERE / "maw_dataset.npz")
q_all = d["q_train"]
n_sub = min(q_all.shape[0], GRAPH_SUB)
sub = np.linspace(0, q_all.shape[0] - 1, n_sub, dtype=int)
q_train = q_all[sub]
print(f"states {q_all.shape[0]} -> {n_sub} for the graph step", flush=True)
lap = build_knn_graph_laplacian(q_train, knn=8, kernel="gaussian")

out = {}
for nm in ("res", "sig"):
    A = d[f"A_{nm}"][sub]
    b = d[f"b_{nm}"][sub]
    z, w = d[f"z_{nm}"], d[f"w_{nm}"]
    print(f"\n=== {nm}: {z.size} candidates -> phase1 {PHASE1_STOP} "
          f"-> phase2 {N_STOP} (alpha={ALPHA_SMOOTH:g}) ===", flush=True)
    t0 = time.perf_counter()
    res = run_mawecm_pruning(
        A_blocks=[A[k] for k in range(A.shape[0])],
        b_blocks=[b[k] for k in range(b.shape[0])],
        z_ini=z, w_ini=w, q_train=q_train,
        options=dict(verbose=False, n_stop=N_STOP,
                     phase1_stop_size=PHASE1_STOP,
                     alpha_smooth=ALPHA_SMOOTH,
                     number_of_candidates_to_try=N_CAND_TRY,
                     enforce_nonnegativity=True))
    dt = time.perf_counter() - t0
    W, Z = res["W_support"], res["Z_support"]
    W = np.asarray(W); Z = np.asarray(Z)
    # residual of the pruned adaptive rule, per state
    loc = np.searchsorted(z, Z) if np.all(np.diff(z) > 0) else \
        np.array([int(np.where(z == e)[0][0]) for e in Z])
    r = np.array([np.linalg.norm(A[k][:, loc] @ (W[:, k] if W.ndim == 2 and
                  W.shape[1] == A.shape[0] else W) - b[k])
                  / max(np.linalg.norm(b[k]), 1e-300) for k in range(A.shape[0])])
    print(f"  support {Z.size} elements, W shape {W.shape}, {dt:.1f}s")
    print(f"  weights min {W.min():.4e}  all >= 0: {bool(np.all(W >= -1e-9))}")
    print(f"  target reproduction: median {np.median(r):.4e}  max {r.max():.4e}")
    out[f"Z_{nm}"] = Z
    out[f"W_{nm}"] = W
np.savez_compressed(HERE / "maw_rules.npz", sub=sub, **out)
print("\nMAW_DONE")
