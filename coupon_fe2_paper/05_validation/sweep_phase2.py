#!/usr/bin/env python3
"""MAW-ECM as the actual algorithm: prune the support, then fit the field.

WHAT WAS WRONG BEFORE. Every "MAW" number reported so far used the CLASSIC
ECM's own support with an adaptive weight field laid on top. That is not
MAW-ECM; it is ECM with variable weights. The support was never chosen by
adaptive-weight pruning, which is the part of the method that is supposed to
reach point counts a fixed-weight rule cannot.

THE RECIPE, as specified:

    classic ECM at tol 1e-4  ->  candidates (75 stress / 88 residual)
    phase 2, graph-regularized (alpha = 1e4), active FROM THE FIRST ITERATION
                             ->  30, 25, 20, 15, 10

Phase 1 has no awareness of smoothness across neighbouring states, so the whole
elimination is put through the regularized phase, which sees the manifold
structure via a k-NN graph Laplacian over q. This is requested through the
library's own `smooth_laplacian_all_iterations` flag rather than by setting
`phase1_stop_size` to the candidate count, which would only imitate the effect.

Each target is pruned independently from the same 75 candidates, since the
pruning returns only its final support. At every target the MAW support is
compared against the classic ECM at the SAME point count, which is the only
comparison that isolates what the pruning buys.
"""
from __future__ import annotations

import sys
import time

import numpy as np

import maw_lab as L

CAND_TOL = 1.0e-4
SMOOTH_ALL = True     # phase 2 from the first iteration, no phase 1
TARGETS = (30, 25, 20, 15, 10)
ALPHA_SMOOTH = 1.0e4
N_CAND_TRY = 20
GRAPH_SUB = 500
HIDDEN = (128, 128, 128)
WARM_EPOCHS = 5000
EPOCHS = 20000
PATIENCE = 4000
LR = 1.5e-3


def rank_for(sv, tol):
    tail = np.cumsum(sv[::-1] ** 2)[::-1]
    total = np.sum(sv ** 2)
    for i in range(1, sv.size + 1):
        if np.sqrt((tail[i] if i < sv.size else 0.0) / total) <= tol:
            return i
    return sv.size


def main():
    import argparse

    from mawecm_graph_utils_claude import build_knn_graph_laplacian
    from mawecm_pruning_claude import run_mawecm_pruning

    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", choices=("res", "sig"), default="sig")
    ap.add_argument("--targets", default=None,
                    help="comma-separated point counts; default the full sweep")
    ap.add_argument("--out", default=None, help="output npz name")
    _a = ap.parse_args()
    RULE = _a.rule
    targets = (tuple(int(t) for t in _a.targets.split(","))
               if _a.targets else TARGETS)
    outname = _a.out or f"maw_phase2_{RULE}.npz"

    d = L.load()
    q, ne = d["q"], d["ne"]
    C = d["C_res"] if RULE == "res" else d["C_sig"]
    b = L.targets(C)
    fi, vi = L.split(q.shape[0])
    sub = np.linspace(0, q.shape[0] - 1, min(q.shape[0], GRAPH_SUB), dtype=int)

    U, sv = L.integrand_svd(C, fi)
    z_cand, w_cand, _ = L.classic_ecm(C, rank_for(sv, CAND_TOL), U=U)
    print(f"rule {RULE}: classic ECM at tol {CAND_TOL:.0e} -> "
          f"{z_cand.size} candidates")
    A_cand = L.blocks(C, z_cand)
    print(f"  candidates reproduce b to median "
          f"{np.median(L.const_err(A_cand[vi], b[vi], w_cand)):.4e}")
    print(f"  pruning on {sub.size} states, phase 2 from the first iteration "
          f"(smooth_all={SMOOTH_ALL}), alpha {ALPHA_SMOOTH:g}\n", flush=True)

    Asub, bsub = A_cand[sub], b[sub]
    build_knn_graph_laplacian(q[sub], knn=8, kernel="gaussian")

    store, rows = {}, []
    for n_stop in targets:
        print(f"=== target {n_stop} points ===", flush=True)
        t0 = time.perf_counter()
        r = run_mawecm_pruning(
            A_blocks=[Asub[k] for k in range(Asub.shape[0])],
            b_blocks=[bsub[k] for k in range(bsub.shape[0])],
            z_ini=z_cand, w_ini=w_cand, q_train=q[sub],
            options=dict(verbose=False, n_stop=int(n_stop),
                         smooth_laplacian_all_iterations=SMOOTH_ALL,
                         alpha_smooth=ALPHA_SMOOTH,
                         number_of_candidates_to_try=N_CAND_TRY,
                         enforce_nonnegativity=True))
        Z = np.asarray(r["Z_support"], dtype=np.int64)
        o = np.argsort(Z)
        Z = Z[o]
        print(f"  pruned to {Z.size} points in {time.perf_counter() - t0:.0f}s",
              flush=True)

        A = L.blocks(C, Z)
        # classic ECM at the SAME point count, the honest cost-matched baseline
        zc, wc, _ = L.classic_ecm(C, Z.size - 1, U=U)
        Ac = L.blocks(C, zc)
        e_fix = np.median(L.const_err(Ac[vi], b[vi], wc))
        # a fixed-weight rule ON THE MAW SUPPORT, to separate support from weights
        wm = np.linalg.lstsq(
            np.concatenate([A[fi][:, i, :] for i in range(A.shape[1])], axis=0),
            np.concatenate([b[fi][:, i] for i in range(A.shape[1])], axis=0),
            rcond=None)[0]
        e_mawfix = np.median(L.const_err(A[vi], b[vi], np.maximum(wm, 0.0)))
        Wstar, nc = L.optimal_weights(A, b, ne)
        e_star = np.median(L.const_err(A[vi], b[vi], Wstar[:, vi]))

        m = L.fit_field(q, A, b, ne, fi, vi, hidden=HIDDEN, epochs=EPOCHS,
                        patience=PATIENCE, lr=LR, warm_target=Wstar,
                        warm_epochs=WARM_EPOCHS, label=f"{RULE}{Z.size}")
        e = np.median(L.const_err(A[vi], b[vi], m["W"][:, vi]))
        print(f"  classic ECM support, fixed w   {e_fix:.4e}")
        print(f"  MAW support, best fixed w      {e_mawfix:.4e}")
        print(f"  MAW support, oracle w(q)       {e_star:.4e}   "
              f"bound active {100.0 * nc / A.shape[0]:.1f}%")
        print(f"  MAW support, FITTED w(q)       {e:.4e}   "
              f"[{m['seconds']:.0f}s ep{m['best_epoch']}]")
        print(f"  gain vs classic at {Z.size} pts: {e_fix / e:.2f}x\n", flush=True)

        rows.append((Z.size, e_fix, e_mawfix, e_star, e))
        store[f"{RULE}_{Z.size}_z"] = Z
        store[f"{RULE}_{Z.size}_W_field"] = m["W"]
        store[f"{RULE}_{Z.size}_W_oracle"] = Wstar
        store[f"{RULE}_{Z.size}_mu"] = m["mu"]
        store[f"{RULE}_{Z.size}_sd"] = m["sd"]
        store[f"{RULE}_{Z.size}_act"] = np.array(m["act"])
        for k, v in m["state"].items():
            store[f"{RULE}_{Z.size}_net_{k}"] = v
        np.savez_compressed(L.HERE / outname, **store)

    print("\n=== SUMMARY: median relative constraint error, held-out states ===\n")
    print(f"{'pts':>4} {'classic sup':>13} {'MAW sup fixed':>15} "
          f"{'MAW oracle':>12} {'MAW fitted':>12} {'gain':>8}")
    print("-" * 70)
    for n, ef, emf, es, e in rows:
        print(f"{n:>4} {ef:>13.4e} {emf:>15.4e} {es:>12.4e} {e:>12.4e} "
              f"{ef / e:>7.2f}x")
    print("\nPHASE2_SWEEP_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
