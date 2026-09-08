#!/usr/bin/env python3
"""MAW-ECM against the classic rule at EQUAL point count, and the question of
which ingredient is doing the work.

MAW-ECM changes two things at once, and they had never been separated:

  (i)  the SUPPORT is chosen by adaptive-weight pruning rather than by the
       classic greedy ECM;
  (ii) the WEIGHTS become a field w(q) rather than one fixed vector.

If (ii) alone recovers the accuracy, the pruning machinery in (i) is not
earning its complexity. So every point count is run on BOTH supports -- the
classic ECM's own, and the MAW-pruned one -- with the same field fitted on each.

The field is trained directly on constraint satisfaction over all 4950 states;
see maw_lab for why the pruned weights are no longer used as a regression
target. Non-negativity and exact volume conservation hold by construction in
every row of the output.
"""
from __future__ import annotations

import sys
import time

import numpy as np

import maw_lab as L

PTS = (5, 10, 15, 20, 30)
CAND_RANK = dict(res=39, sig=72)      # classic supports used as MAW candidates
PHASE1_STOP = 30
ALPHA_SMOOTH = 1.0e4
N_CAND_TRY = 20
GRAPH_SUB = 500
EPOCHS = 12000
PATIENCE = 1200
HIDDEN = (128, 128, 128)


def maw_support(C, q, z_cand, w_cand, n_stop, sub):
    from mawecm_graph_utils_claude import build_knn_graph_laplacian
    from mawecm_pruning_claude import run_mawecm_pruning

    A = L.blocks(C, z_cand)[sub]
    b = L.targets(C)[sub]
    build_knn_graph_laplacian(q[sub], knn=8, kernel="gaussian")
    r = run_mawecm_pruning(
        A_blocks=[A[k] for k in range(A.shape[0])],
        b_blocks=[b[k] for k in range(b.shape[0])],
        z_ini=np.asarray(z_cand), w_ini=np.asarray(w_cand), q_train=q[sub],
        options=dict(verbose=False, n_stop=int(n_stop),
                     phase1_stop_size=max(PHASE1_STOP, int(n_stop)),
                     alpha_smooth=ALPHA_SMOOTH,
                     number_of_candidates_to_try=N_CAND_TRY,
                     enforce_nonnegativity=True))
    Z = np.asarray(r["Z_support"], dtype=np.int64)
    o = np.argsort(Z)
    return Z[o], np.asarray(r["W_support"])[o]


def main():
    d = L.load()
    q, ne = d["q"], d["ne"]
    fi, vi = L.split(q.shape[0])
    sub = np.linspace(0, q.shape[0] - 1, min(q.shape[0], GRAPH_SUB), dtype=int)
    print(f"states {q.shape[0]}  ({fi.size} fit / {vi.size} held out), "
          f"pruning on {sub.size}\n", flush=True)

    rows, store = [], {}
    for nm, C in (("res", d["C_res"]), ("sig", d["C_sig"])):
        b = L.targets(C)
        z_cand, w_cand, _ = L.classic_ecm(C, CAND_RANK[nm], states=fi)
        print(f"##### {nm}: {C.shape[2]} physical rows, "
              f"{z_cand.size} MAW candidates #####", flush=True)

        for npts in PTS:
            print(f"\n--- {nm}, {npts} points ---", flush=True)
            # (a) classic support, fixed weights -- the baseline
            zc, wc, _ = L.classic_ecm(C, npts - 1, states=fi)
            Ac = L.blocks(C, zc)
            e_fix = L.report("classic ECM, fixed w", Ac, b, wc, vi)

            sups = {"classic": (zc, Ac)}
            # (b) MAW-pruned support
            if npts <= z_cand.size:
                try:
                    t0 = time.perf_counter()
                    zm, _ = maw_support(C, q, z_cand, w_cand, npts, sub)
                    Am = L.blocks(C, zm)
                    print(f"  {'MAW pruning':<34} -> {zm.size} points "
                          f"({time.perf_counter() - t0:.0f}s)", flush=True)
                    sups["maw"] = (zm, Am)
                except Exception as e:                       # noqa: BLE001
                    print(f"  MAW pruning failed: {type(e).__name__}: {e}")

            best = dict(err=e_fix, tag="classic/fixed")
            for sname, (zz, AA) in sups.items():
                m = L.fit_field(q, AA, b, ne, fi, vi, hidden=HIDDEN,
                                epochs=EPOCHS, patience=PATIENCE,
                                label=f"{nm}-{npts}-{sname}")
                e = L.report(f"{sname} support, field w(q)", AA, b, m["W"], vi,
                             extra=f"[{m['seconds']:.0f}s ep{m['best_epoch']}]")
                store[f"{nm}_{npts}_{sname}_z"] = zz
                store[f"{nm}_{npts}_{sname}_W"] = m["W"]
                for kk, vv in m["state"].items():
                    store[f"{nm}_{npts}_{sname}_net_{kk}"] = vv
                store[f"{nm}_{npts}_{sname}_mu"] = m["mu"]
                store[f"{nm}_{npts}_{sname}_sd"] = m["sd"]
                if e < best["err"]:
                    best = dict(err=e, tag=f"{sname}/field")
                rows.append((nm, npts, sname, e))
            rows.append((nm, npts, "fixed", e_fix))
            print(f"  BEST at {npts} points: {best['tag']} {best['err']:.4e}")

    print("\n\n=== SUMMARY: median relative constraint error, held-out states ===\n")
    print(f"{'rule':<5} {'pts':>4} {'classic fixed w':>17} "
          f"{'classic sup, w(q)':>19} {'MAW sup, w(q)':>15} {'gain':>8}")
    print("-" * 74)
    for nm in ("res", "sig"):
        for npts in PTS:
            g = {s: e for (n_, p_, s, e) in rows if n_ == nm and p_ == npts}
            fx = g.get("fixed", np.nan)
            cf = g.get("classic", np.nan)
            mw = g.get("maw", np.nan)
            bestf = np.nanmin([cf, mw])
            print(f"{nm:<5} {npts:>4} {fx:>17.4e} {cf:>19.4e} "
                  f"{mw:>15.4e} {fx / bestf:>7.1f}x")

    np.savez_compressed(L.HERE / "maw_sweep.npz", **store)
    print("\nMAW_SWEEP_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
