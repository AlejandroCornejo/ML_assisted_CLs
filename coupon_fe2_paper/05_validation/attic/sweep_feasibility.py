#!/usr/bin/env python3
"""Is an adaptive rule of this size REGRESSABLE at all? A training-free test.

The first attempts at MAW-ECM treated a bad weight field as a fitting problem
and spent the effort on losses and architectures. It is not. Measured at the
10-point residual rule:

    4892 of 4950 states hit the w >= 0 bound
    weights spanning 1e-09 .. 1.34e+03, of a total of 1546
    12.6% median change in the weight vector between NEAREST NEIGHBOURS in q

The first line is the mechanism. When non-negativity is active, the exact
solution sits on a FACE of the feasible set, and which face is active changes
with q, so the optimal weight field is piecewise, not smooth. No continuous
network can represent it, and no amount of training fixes that. The third line
is the consequence, in the units that matter for a regression: a field asked to
interpolate between neighbouring states must absorb a 12.6% jump.

So before training anything, measure -- for each rule and each point count --
how binding non-negativity is, how peaked the weights are, and how smooth the
optimal field is across the state manifold. That predicts where an adaptive
rule can work, and it costs no training at all.

Three quantities per row:

  bound%    fraction of states whose exact solution needs the w >= 0 bound
  peak      max weight / (target_sum / n_points): 1 is uniform, large is
            near-degenerate -- the rule is really using fewer points than it has
  nn jump   median relative change of the optimal weight vector between nearest
            neighbours in q, i.e. the Lipschitz load the regression must carry
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.spatial import cKDTree

import maw_lab as L

PTS = (5, 10, 15, 20, 30, 40, 60, 80)


def main():
    d = L.load()
    q, ne = d["q"], d["ne"]
    fi, vi = L.split(q.shape[0])
    _, nb = cKDTree(q).query(q, k=2)
    jj = nb[:, 1]
    print(f"states {q.shape[0]}, n_elements {ne}\n")

    for nm, C in (("res", d["C_res"]), ("sig", d["C_sig"])):
        b = L.targets(C)
        U, _ = L.integrand_svd(C, fi)
        print(f"=== {nm}: {C.shape[2]} physical rows + volume row ===")
        print(f"{'pts':>4} {'fixed w':>11} {'exact adaptive':>15} {'bound%':>8} "
              f"{'peak':>9} {'nn jump':>9}")
        print("-" * 62)
        for npts in PTS:
            z, w, _ = L.classic_ecm(C, npts - 1, U=U)
            A = L.blocks(C, z)
            e_fix = np.median(L.const_err(A, b, w))
            W, nc = L.optimal_weights(A, b, ne)
            e_ad = np.median(L.const_err(A, b, W))
            peak = W.max() / (ne / W.shape[0])
            jump = np.median(np.linalg.norm(W - W[:, jj], axis=0)
                             / np.linalg.norm(W, axis=0))
            print(f"{z.size:>4} {e_fix:>11.3e} {e_ad:>15.3e} "
                  f"{100.0 * nc / A.shape[0]:>7.1f}% {peak:>9.1f} {jump:>9.3e}",
                  flush=True)
        print()

    print("FEASIBILITY_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
