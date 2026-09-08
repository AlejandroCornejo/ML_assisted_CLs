#!/usr/bin/env python3
"""Stage 00, mesh choice: convergence of the LOCAL stress field, not of the
homogenized average.

Why this is the criterion that should pick the mesh. C0 and the homogenized
stress are volume AVERAGES, and averages converge much faster than the fields
they average -- a general property of finite elements. The measured
convergence bears this out: 134 elements already give the homogenized response
to 0.57%. But POD and the ECM do not operate on the average. They operate on
the microscopic displacement and stress FIELDS, which concentrate at the pore
rim. So the mesh must be chosen on local-field accuracy, and a mesh finer than
C0 convergence demands is justified by that rather than by wanting more
elements to hyperreduce.

Measured quantity: the first Piola-Kirchhoff stress P = F S at every Gauss
point. P is chosen deliberately -- it is the integrand of the homogenized
output in first-order computational homogenization, P_bar = (1/|Omega_0|)
integral of P, which is the same output of interest Hernandez uses in the
MAW-ECM metamaterial benchmark, so the field being converged here is the one
the cubature rule actually has to integrate.

Statistics are all mesh-independent functionals, so meshes of different
topology are comparable: the maximum, volume-weighted percentiles, and the
volume-weighted L2 norm. The maximum is the most sensitive to rim resolution
and the L2 norm the least.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from periodic_fom import PeriodicRVE, _build  # noqa: E402
from uniaxial_envelope import uniaxial_step  # noqa: E402

GRIDS = [(0.40, 0.20), (0.30, 0.14), (0.22, 0.10), (0.14, 0.05), (0.10, 0.035)]


def pk1_field_stats(rve):
    """Statistics of ||P||_F over Gauss points, from the most recent assembly."""
    a = rve.assembler
    S = a._S_voigt                       # (ne, ng, 3) Voigt [S11, S22, S12]
    F = a._F                             # (ne, ng, 2, 2)
    St = np.empty(S.shape[:2] + (2, 2))
    St[..., 0, 0] = S[..., 0]
    St[..., 1, 1] = S[..., 1]
    St[..., 0, 1] = S[..., 2]
    St[..., 1, 0] = S[..., 2]
    P = np.matmul(F, St)
    nrm = np.sqrt(np.sum(P ** 2, axis=(-2, -1))).ravel()
    w = np.asarray(a.w_detJ, dtype=float).ravel()

    order = np.argsort(nrm)
    n_sorted, w_sorted = nrm[order], w[order]
    cdf = np.cumsum(w_sorted) / np.sum(w_sorted)
    pct = {p: float(np.interp(p, cdf, n_sorted)) for p in (0.50, 0.90, 0.99)}
    l2 = float(np.sqrt(np.sum(w * nrm ** 2) / np.sum(w)))
    return dict(max=float(nrm.max()), p99=pct[0.99], p90=pct[0.90],
                p50=pct[0.50], l2=l2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e11", type=float, default=0.20)
    a = ap.parse_args()

    from _material_law_guard_claude import true_neo_hookean_active

    rows = []
    for (sf, sh) in GRIDS:
        base, geom = _build(sf, sh, tag=f"loc_sf{sf:g}_sh{sh:g}")
        with true_neo_hookean_active():
            rve = PeriodicRVE(base, cell_area=geom["block_area"])
            E, S, _C, ok = uniaxial_step(rve, a.e11, np.array([-0.42 * a.e11, 0.0]))
            if not ok:
                print(f"  sf={sf} sh={sh}: uniaxial solve failed")
                continue
            st = pk1_field_stats(rve)
        ng = geom["n_elements"] * 3
        rows.append((geom["n_elements"], ng, st, float(S[0])))
        print(f"  {geom['n_elements']:5d} elems ({ng:5d} GPs)  "
              f"max {st['max']:.5e}  p99 {st['p99']:.5e}  p90 {st['p90']:.5e}  "
              f"p50 {st['p50']:.5e}  L2 {st['l2']:.5e}  S11 {S[0]:.5e}")

    if len(rows) < 2:
        return 1
    ref = rows[-1]
    print(f"\nrelative error vs the finest mesh ({ref[0]} elements), at E11={a.e11}:")
    print(f"  {'elems':>6} {'GPs':>6} {'max':>10} {'p99':>10} {'p90':>10} "
          f"{'p50':>10} {'L2':>10} {'S11 (avg)':>10}")
    for (ne, ng, st, s11) in rows:
        e = {k: abs(st[k] - ref[2][k]) / abs(ref[2][k]) for k in st}
        es = abs(s11 - ref[3]) / abs(ref[3])
        print(f"  {ne:6d} {ng:6d} {e['max']:10.2e} {e['p99']:10.2e} "
              f"{e['p90']:10.2e} {e['p50']:10.2e} {e['l2']:10.2e} {es:10.2e}")
    print("\nThe rightmost column is the homogenized AVERAGE; the columns to "
          "its left are the LOCAL field the cubature rule must integrate. If "
          "the average converges well before the local maximum does, the mesh "
          "must be chosen on the local field.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
