#!/usr/bin/env python3
"""Stage 01, closing item: macro mesh convergence of the strain CLOUD.

Two outputs, and the second one matters as much as the first:

1. Whether the envelope is mesh-converged. The cloud bounds are what stage 02
   turns into a sampling domain, so they are the quantity that has to converge
   -- not the displacement or the reaction.

2. The macro mesh for the FE^2 runs. Every macro Gauss point costs a full
   nonlinear RVE solve there, so the right macro mesh is the COARSEST one that
   resolves the cloud. The pre-pass mesh currently carries 6048 Gauss points;
   if a coarser mesh gives the same envelope, the FE^2 reference gets cheaper
   in direct proportion.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as cfg  # noqa: E402
from gen_coupon_mesh import build_mesh  # noqa: E402
from macro_prepass import (MacroCoupon, calibrate_delta, load_C0,  # noqa: E402
                           svk_material, write_coupon_mdpa)

TARGET_E11 = 0.15
N_STEPS = 10
DIVISORS = (3.0, 4.0, 6.0, 8.0, 12.0)


def bounds(cloud):
    out = {}
    for j, nm in enumerate(("E11", "E22", "g12")):
        c = cloud[:, j]
        out[nm] = (float(c.min()), float(np.percentile(c, 1)),
                   float(np.percentile(c, 99)), float(c.max()))
    return out


def main():
    C0 = load_C0()
    rows = []
    for div in DIVISORS:
        size = cfg.COUPON_W_GAUGE / div
        coords, tris, left, right, geom = build_mesh(size_gauge=size)
        base = HERE / f"coupon_conv_div{div:g}"
        write_coupon_mdpa(str(base) + ".mdpa", coords, tris,
                          np.concatenate((left, right)))
        with svk_material(C0):
            m = MacroCoupon(base)
            delta, got = calibrate_delta(m, TARGET_E11, N_STEPS, verbose=False)
            _u, cloud = m.solve(delta, n_steps=N_STEPS)
        b = bounds(cloud)
        ngp = geom["n_elements"] * 3
        rows.append((div, size, geom["n_elements"], ngp, delta, b))
        print(f"  W/{div:g} (h={size * 1e3:5.3f}mm)  {geom['n_elements']:5d} elems  "
              f"{ngp:6d} GPs  delta={delta * 1e3:7.4f}mm  "
              f"E11max={b['E11'][3]:+.5f}  E22min={b['E22'][0]:+.5f}  "
              f"g12min={b['g12'][0]:+.5f}  g12max={b['g12'][3]:+.5f}", flush=True)

    ref = rows[-1][5]
    print(f"\nrelative error of the cloud bounds vs the finest mesh "
          f"(W/{DIVISORS[-1]:g}, {rows[-1][3]} GPs):")
    hdr = "  " + f"{'mesh':>8} {'elems':>6} {'GPs':>7}"
    for nm in ("E11", "E22", "g12"):
        hdr += f" {nm + '.min':>9} {nm + '.max':>9}"
    print(hdr)
    for (div, size, ne, ngp, delta, b) in rows:
        line = f"  {'W/' + f'{div:g}':>8} {ne:6d} {ngp:7d}"
        for nm in ("E11", "E22", "g12"):
            for k in (0, 3):
                r = abs(b[nm][k] - ref[nm][k]) / max(abs(ref[nm][k]), 1e-30)
                line += f" {r:9.2e}"
        print(line)
    print("\nThe FE^2 macro mesh should be the coarsest row whose cloud bounds "
          "are converged: every macro Gauss point there is a full nonlinear "
          "RVE solve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
