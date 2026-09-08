#!/usr/bin/env python3
"""Stage 02: the training sample grid, derived from the stage-01 cloud.

A STRUCTURED CARTESIAN BOX with uniform spacing. Structured because the
hyperreduction machinery requires it; a box rather than anything cleverer
because measuring showed cleverness cost more than it bought.

WHY STRUCTURED AT ALL, not Sobol. Three independent requirements:

  a. MAW-ECM's weight optimization couples NEIGHBOURING sampled states through
     a graph Laplacian, and Hernandez shows accuracy degrades without it
     (alpha_G = 0 against 10^4). Uniform, well-conditioned neighbourhood
     relations are needed.
  b. The factorized decoder's latent coordinates q_M = T_m Phi_ROM^T d_red are
     identified input-informed, needing a manifold covered densely and
     uniformly INDEXED BY mu. Latent dimension r_D = 3 here, against 2 in
     Hernandez's benchmark.
  c. The affine mu -> q_p least-squares map -- the object that distinguishes
     D-HPROM-ANN, where it IS the answer, from HPROM-ANN, where it is only the
     Newton initializer -- needs mu spanning all three directions to condition.

WHY A BOX, after trying a cone. The pre-pass cloud is a thin bundle about a
dominant ray whose deviations scale linearly with load, so a load-normalized
"cone" fits it far more tightly: measured 20^3-histogram occupancy is 9.1% for
the raw box against 17.8% normalized. A cone grid was built and then rejected,
because it lost more than it gained:

  * Conditioning of the affine mu -> q_p fit, MEASURED: box 23.4, cone 57.1.
    That is the very object requirement (c) is about.
  * Uniform neighbourhoods. The cone's cells GROW with the load, so the graph
    Laplacian of requirement (a) becomes anisotropic -- and a uniform
    Cartesian grid is precisely what Hernandez uses to avoid that.
  * The cone's axis is a ray FITTED TO THE PRE-PASS material, which is known
    to be too stiff (S11 over-predicted 19.7% to 109.6%). The box depends on
    no such fit.
  * "Outside the training domain" is unambiguous for a box, which matters for
    the stage 1b probe. The cone needed three separate overshoot directions.

The cone's only advantage was density in the visited region, roughly 5x, and
density is buyable: samples here are cheap warm-started RVE solves.

RESOLUTION from a uniform ABSOLUTE step, so the graph Laplacian is isotropic.
At h = 0.014 the grid is 18 x 12 x 26 = 5616 states, comparable to the 4851 of
Hernandez's 2D benchmark. Conditioning is essentially flat in the resolution
(22.2 at h = 0.018 through 23.1 at h = 0.011): it is a property of the box's
shape, not of how finely it is sampled.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402

GRID_STEP = 0.0115   # gives ~4900 states, matching the 4851 of the reference benchmark


def load_cloud():
    d = np.load(ROOT / "01_macro_prepass" / "prepass_cloud_svk.npz")
    return np.asarray(d["cloud"], dtype=float)


def box_bounds(cloud, margin=None):
    """Cloud bounds widened so each component's total SPAN grows by the factor
    (1 + margin). That means HALF the margin on each side: applying the full
    margin to both sides would widen the span by 1 + 2*margin = 1.8 for a
    stated margin of 0.4, which is not what "40% margin" means and is not what
    was done elsewhere in this study. E11 is clipped at zero: the coupon is loaded in tension
    only, and admitting macro compression would put the cell near pore
    buckling, where its converged state is no longer a single branch and the
    warm-start exactness argument fails."""
    margin = cfg.ENVELOPE_MARGIN if margin is None else margin
    lo, hi = cloud.min(axis=0), cloud.max(axis=0)
    span = hi - lo
    blo, bhi = lo - 0.5 * margin * span, hi + 0.5 * margin * span
    blo[0] = max(blo[0], 0.0)
    return blo, bhi


def build(step=GRID_STEP):
    cloud = load_cloud()
    blo, bhi = box_bounds(cloud)
    span = bhi - blo
    n = [int(np.ceil(span[j] / step)) + 1 for j in range(3)]

    # Snap each span to an exact multiple of `step`, so the spacing is
    # EXACTLY equal in all three directions and the graph Laplacian is
    # isotropic by construction rather than to within rounding. Without this
    # the integer node counts leave an 8% spread in step size. The widening is
    # at most one cell per direction and is conservative. E11 is widened
    # upward only, its lower bound being clipped at zero.
    for j in range(3):
        extra = (n[j] - 1) * step - span[j]
        if j == 0:
            bhi[j] += extra
        else:
            blo[j] -= 0.5 * extra
            bhi[j] += 0.5 * extra
    span = bhi - blo

    axes = [np.linspace(blo[j], bhi[j], n[j]) for j in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    # The zero state, added explicitly rather than relying on 0.0 happening to
    # land on a node in all three axes.
    if not np.any(np.all(np.abs(G) < 1e-14, axis=1)):
        G = np.vstack((np.zeros((1, 3)), G))

    info = dict(blo=blo, bhi=bhi, shape=tuple(n), step=step,
                steps=tuple(float(span[j] / (n[j] - 1)) for j in range(3)),
                margin=cfg.ENVELOPE_MARGIN)
    return G, cloud, info


def main():
    G, cloud, info = build()
    print(f"box: each span widened by {(1 + info['margin']) * 100:.0f}% "
          f"({info['margin'] * 50:.0f}% of the span added on each side), "
          f"E11 clipped at 0 (tension only)")
    for j, nm in enumerate(("E11", "E22", "g12")):
        print(f"  {nm:4s} [{info['blo'][j]:+.5f}, {info['bhi'][j]:+.5f}]   "
              f"cloud [{cloud[:, j].min():+.5f}, {cloud[:, j].max():+.5f}]   "
              f"step {info['steps'][j]:.5f}")
    print(f"grid {'x'.join(map(str, info['shape']))} = "
          f"{int(np.prod(info['shape']))} states -> {G.shape[0]} with the zero state")

    inside = np.all((cloud >= info["blo"] - 1e-12)
                    & (cloud <= info["bhi"] + 1e-12), axis=1)
    cond = float(np.linalg.cond(np.column_stack([G, np.ones(G.shape[0])])))
    dup = G.shape[0] - np.unique(np.round(G, 12), axis=0).shape[0]
    step_spread = max(info["steps"]) / min(info["steps"])

    checks = [
        ("box contains the measured cloud", bool(np.all(inside)),
         f"{100 * np.mean(inside):.3f}% inside, {int(np.sum(~inside))} outside"),
        ("affine mu-fit design matrix conditioned", cond < 100.0, f"cond = {cond:.2f}"),
        ("spacing isotropic (Laplacian)", step_spread < 1.05,
         f"max/min step = {step_spread:.4f}"),
        ("no duplicate states", dup == 0, f"{dup} duplicates"),
        ("zero state present",
         bool(np.any(np.all(np.abs(G) < 1e-14, axis=1))), "E = 0 included"),
        ("tension only", float(G[:, 0].min()) >= 0.0,
         f"min E11 = {G[:, 0].min():+.5f}"),
    ]
    print("\nacceptance:")
    for name, good, detail in checks:
        print(f"  [{'OK  ' if good else 'FAIL'}] {name}  ({detail})")
    ok = all(g for _, g, _ in checks)

    np.savez(HERE / "train_grid.npz", E=G, shape=np.array(info["shape"]),
             blo=info["blo"], bhi=info["bhi"], step=info["step"],
             steps=np.array(info["steps"]))
    print("\nGRID_PASS" if ok else "\nGRID_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
