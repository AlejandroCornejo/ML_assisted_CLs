#!/usr/bin/env python3
"""Stage 00, closing items: the true uniaxial-tension path of the cell, and
mesh convergence at FINITE strain.

Why a solved path rather than an imposed one. The transverse budget was
previously estimated from the linear-limit Poisson ratio (nu_xy = 0.4718),
which is only valid near E = 0. What the coupon's gauge section actually does
is uniaxial macro STRESS, so the honest quantity is the path that satisfies

    S22(E) = 0   and   S12(E) = 0

solved for (E22, gamma12) at each E11. With shear-normal coupling present,
true uniaxial stress requires gamma12 != 0 -- the material shears under a
purely axial load, which is the whole design premise. A 2x2 Newton on
(E22, gamma12) does it, using the macro tangent already available.

Two outputs:
  * E22(E11) and gamma12(E11) -- the real envelope the RVE must survive, and
    the real transverse budget against pore closure.
  * Where the solve stops converging -- the usable tensile range.

And mesh convergence is re-checked ON this path, because C0 is a zero-strain
quantity: at finite strain the deformation localises at the pore rim, so
convergence at E = 0 is necessary but not sufficient.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from periodic_fom import PeriodicRVE, _build  # noqa: E402


def uniaxial_step(rve, e11, guess, tol=1.0e-9, max_it=20):
    """Solve S22 = S12 = 0 for (E22, gamma12) at fixed E11.

    Uses the 2x2 sub-block of the macro tangent, which is exactly the Jacobian
    of the two residuals with respect to the two unknowns."""
    E = np.array([e11, guess[0], guess[1]], dtype=float)
    for _ in range(max_it):
        S, C = rve.stress_and_tangent(E)
        r = np.array([S[1], S[2]])
        scale = max(abs(S[0]), 1.0)
        if np.linalg.norm(r) / scale < tol:
            return E, S, C, True
        J = np.array([[C[1, 1], C[1, 2]], [C[2, 1], C[2, 2]]])
        E[1:] -= np.linalg.solve(J, r)
    return E, S, C, False


def run_path(rve, e11_grid, label="", verbose=True):
    rows = []
    guess = np.array([0.0, 0.0])
    for e11 in e11_grid:
        t0 = time.perf_counter()
        try:
            E, S, C, ok = uniaxial_step(rve, e11, guess)
        except RuntimeError as exc:
            if verbose:
                print(f"  E11={e11:5.3f}  SOLVE FAILED: {exc}")
            rows.append((e11, np.nan, np.nan, np.nan, False))
            break
        if not ok:
            if verbose:
                print(f"  E11={e11:5.3f}  uniaxial Newton did not converge")
            rows.append((e11, E[1], E[2], S[0], False))
            break
        guess = E[1:].copy()
        rows.append((e11, E[1], E[2], S[0], True))
        if verbose:
            print(f"  E11={e11:5.3f}  E22={E[1]:+8.5f}  g12={E[2]:+8.5f}  "
                  f"S11={S[0]:+.4e}  nu_sec={-E[1] / e11:6.4f}  "
                  f"{time.perf_counter() - t0:5.1f}s")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("path", "mesh"), default="path")
    ap.add_argument("--e11-max", type=float, default=0.60)
    ap.add_argument("--e11-step", type=float, default=0.05)
    a = ap.parse_args()

    from _material_law_guard_claude import true_neo_hookean_active

    if a.mode == "path":
        base, geom = _build(0.30, 0.14, tag="periodic")
        print(f"mesh: {geom['n_elements']} elements, "
              f"porosity {geom['porosity_mesh'] * 100:.3f}%")
        print("\ntrue uniaxial-stress path (S22 = S12 = 0):")
        grid = np.arange(a.e11_step, a.e11_max + 1.0e-12, a.e11_step)
        with true_neo_hookean_active():
            rve = PeriodicRVE(base, cell_area=geom["block_area"])
            rows = run_path(rve, grid)
        ok_rows = [r for r in rows if r[4]]
        if ok_rows:
            last = ok_rows[-1]
            print(f"\nusable tensile range reached: E11 = {last[0]:.3f}, "
                  f"E22 = {last[1]:+.5f}, gamma12 = {last[2]:+.5f}")
            print(f"secant transverse ratio at the far end: {-last[1] / last[0]:.4f}  "
                  f"(linear-limit nu_xy was 0.4718)")
        np.savez(HERE / "uniaxial_path.npz",
                 e11=np.array([r[0] for r in rows]),
                 e22=np.array([r[1] for r in rows]),
                 g12=np.array([r[2] for r in rows]),
                 s11=np.array([r[3] for r in rows]),
                 ok=np.array([r[4] for r in rows]))
        return 0

    # mesh convergence ON the finite-strain path
    grids = [(0.40, 0.20), (0.30, 0.14), (0.22, 0.10), (0.14, 0.05)]
    probe = np.array([0.10, 0.20, 0.30])
    out = []
    for (sf, sh) in grids:
        base, geom = _build(sf, sh, tag=f"per_sf{sf:g}_sh{sh:g}")
        with true_neo_hookean_active():
            rve = PeriodicRVE(base, cell_area=geom["block_area"])
            vals = []
            for e11 in probe:
                E, S, _C, ok = uniaxial_step(rve, e11, np.array([-0.47 * e11, 0.0]))
                vals.append((E[1], E[2], S[0], ok))
        out.append((sf, sh, geom["n_elements"], vals))
        print(f"  sf={sf} sh={sh}  {geom['n_elements']:4d} elems  " + "  ".join(
            f"E11={p:.2f}: E22={v[0]:+.5f} g12={v[1]:+.5f} S11={v[2]:.5e}"
            for p, v in zip(probe, vals)))

    print("\nrelative change vs the FINEST mesh (the reference):")
    ref = out[-1][3]
    print(f"  {'elems':>6} " + " ".join(f"{'dE22':>9} {'dg12':>9} {'dS11':>9}"
                                        for _ in probe))
    for (sf, sh, ne, vals) in out:
        cells = []
        for v, r in zip(vals, ref):
            cells.append(f"{abs(v[0] - r[0]) / max(abs(r[0]), 1e-30):9.2e} "
                         f"{abs(v[1] - r[1]) / max(abs(r[1]), 1e-30):9.2e} "
                         f"{abs(v[2] - r[2]) / max(abs(r[2]), 1e-30):9.2e}")
        print(f"  {ne:6d} " + " ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
