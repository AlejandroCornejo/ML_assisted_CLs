#!/usr/bin/env python3
"""Stage 00: the effective zero-strain tangent C0 of the rotated-ellipse cell,
plus this stage's acceptance tests.

C0 comes from ONE call to the project's validated analytic-tangent path at
E = 0. That is exact, not a fit: dS/dE at zero strain IS the effective linear
stiffness, and at E = 0 the micro solve is trivial (u = 0, zero residual), so
the whole study is nearly free. No finite differences over solves anywhere.

What the numbers are for:

* C0[0,2] and C0[1,2] nonzero is the single most important check in this
  stage. If they vanish, the rotated ellipse has left the effective response
  orthotropic and the paper's group-free claim would go untested.
* nu_xy_eff = -S12/S11 (S = C0^-1) is the transverse contraction per unit
  axial strain under uniaxial macro STRESS. It already contains the plane
  strain effect, since C0 is computed in plane strain, so it is directly the
  number that sets the E22 budget against pore closure -- no conversion.
* gamma12/E11 = S13/S11 is how much shear an axial macro load produces. This
  is the quantity that makes the smooth ASTM coupon a non-trivial FE^2
  problem at all: the richness comes from the material, not from a stress
  concentration.
* The 1x1 vs 3x3 vs 5x5 comparison quantifies the affine-BC boundary layer
  (C_Neumann <= C_periodic <= C_affine), turning a reviewer's objection into
  a reported number.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJ = ROOT.parent / "RVE_NeoHookean_Homogenization"
for p in (str(ROOT), str(PROJ / "fe2_extension"), str(PROJ / "core")):
    if p not in sys.path:
        sys.path.insert(0, p)

import config as cfg  # noqa: E402
from gen_rve_mesh import build_mesh, write_mdpa  # noqa: E402


def compute_C0(n_cells=1, size_far=0.14, size_hole=0.05, porosity=None,
               aspect=None, angle_deg=None, tag=None, silence=True):
    """(C0 (3,3), S0 (3,), geom dict) for one cell block."""
    from fom_nested_consistent_law_claude import evaluate_with_tangent
    from fom_nested_consistent_warm_law_claude import _silenced

    coords, tris, outer, geom = build_mesh(
        n_cells=n_cells, porosity=porosity, aspect=aspect, angle_deg=angle_deg,
        size_far=size_far, size_hole=size_hole)
    tag = tag or f"n{n_cells}_sf{size_far:g}_sh{size_hole:g}"
    base = HERE / f"mesh_{tag}"
    write_mdpa(str(base) + ".mdpa", coords, tris, outer)

    # Denominator is the BLOCK area, void included: macro stress is force per
    # unit macro area. This overrides the solver's solid-area default.
    with _silenced(silence):
        S0, C0 = evaluate_with_tangent(
            np.zeros(3), mesh_base=str(base),
            hom_reference_measure=geom["block_area"],
            out_dir=str(HERE / f"scratch_{tag}"))
    return np.asarray(C0), np.asarray(S0), geom


def engineering_constants(C0):
    """Effective constants from C0, in plane strain, under uniaxial macro
    STRESS -- i.e. what a macro point actually experiences."""
    S = np.linalg.inv(C0)
    return dict(
        Ex=1.0 / S[0, 0], Ey=1.0 / S[1, 1],
        nu_xy=-S[0, 1] / S[0, 0],           # -E22/E11 under sigma11 only
        shear_coupling=S[0, 2] / S[0, 0],   # gamma12/E11 under sigma11 only
        C13_rel=abs(C0[0, 2]) / abs(C0[0, 0]),
        C23_rel=abs(C0[1, 2]) / abs(C0[1, 1]),
    )


def report(C0, S0, geom, label=""):
    e = engineering_constants(C0)
    asym = np.max(np.abs(C0 - C0.T)) / np.max(np.abs(C0))
    eig = np.linalg.eigvalsh(0.5 * (C0 + C0.T))
    print(f"\n--- {label} ---")
    print(f"nodes {geom['n_nodes']}  elems {geom['n_elements']}  "
          f"porosity {geom['porosity_mesh'] * 100:.3f}%  A0 {geom['block_area']:.4f}")
    with np.printoptions(precision=4, suppress=False, linewidth=120):
        print("C0 (Pa) =\n", C0)
    print(f"  symmetry            rel asym {asym:.3e}")
    print(f"  eigenvalues         {eig}")
    print(f"  |S(0)| (should ~0)  {np.linalg.norm(S0):.3e} Pa")
    print(f"  Ex, Ey              {e['Ex']:.4e}, {e['Ey']:.4e} Pa "
          f"(matrix E = {cfg.MATRIX_YOUNG:.4e})")
    print(f"  nu_xy_eff           {e['nu_xy']:.4f}   <- sets the E22 budget")
    print(f"  gamma12/E11         {e['shear_coupling']:+.4f}  <- axial load -> shear")
    print(f"  |C13|/C11           {e['C13_rel']:.4e}")
    print(f"  |C23|/C22           {e['C23_rel']:.4e}")
    return e, asym, eig


def acceptance(C0, S0, e, asym, eig):
    checks = [
        ("C0 symmetric", asym < 1.0e-8),
        ("C0 positive definite", bool(np.all(eig > 0.0))),
        ("S(0) ~ 0", float(np.linalg.norm(S0)) < 1.0e-3 * cfg.MATRIX_YOUNG),
        ("C13 nonzero (group-free)", e["C13_rel"] > 1.0e-3),
        ("C23 nonzero (group-free)", e["C23_rel"] > 1.0e-3),
    ]
    print("\nacceptance:")
    for name, good in checks:
        print(f"  [{'OK  ' if good else 'FAIL'}] {name}")
    return all(g for _, g in checks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("single", "mesh", "cells"), default="single")
    ap.add_argument("--n-cells", type=int, default=1)
    ap.add_argument("--size-far", type=float, default=0.14)
    ap.add_argument("--size-hole", type=float, default=0.05)
    ap.add_argument("--n-list", default="1,2,3")
    a = ap.parse_args()

    if a.mode == "single":
        C0, S0, geom = compute_C0(n_cells=a.n_cells, size_far=a.size_far,
                                  size_hole=a.size_hole)
        e, asym, eig = report(C0, S0, geom, f"{a.n_cells}x{a.n_cells}")
        ok = acceptance(C0, S0, e, asym, eig)
        np.savez(HERE / f"C0_n{a.n_cells}.npz", C0=C0, S0=S0,
                 **{k: v for k, v in geom.items() if k != "centres"})
        print("\nC0_STAGE00_PASS" if ok else "\nC0_STAGE00_FAIL")
        return 0 if ok else 1

    if a.mode == "mesh":
        # Coarsest mesh that converges C0 is the one to deploy: every element
        # here is paid for again at every Gauss point of the FE^2 reference.
        grid = [(0.30, 0.14), (0.22, 0.10), (0.14, 0.05), (0.10, 0.035)]
        prev = None
        rows = []
        for sf, sh in grid:
            C0, S0, geom = compute_C0(size_far=sf, size_hole=sh)
            e, _asym, _eig = report(C0, S0, geom, f"mesh sf={sf} sh={sh}")
            d = np.nan if prev is None else float(
                np.max(np.abs(C0 - prev)) / np.max(np.abs(C0)))
            rows.append((sf, sh, geom["n_elements"], e["nu_xy"],
                         e["shear_coupling"], d))
            prev = C0
        print("\n  size_far  size_hole  elems     nu_xy   g12/E11   dC0 vs coarser")
        for sf, sh, ne, nu, sc, d in rows:
            print(f"  {sf:8.3f}  {sh:9.3f}  {ne:5d}  {nu:8.4f}  {sc:+8.4f}   "
                  f"{'--' if np.isnan(d) else f'{d:.3e}'}")
        return 0

    # cells: quantify the affine-Dirichlet boundary layer
    prev = None
    rows = []
    for n in [int(v) for v in a.n_list.split(",")]:
        C0, S0, geom = compute_C0(n_cells=n, size_far=a.size_far,
                                  size_hole=a.size_hole)
        e, _asym, _eig = report(C0, S0, geom, f"{n}x{n} cells")
        d = np.nan if prev is None else float(
            np.max(np.abs(C0 - prev)) / np.max(np.abs(C0)))
        rows.append((n, geom["n_elements"], e["Ex"], e["nu_xy"],
                     e["shear_coupling"], d))
        prev = C0
    print("\n  n  elems      Ex (Pa)     nu_xy   g12/E11   dC0 vs (n-1)")
    for n, ne, Ex, nu, sc, d in rows:
        print(f"  {n}  {ne:5d}  {Ex:.5e}  {nu:8.4f}  {sc:+8.4f}   "
              f"{'--' if np.isnan(d) else f'{d:.3e}'}")
    print("\n(affine BCs over-stiffen; Ex should DECREASE toward the periodic "
          "answer as n grows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
