#!/usr/bin/env python3
"""Diagnostic, not a paper artifact: do pann_free/pann_regression -- the
two PANN tiers found to stall (bounded, not diverge) at Cook's nx=8 mesh,
unlike the polyconvex-by-construction certified/ickan tiers -- show a
LOCAL loss of positive-definiteness (material instability) at the actual
macro Gauss-point strains their own stalled Cook runs visit? Complements
the architectural observation (certified/ickan are polyconvex by
construction; free/regression are not) with a direct numerical check,
mirroring diagnose_cook_gp_tangents_claude.py's earlier check on
D-HPROM-ANN (which found no such loss there).

For "free" (has an energy potential -> CC is a genuine, already-
symmetrized Hessian by pann_constitutive_law_claude.py's own code):
eigenvalues of CC itself are the physically meaningful quantity.
For "regression" (no energy potential -> CC is a plain, not-necessarily-
symmetric stress Jacobian, per that file's own docstring): eigenvalues of
CC's symmetric part are reported as the closest analogous numerical
check, but do not carry the same "loss of material stability" reading --
noted explicitly in the output rather than silently treated the same way.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COOK_DIR = ROOT / "Cook.gid"
for p in (str(ROOT / "core"), str(COOK_DIR), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pann_constitutive_law_claude as pann_law  # noqa: E402


def report(which, has_potential):
    d = np.load(HERE / f"cook_results_pann_{which}_claude.npz")
    e_gp = d["e_gp"]
    S, CC = pann_law.pann_pk2_2d_vectorized(e_gp, which=which)

    strain_norms = np.linalg.norm(e_gp, axis=1)
    print(f"\n--- {which} ({'has' if has_potential else 'NO'} energy potential) ---")
    print(f"strain norm range across {len(e_gp)} GPs: [{strain_norms.min():.3e}, {strain_norms.max():.3e}]")

    min_eigs = np.zeros(len(e_gp))
    asym_rel = np.zeros(len(e_gp))
    for i in range(len(e_gp)):
        cc = CC[i]
        asym_rel[i] = np.linalg.norm(cc - cc.T) / max(np.linalg.norm(cc), 1e-30)
        sym = 0.5 * (cc + cc.T)
        min_eigs[i] = np.linalg.eigvalsh(sym).min()

    n_negative = int(np.sum(min_eigs < 0))
    print(f"asymmetry ||CC-CC^T||/||CC|| range: [{asym_rel.min():.3e}, {asym_rel.max():.3e}]")
    print(f"min(eig_sym) range: [{min_eigs.min():.3e}, {min_eigs.max():.3e}]")
    print(f"Gauss points with a NEGATIVE min eigenvalue: {n_negative}/{len(e_gp)}")
    if not has_potential:
        print("NOTE: no energy potential exists for this tier, so eig(sym(CC)) is a numerical-"
              "conditioning check only, not a material-stability/loss-of-ellipticity statement.")

    order = np.argsort(min_eigs)
    print("5 smallest-min-eigenvalue Gauss points (index, min_eig, |E|):")
    for i in order[:5]:
        print(f"    gp {i}: min_eig={min_eigs[i]:.3e}, |E|={strain_norms[i]:.3e}, E={e_gp[i]}")

    cond = np.array([np.linalg.cond(CC[i]) for i in order[:5]] + [np.linalg.cond(CC[i]) for i in order[-5:]])
    print(f"cond(CC) at 5 worst + 5 best min-eig points: min={cond.min():.3e}, max={cond.max():.3e}")


if __name__ == "__main__":
    report("free", has_potential=True)
    report("regression", has_potential=False)
