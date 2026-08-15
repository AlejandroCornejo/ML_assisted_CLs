#!/usr/bin/env python3
"""Diagnostic, not a paper artifact: does the D-HPROM-ANN surrogate's
consistent homogenized tangent CC=dSig_hom/dE lose positive-definiteness
(or become severely ill-conditioned/asymmetric) at the actual macro Gauss
point strains Cook's membrane visits during the stalled step-1 Newton
loop? Motivated by the finding that the consistent analytic tangent
reproduces the finite-difference tangent's stall almost exactly (same
iteration-by-iteration residual trace) -- ruling out "the tangent was
merely approximate" and raising "the surrogate's true local tangent is
itself poorly conditioned there" as the next hypothesis to check.

Loads the actual 96 Gauss-point strains saved by the (non-converged)
consistent-tangent Cook run (cook_results_dhprom_consistent_claude.npz's
e_gp, computed from the LAST, still-unconverged iterate), evaluates
evaluate_with_tangent at each, and reports:
  - ||CC - CC^T||/||CC|| (symmetry, expected to hold if the surrogate
    reproduces a well-defined local energy potential -- not the same
    thing as ellipticity, but a first honesty check)
  - eigenvalues of the symmetric part sym(CC)=(CC+CC^T)/2 (loss of
    positive-definiteness here is loss of the small-strain-stability/
    rank-one-convexity type of condition the paper's own Table 8 already
    tracks for the trained PANNs, extended here to the ROM surrogate)
  - condition number of CC itself (relevant to Newton's linear solves
    regardless of symmetry)
For contrast, also reports the same three numbers at E=0 (a state deep
inside the training distribution, expected to be well-behaved).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for sub in ("core", "prom/ann", "hprom/ann"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from dhprom_ann_direct_law_claude import DHpromAnnDirectLaw


def report_one(label, E, CC):
    sym = 0.5 * (CC + CC.T)
    asym_rel = np.linalg.norm(CC - CC.T) / max(np.linalg.norm(CC), 1e-30)
    eig_sym = np.linalg.eigvalsh(sym)
    eig_raw = np.linalg.eigvals(CC)
    cond = np.linalg.cond(CC)
    print(
        f"{label}: E={E}\n"
        f"    asym_rel={asym_rel:.3e}, eig(sym)={eig_sym}, "
        f"min(eig_sym)/max(eig_sym)={eig_sym.min() / max(abs(eig_sym.max()), 1e-30):.3e}, "
        f"cond(CC)={cond:.3e}, eig(raw) real parts={np.real(eig_raw)}"
    )
    return eig_sym.min()


def main():
    print("[diagnose-gp-tangents] building DHpromAnnDirectLaw ...")
    law = DHpromAnnDirectLaw()

    print("\n--- reference state E=0 (deep inside training distribution) ---")
    _, _, _, CC0 = law.evaluate_with_tangent(np.zeros(3))
    report_one("E=0", np.zeros(3), CC0)

    print("\n--- actual Cook step-1 Gauss-point strains (from the stalled, unconverged iterate) ---")
    d = np.load(HERE / "cook_results_dhprom_consistent_claude.npz")
    e_gp = d["e_gp"]
    print(f"e_gp shape={e_gp.shape}")

    min_eigs = []
    worst_idx, worst_val = -1, np.inf
    for i, E in enumerate(e_gp):
        _, _, _, CC = law.evaluate_with_tangent(E)
        sym = 0.5 * (CC + CC.T)
        eig_sym = np.linalg.eigvalsh(sym)
        min_eigs.append(eig_sym.min())
        if eig_sym.min() < worst_val:
            worst_val = eig_sym.min()
            worst_idx = i

    min_eigs = np.array(min_eigs)
    n_negative = int(np.sum(min_eigs < 0))
    print(f"\nAcross {len(e_gp)} Gauss points: min(eig_sym) range=[{min_eigs.min():.3e}, {min_eigs.max():.3e}]")
    print(f"Gauss points with a NEGATIVE min eigenvalue (loss of positive-definiteness): {n_negative}/{len(e_gp)}")
    print(f"\nWorst Gauss point (index {worst_idx}):")
    report_one(f"worst gp[{worst_idx}]", e_gp[worst_idx], law.evaluate_with_tangent(e_gp[worst_idx])[3])

    strain_norms = np.linalg.norm(e_gp, axis=1)
    print(f"\nStrain norm range across these 96 GPs: [{strain_norms.min():.3e}, {strain_norms.max():.3e}]")
    order = np.argsort(min_eigs)
    print("5 most-negative-or-smallest-min-eigenvalue Gauss points (index, min_eig, |E|):")
    for i in order[:5]:
        print(f"    gp {i}: min_eig={min_eigs[i]:.3e}, |E|={strain_norms[i]:.3e}, E={e_gp[i]}")


if __name__ == "__main__":
    main()
