#!/usr/bin/env python3
"""Replicates Cook.gid/run_cook_pann_claude.py's own _rank_one_check
methodology -- Table 8 (tab:cook-rankone) in the paper -- for the three
laws that check cannot currently cover: regression (no energy potential,
the paper's own acknowledged gap; uses the paper's real nx=16 Cook data,
already saved in cook_results_regression_claude.npz), and D-HPROM-ANN /
HPROM-ANN-iterative (not part of the paper's original comparison; uses
this project's own nx=8 Cook data, the only resolution run so far).

Same protocol as _rank_one_check: 20 random rank-one directions PER
Gauss point, using the REAL deformation gradient F each model actually
visits at its own final state (not a synthetic broad sweep -- that is
rank_one_convexity_check_all6_claude.py's job, already run separately).

Uses the validated exact stress/tangent formula (rank_one_curvature_from
_stress_tangent, already checked to machine precision against the energy-
based finite-difference method _rank_one_check itself uses) instead of
finite differences -- more accurate, and the only option for the two laws
with no energy potential at all.
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

from rank_one_convexity_check_claude import rank_one_curvature_from_S_CC  # noqa: E402
from rank_one_convexity_check_claude import strain_voigt_from_F  # noqa: E402


def deformation_gradients_from_strains(e_gp):
    """e_gp (N,3) Voigt [E11,E22,gamma12] -> F (N,2,2), via the SAME
    eigendecomposition-based symmetric matrix square root used throughout
    this project (core/fom_solver_rve.py's DeformationGradientFromGreenLagrange2D)."""
    sys.path.insert(0, str(ROOT / "core"))
    from fom_solver_rve import DeformationGradientFromGreenLagrange2D
    return np.stack([DeformationGradientFromGreenLagrange2D(e) for e in e_gp], axis=0)


def audit_at_real_states(F_all, eval_S_and_CC, label, n_directions=20, seed=0):
    """Exact analogue of _rank_one_check, but using the exact stress/tangent
    curvature formula (works with or without an energy potential) instead
    of an energy finite difference. Evaluates each Gauss point's own (S,CC)
    ONCE (the base state F0/E0 is the same for all 20 directions -- only
    (a,b) differ), then reuses it across all n_directions samples at that
    point -- a 20x reduction in law evaluations versus evaluating fresh
    per (state,direction) pair, which matters for the two expensive laws
    (each evaluate_with_tangent call costs real time, unlike the near-
    instant PANN evaluations)."""
    n_gp = F_all.shape[0]
    print(f"    [{label}] evaluating (S,CC) once per Gauss point ({n_gp} states) ...")
    S_all = np.zeros((n_gp, 3), dtype=float)
    CC_all = np.zeros((n_gp, 3, 3), dtype=float)
    for g in range(n_gp):
        E0_voigt = strain_voigt_from_F(F_all[g])
        S0, CC0 = eval_S_and_CC(E0_voigt)
        S_all[g] = np.asarray(S0, dtype=float).reshape(3)
        CC_all[g] = np.asarray(CC0, dtype=float).reshape(3, 3)
        if (g + 1) % 100 == 0:
            print(f"    [{label}] {g + 1}/{n_gp} states evaluated")

    rng = np.random.default_rng(seed)
    all_curv = np.zeros((n_directions, n_gp), dtype=float)
    for d in range(n_directions):
        a = rng.standard_normal((n_gp, 2)); a /= np.linalg.norm(a, axis=1, keepdims=True)
        b = rng.standard_normal((n_gp, 2)); b /= np.linalg.norm(b, axis=1, keepdims=True)
        for g in range(n_gp):
            all_curv[d, g] = rank_one_curvature_from_S_CC(F_all[g], a[g], b[g], S_all[g], CC_all[g])
        if (d + 1) % 5 == 0:
            print(f"    [{label}] direction {d + 1}/{n_directions} done")

    flat = all_curv.reshape(-1)
    n_violations = int(np.sum(flat < 0.0))
    n_total = flat.size
    print(f"\n[{label}] {n_gp} real Gauss-point states x {n_directions} directions "
          f"({n_total} samples): {n_violations} negative ({100 * n_violations / n_total:.2f}%), "
          f"worst curvature = {flat.min():.4e}")
    return {"n_gp": n_gp, "n_directions": n_directions, "n_total": n_total,
            "n_violations": n_violations, "fraction": n_violations / n_total,
            "worst_curvature": float(flat.min())}


if __name__ == "__main__":
    results = {}

    print("=== regression (paper's own nx=16 Cook data, filling its acknowledged energy-potential gap) ===")
    import pann_constitutive_law_claude as pann_law

    d = np.load(COOK_DIR / "cook_results_regression_claude.npz")
    e_gp = d["e_gp"]
    F_all = deformation_gradients_from_strains(e_gp)
    law = pann_law.get_law("regression")

    def eval_regression(E_voigt, _law=law):
        S, CC = _law.pk2_and_tangent(E_voigt.reshape(1, 3))
        return S[0], CC[0]

    results["regression_nx16"] = audit_at_real_states(F_all, eval_regression, "regression (nx=16)")

    print("\n=== dhprom_consistent (this project's own nx=8 Cook data) ===")
    d = np.load(HERE / "cook_results_dhprom_consistent_claude.npz")
    e_gp = d["e_gp"]
    F_all = deformation_gradients_from_strains(e_gp)
    from dhprom_ann_direct_law_claude import DHpromAnnDirectLaw

    dhprom = DHpromAnnDirectLaw()

    def eval_dhprom(E_voigt):
        _eps, sig, _dEps, dSig = dhprom.evaluate_with_tangent(E_voigt)
        return sig, dSig

    results["dhprom_nx8"] = audit_at_real_states(F_all, eval_dhprom, "dhprom_consistent (nx=8)")

    print("\n=== hprom_iterative_consistent (this project's own nx=8 Cook data) ===")
    d = np.load(HERE / "cook_results_hprom_iterative_consistent_claude.npz")
    e_gp = d["e_gp"]
    F_all = deformation_gradients_from_strains(e_gp)
    from hprom_ann_iterative_law_claude import HpromAnnIterativeLaw

    hprom_iter = HpromAnnIterativeLaw()

    def eval_hprom_iter(E_voigt):
        _eps, sig, _q_p, _n_it, _converged, _dEps, dSig, _eps_check, _sig_check = hprom_iter.evaluate_with_tangent(E_voigt)
        return sig, dSig

    results["hprom_iterative_nx8"] = audit_at_real_states(F_all, eval_hprom_iter, "hprom_iterative_consistent (nx=8)")

    print("\n\n=== SUMMARY: rank-one check at real Cook-visited states (Table-8 style) ===")
    for name, r in results.items():
        print(f"{name:24s} n_gp={r['n_gp']:5d}  n_total={r['n_total']:6d}  "
              f"violations={r['n_violations']:5d} ({r['fraction']:.3%})  worst={r['worst_curvature']:.4e}")
