#!/usr/bin/env python3
"""Runs the same validated stress/tangent-only rank-one convexity audit
(rank_one_convexity_check_claude.py, validated to machine precision
against the energy-based double-autograd method on 'certified') uniformly
across all 6 material-law variants compared throughout this project's
fe2_extension work: the four PANN tiers (certified, ickan, free,
regression) and the two ROM/HPROM laws (D-HPROM-ANN, HPROM-ANN-iterative).

For certified/ickan/free (which have an energy potential), this is a
cross-check against the paper's own energy-based audit
(evaluate_free_ellipticity_claude.py) -- should agree closely if
everything here is consistent with it. For regression (no energy
potential, same limitation as the two ROM/HPROM laws) this audit provides
information the paper's own energy-based method could not compute at all.

Same sampling protocol throughout (log-uniform stretches, range 1.0;
shear entries uniform(-0.4,0.4); n_samples=2000; seed=20260828) so all six
results are directly comparable to each other and to the paper's own
free-tier numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COOK_DIR = ROOT / "Cook.gid"
for p in (str(ROOT / "core"), str(COOK_DIR), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from rank_one_convexity_check_claude import run_audit  # noqa: E402

if __name__ == "__main__":
    import pann_constitutive_law_claude as pann_law  # noqa: E402

    results = {}

    for tier in ("certified", "ickan", "free", "regression"):
        law = pann_law.get_law(tier)

        def eval_pann(E_voigt, _law=law):
            S, CC = _law.pk2_and_tangent(E_voigt.reshape(1, 3))
            return S[0], CC[0]

        results[tier] = run_audit(eval_pann, label=f"pann_{tier}", n_samples=2000,
                                   stretch_log_range=1.0, seed=20260828)

    from dhprom_ann_direct_law_claude import DHpromAnnDirectLaw

    dhprom = DHpromAnnDirectLaw()

    def eval_dhprom(E_voigt):
        _eps, sig, _dEps, dSig = dhprom.evaluate_with_tangent(E_voigt)
        return sig, dSig

    results["dhprom"] = run_audit(eval_dhprom, label="dhprom", n_samples=2000,
                                   stretch_log_range=1.0, seed=20260828)

    from hprom_ann_iterative_law_claude import HpromAnnIterativeLaw

    hprom_iter = HpromAnnIterativeLaw()

    def eval_hprom_iter(E_voigt):
        _eps, sig, _q_p, _n_it, _converged, _dEps, dSig, _eps_check, _sig_check = hprom_iter.evaluate_with_tangent(E_voigt)
        return sig, dSig

    results["hprom_iterative"] = run_audit(eval_hprom_iter, label="hprom_iterative", n_samples=2000,
                                            stretch_log_range=1.0, seed=20260828)

    print("\n\n=== SUMMARY: rank-one convexity audit, all 6 laws, 2000 samples each ===")
    print(f"{'law':20s} {'n_valid':>8s} {'n_violations':>13s} {'fraction':>10s} {'min_curv':>14s}")
    for name, r in results.items():
        print(f"{name:20s} {r['n_valid_paths']:8d} {r['n_violations']:13d} "
              f"{r['fraction_violations']:10.3%} {r['minimum_curvature']:14.4e}")
