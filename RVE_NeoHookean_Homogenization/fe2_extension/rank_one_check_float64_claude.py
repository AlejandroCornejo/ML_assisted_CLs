#!/usr/bin/env python3
"""Re-runs both rank-one convexity audits (broad-synthetic 2000-sample,
and at-real-Cook-visited-states) for D-HPROM-ANN and HPROM-ANN-iterative
using the float64 laws instead of the float32 originals, now that the
Cook nx=8 runs actually converge (dhprom_f64, dhprom_f64_consistent,
hprom_iterative_f64_consistent all reach 20/20 converged steps, so the
"real states" query set is a genuine converged trajectory rather than a
stalled best-iterate one).

Reuses the already-validated formula/audit machinery from
rank_one_convexity_check_claude.py and rank_one_check_at_cook_states_claude.py
unchanged -- only the law being audited differs.
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

from rank_one_convexity_check_claude import run_audit  # noqa: E402
from rank_one_check_at_cook_states_claude import audit_at_real_states, deformation_gradients_from_strains  # noqa: E402

if __name__ == "__main__":
    from dhprom_ann_direct_law_float64_claude import DHpromAnnDirectLawFloat64
    from hprom_ann_iterative_law_float64_claude import HpromAnnIterativeLawFloat64

    print("=== building float64 law instances ===")
    dhprom64 = DHpromAnnDirectLawFloat64()
    hprom_iter64 = HpromAnnIterativeLawFloat64()

    def eval_dhprom64(E_voigt):
        _eps, sig, _dEps, dSig = dhprom64.evaluate_with_tangent(E_voigt)
        return sig, dSig

    def eval_hprom_iter64(E_voigt):
        _eps, sig, _q_p, _n_it, _converged, _dEps, dSig, _eps_c, _sig_c = hprom_iter64.evaluate_with_tangent(E_voigt)
        return sig, dSig

    results = {}

    print("\n=== [1/2] broad-synthetic 2000-sample audit (float64) ===")
    results["dhprom_f64_broad"] = run_audit(eval_dhprom64, label="dhprom_f64 (broad)", n_samples=2000,
                                             stretch_log_range=1.0, seed=20260828)
    results["hprom_iterative_f64_broad"] = run_audit(eval_hprom_iter64, label="hprom_iterative_f64 (broad)",
                                                       n_samples=2000, stretch_log_range=1.0, seed=20260828)

    print("\n=== [2/2] at-real-Cook-visited-states audit (float64, now genuinely converged) ===")
    d = np.load(HERE / "cook_results_dhprom_f64_consistent_claude.npz")
    F_all = deformation_gradients_from_strains(d["e_gp"])
    results["dhprom_f64_real"] = audit_at_real_states(F_all, eval_dhprom64, "dhprom_f64 (real states)")

    d = np.load(HERE / "cook_results_hprom_iterative_f64_consistent_claude.npz")
    F_all = deformation_gradients_from_strains(d["e_gp"])
    results["hprom_iterative_f64_real"] = audit_at_real_states(F_all, eval_hprom_iter64, "hprom_iterative_f64 (real states)")

    print("\n\n=== SUMMARY: rank-one convexity, float64 laws ===")
    for name, r in results.items():
        if "n_valid_paths" in r:
            print(f"{name:28s} n_valid={r['n_valid_paths']:6d}  violations={r['n_violations']:5d} "
                  f"({r['fraction_violations']:.3%})  min_curv={r['minimum_curvature']:.4e}")
        else:
            print(f"{name:28s} n_gp={r['n_gp']:5d}  n_total={r['n_total']:6d}  "
                  f"violations={r['n_violations']:5d} ({r['fraction']:.3%})  worst={r['worst_curvature']:.4e}")
