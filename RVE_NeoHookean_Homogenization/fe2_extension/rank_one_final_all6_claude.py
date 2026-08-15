#!/usr/bin/env python3
"""Single, consolidated, authoritative pass: rank-one convexity, both
methodologies (broad-synthetic 2000-sample global sweep; at-real-Cook-
nx8-visited-states, 20 directions/GP), for all six model variants, so
the paper's new six-law table cites one self-consistent source instead
of numbers stitched together from several earlier scripts run at
different points this session. D-HPROM-ANN/HPROM-ANN-iterative use the
float64, consistent-tangent laws (the "official" ones per the
project's own convention); the four PANN tiers are evaluated as-is
(never affected by the float32 precision floor).

Confirmed-correct nx=8 source files (verified by inspecting u_nodal's
max|uy| against expected converged/stalled displacement ranges):
  certified          -> cook_results_certified_nx8_claude.npz
  ickan               -> cook_results_ickan_nx8_claude.npz
  free                -> cook_results_pann_free_claude.npz   (stall-tolerant, full ramp)
  regression          -> cook_results_pann_regression_claude.npz (stall-tolerant, full ramp)
  dhprom (f64)        -> cook_results_dhprom_f64_consistent_claude.npz
  hprom_iterative(f64)-> cook_results_hprom_iterative_f64_consistent_claude.npz
NOT cook_results_free_nx8_claude.npz / cook_results_regression_nx8_claude.npz,
which are a different, old-driver run frozen at load step 1 (max|uy|~0.17).
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

NX8_FILES = {
    "regression": "cook_results_pann_regression_claude.npz",
    "free": "cook_results_pann_free_claude.npz",
    "certified": "cook_results_certified_nx8_claude.npz",
    "ickan": "cook_results_ickan_nx8_claude.npz",
    "dhprom_f64_consistent": "cook_results_dhprom_f64_consistent_claude.npz",
    "hprom_iterative_f64_consistent": "cook_results_hprom_iterative_f64_consistent_claude.npz",
}

if __name__ == "__main__":
    import pann_constitutive_law_claude as pann_law
    from dhprom_ann_direct_law_float64_claude import DHpromAnnDirectLawFloat64
    from hprom_ann_iterative_law_float64_claude import HpromAnnIterativeLawFloat64

    print("=== building law evaluators ===")
    dhprom64 = DHpromAnnDirectLawFloat64()
    hprom_iter64 = HpromAnnIterativeLawFloat64()

    def make_pann_eval(tier):
        law = pann_law.get_law(tier)

        def _eval(E_voigt, _law=law):
            S, CC = _law.pk2_and_tangent(E_voigt.reshape(1, 3))
            return S[0], CC[0]
        return _eval

    def eval_dhprom64(E_voigt):
        _eps, sig, _dEps, dSig = dhprom64.evaluate_with_tangent(E_voigt)
        return sig, dSig

    def eval_hprom_iter64(E_voigt):
        _eps, sig, _q_p, _n_it, _converged, _dEps, dSig, _eps_c, _sig_c = hprom_iter64.evaluate_with_tangent(E_voigt)
        return sig, dSig

    evaluators = {
        "regression": make_pann_eval("regression"),
        "free": make_pann_eval("free"),
        "certified": make_pann_eval("certified"),
        "ickan": make_pann_eval("ickan"),
        "dhprom_f64_consistent": eval_dhprom64,
        "hprom_iterative_f64_consistent": eval_hprom_iter64,
    }

    broad = {}
    real = {}
    for name, ev in evaluators.items():
        print(f"\n=== [broad-synthetic] {name} ===")
        broad[name] = run_audit(ev, label=f"{name} (broad)", n_samples=2000, stretch_log_range=1.0, seed=20260828)

        print(f"\n=== [at-real-nx8-states] {name} ===")
        d = np.load(HERE / NX8_FILES[name])
        print(f"    file={NX8_FILES[name]}  u_nodal max|uy|={np.abs(d['u_nodal'][:, 1]).max():.4f}")
        F_all = deformation_gradients_from_strains(d["e_gp"])
        real[name] = audit_at_real_states(F_all, ev, f"{name} (real nx=8)")

    print("\n\n=== FINAL SUMMARY: rank-one convexity, all six laws, nx=8 ===")
    print(f"{'model':32s} {'broad %':>10s} {'broad worst':>14s} {'real %':>10s} {'real worst':>14s}")
    for name in evaluators:
        b = broad[name]
        r = real[name]
        print(f"{name:32s} {b['fraction_violations']:>9.3%} {b['minimum_curvature']:>14.4e} "
              f"{r['fraction']:>9.3%} {r['worst_curvature']:>14.4e}")
