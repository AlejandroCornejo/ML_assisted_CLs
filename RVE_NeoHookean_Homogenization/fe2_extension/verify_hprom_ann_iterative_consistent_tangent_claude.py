#!/usr/bin/env python3
"""Full verification of HpromAnnIterativeLaw.evaluate_with_tangent, same
discipline as verify_dhprom_ann_consistent_tangent_claude.py (Stage 1):

0. Internal consistency: evaluate_with_tangent's own (hom_eps, hom_sig)
   AND its internal recomputation (hom_eps_check, hom_sig_check, from the
   extra residual/Jacobian re-evaluation pass) must both match
   evaluate()'s already-verified output.
1. Independent first-order check: S(E+dE) from a FRESH evaluate() call vs.
   S(E) + CC_consistent @ dE, at the same states/magnitudes used
   throughout this project's tangent verifications.
2. Comparison against the existing central-finite-difference tangent
   (hprom_ann_iterative_pk2_2d_vectorized, h=1e-4).

dq_p/dE itself (the IFT piece) is already verified in isolation by
verify_hprom_ann_iterative_tangent_ift_claude.py (PASS, ~1e-5 to 2e-3 rel.
error against finite differences of the inner loop's own converged q_p).
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

from hprom_ann_iterative_law_claude import HpromAnnIterativeLaw, hprom_ann_iterative_pk2_2d_vectorized
import hprom_ann_iterative_law_claude as law_module


def main():
    print("[verify-consistent-tangent-stage2] building HpromAnnIterativeLaw ...")
    law = HpromAnnIterativeLaw()
    law_module._DEFAULT_LAW = law

    test_states = [
        np.array([0.05, 0.0, 0.0]),
        np.array([0.3, -0.1, 0.05]),
        np.array([0.8, 0.4, -0.05]),
        np.array([1.2, 0.6, 0.06]),
    ]

    print("\n[0] internal value consistency (evaluate_with_tangent's own value path vs. evaluate()):")
    all_ok_values = True
    results = {}
    for E in test_states:
        eps_ref, sig_ref, _, _, _ = law.evaluate(E)
        eps_new, sig_new, q_p, n_it, conv, dEps_dE, dSig_dE, eps_check, sig_check = law.evaluate_with_tangent(E)
        results[tuple(E)] = (eps_new, sig_new, dEps_dE, dSig_dE)
        err_eps = np.linalg.norm(eps_new - eps_ref) / max(np.linalg.norm(eps_ref), 1e-30)
        err_sig = np.linalg.norm(sig_new - sig_ref) / max(np.linalg.norm(sig_ref), 1e-30)
        err_eps_c = np.linalg.norm(eps_check - eps_ref) / max(np.linalg.norm(eps_ref), 1e-30)
        err_sig_c = np.linalg.norm(sig_check - sig_ref) / max(np.linalg.norm(sig_ref), 1e-30)
        ok = max(err_eps, err_sig, err_eps_c, err_sig_c) < 1e-6
        all_ok_values = all_ok_values and ok
        print(f"    E={E}: n_it={n_it}, converged={conv}, "
              f"rel_err eps={err_eps:.3e}, sig={err_sig:.3e}, "
              f"eps_check={err_eps_c:.3e}, sig_check={err_sig_c:.3e} [{'ok' if ok else 'FAIL'}]")
    print(f"    -> {'PASS' if all_ok_values else 'FAIL'}")
    if not all_ok_values:
        print("[verify-consistent-tangent-stage2] ABORTING: value path disagrees with evaluate().")
        return

    print("\n[1] independent first-order check, S(E+dE) vs S(E)+CC_consistent@dE:")
    all_ok = True
    for E in test_states:
        sig0 = results[tuple(E)][1]
        CC = results[tuple(E)][3]
        for dE_mag in (1.0e-3, 1.0e-2):
            rng = np.random.default_rng(0)
            dE = rng.standard_normal(3) * dE_mag
            _, sig_actual, _, _, _ = law.evaluate(E + dE)
            sig_predicted = sig0 + CC @ dE
            rel_err = np.linalg.norm(sig_actual - sig_predicted) / max(np.linalg.norm(sig_actual - sig0), 1e-30)
            tag = "ok" if rel_err < 0.15 else "HIGH"
            if rel_err >= 0.15:
                all_ok = False
            print(f"    E={E}, |dE|~{dE_mag:.0e}: rel. first-order error={rel_err:.3e} [{tag}]")

    print("\n[2] consistent tangent vs. central-FD tangent (h=1e-4):")
    for E in test_states:
        CC_consistent = results[tuple(E)][3]
        _, CC_fd = hprom_ann_iterative_pk2_2d_vectorized(E.reshape(1, 3), h=1.0e-4)
        CC_fd = CC_fd[0]
        rel_diff = np.linalg.norm(CC_consistent - CC_fd) / max(np.linalg.norm(CC_fd), 1e-30)
        print(f"    E={E}: ||CC_consistent-CC_fd||/||CC_fd||={rel_diff:.3e}")

    print(f"\n[verify-consistent-tangent-stage2] {'PASS' if all_ok else 'CHECK'}")


if __name__ == "__main__":
    main()
