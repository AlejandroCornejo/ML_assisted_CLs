#!/usr/bin/env python3
"""Verification of DHpromAnnDirectLaw.evaluate_with_tangent's consistent
(analytic, non-finite-difference) tangent, same discipline as
verify_dhprom_ann_tangent_claude.py (same test states, same first-order
check) plus two checks specific to a from-scratch analytic derivation:

0. Internal consistency: evaluate_with_tangent's own (hom_eps, hom_sig)
   must match the already-verified evaluate()'s output essentially to
   machine precision -- if not, the new method's value path has already
   diverged from the trusted one and the tangent cannot be trusted either.
1. Independent first-order check: S(E+dE) from a FRESH evaluate() call
   (never touched by evaluate_with_tangent) vs. S(E) + CC_consistent @ dE,
   at the same E's and |dE| magnitudes as the FD-tangent verification.
2. Direct comparison against the central-finite-difference tangent
   (dhprom_ann_pk2_2d_vectorized, h=1e-4) at every test state, to see
   where the two agree/disagree -- particularly at the large-strain state
   where the FD tangent was already found to carry a 28-42% first-order
   error.
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

from dhprom_ann_direct_law_claude import DHpromAnnDirectLaw, dhprom_ann_pk2_2d_vectorized
import dhprom_ann_direct_law_claude as law_module


def main():
    print("[verify-consistent-tangent] building DHpromAnnDirectLaw ...")
    law = DHpromAnnDirectLaw()
    law_module._DEFAULT_LAW = law  # reuse the same instance for dhprom_ann_pk2_2d_vectorized (FD)

    test_states = [
        np.array([0.05, 0.0, 0.0]),
        np.array([0.3, -0.1, 0.05]),
        np.array([0.8, 0.4, -0.05]),
        np.array([1.2, 0.6, 0.06]),
    ]

    print("\n[verify-consistent-tangent] (0) internal value consistency vs. evaluate():")
    all_ok_values = True
    for E in test_states:
        eps_ref, sig_ref = law.evaluate(E)
        eps_new, sig_new, dEps_dE, dSig_dE = law.evaluate_with_tangent(E)
        err_eps = np.linalg.norm(eps_new - eps_ref) / max(np.linalg.norm(eps_ref), 1e-30)
        err_sig = np.linalg.norm(sig_new - sig_ref) / max(np.linalg.norm(sig_ref), 1e-30)
        ok = err_eps < 1e-8 and err_sig < 1e-8
        all_ok_values = all_ok_values and ok
        print(f"    E={E}: rel_err eps={err_eps:.3e}, sig={err_sig:.3e} [{'ok' if ok else 'FAIL'}]")
    print(f"    -> {'PASS' if all_ok_values else 'FAIL'}")
    if not all_ok_values:
        print("[verify-consistent-tangent] ABORTING: value path already disagrees with evaluate(); "
              "tangent checks below would not be meaningful.")
        return

    print("\n[verify-consistent-tangent] (1) independent first-order check, "
          "S(E+dE) vs S(E)+CC_consistent@dE:")
    all_ok = True
    for E in test_states:
        _, sig0, _, CC = law.evaluate_with_tangent(E)
        for dE_mag in (1.0e-3, 1.0e-2):
            rng = np.random.default_rng(0)
            dE = rng.standard_normal(3) * dE_mag
            _, sig_actual = law.evaluate(E + dE)
            sig_predicted = sig0 + CC @ dE
            rel_err = np.linalg.norm(sig_actual - sig_predicted) / max(np.linalg.norm(sig_actual - sig0), 1e-30)
            tag = "ok" if rel_err < 0.15 else "HIGH"
            if rel_err >= 0.15:
                all_ok = False
            print(f"    E={E}, |dE|~{dE_mag:.0e}: rel. first-order error={rel_err:.3e} [{tag}]")

    print("\n[verify-consistent-tangent] (1b) same check with a SMALLER perturbation (1e-4), "
          "to see whether error shrinks like a true first-order remainder (~dE) or stalls "
          "(which would indicate a bug rather than truncation error):")
    for E in test_states:
        _, sig0, _, CC = law.evaluate_with_tangent(E)
        rng = np.random.default_rng(1)
        dE = rng.standard_normal(3) * 1.0e-4
        _, sig_actual = law.evaluate(E + dE)
        sig_predicted = sig0 + CC @ dE
        rel_err = np.linalg.norm(sig_actual - sig_predicted) / max(np.linalg.norm(sig_actual - sig0), 1e-30)
        print(f"    E={E}, |dE|~1e-4: rel. first-order error={rel_err:.3e}")

    print("\n[verify-consistent-tangent] (2) consistent tangent vs. central-FD tangent (h=1e-4):")
    for E in test_states:
        _, _, _, CC_consistent = law.evaluate_with_tangent(E)
        _, CC_fd = dhprom_ann_pk2_2d_vectorized(E.reshape(1, 3), h=1.0e-4)
        CC_fd = CC_fd[0]
        rel_diff = np.linalg.norm(CC_consistent - CC_fd) / max(np.linalg.norm(CC_fd), 1e-30)
        print(f"    E={E}: ||CC_consistent-CC_fd||/||CC_fd||={rel_diff:.3e}")
        print(f"      CC_consistent=\n{CC_consistent}")
        print(f"      CC_fd=\n{CC_fd}")

    print(f"\n[verify-consistent-tangent] {'PASS' if all_ok else 'CHECK'}: first-order prediction "
          f"{'stayed' if all_ok else 'did not stay'} under 15% relative error at these perturbation sizes.")


if __name__ == "__main__":
    main()
