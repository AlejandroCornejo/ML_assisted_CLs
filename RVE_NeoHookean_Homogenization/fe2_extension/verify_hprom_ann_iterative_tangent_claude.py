#!/usr/bin/env python3
"""Stage 2 tangent verification -- same two-part discipline as Stage 1
(verify_dhprom_ann_tangent_claude.py): step-size stability of the
central-difference tangent, and an independent first-order check using
fresh evaluate() calls.
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
    print("[verify-tangent] building HpromAnnIterativeLaw (mu_affine) ...")
    law = HpromAnnIterativeLaw(qp_init_mode="mu_affine")
    law_module._DEFAULT_LAW = law

    test_states = [
        np.array([0.05, 0.0, 0.0]),
        np.array([0.3, -0.1, 0.05]),
        np.array([0.8, 0.4, -0.05]),
    ]

    print("\n[verify-tangent] (1) step-size stability of the central-difference tangent:")
    for E in test_states:
        ccs = {}
        for h in (1.0e-3, 1.0e-4, 1.0e-5):
            _, CC = hprom_ann_iterative_pk2_2d_vectorized(E.reshape(1, 3), h=h)
            ccs[h] = CC[0]
        d_3_4 = np.linalg.norm(ccs[1.0e-3] - ccs[1.0e-4]) / max(np.linalg.norm(ccs[1.0e-4]), 1e-30)
        d_4_5 = np.linalg.norm(ccs[1.0e-4] - ccs[1.0e-5]) / max(np.linalg.norm(ccs[1.0e-5]), 1e-30)
        print(f"    E={E}: ||CC(1e-3)-CC(1e-4)||/||CC(1e-4)||={d_3_4:.3e}, "
              f"||CC(1e-4)-CC(1e-5)||/||CC(1e-5)||={d_4_5:.3e}")

    print("\n[verify-tangent] (2) independent first-order check, S(E+dE) vs S(E)+CC@dE:")
    all_ok = True
    for E in test_states:
        _, S0, _, _, _ = law.evaluate(E)
        _, CC = hprom_ann_iterative_pk2_2d_vectorized(E.reshape(1, 3), h=1.0e-4)
        CC = CC[0]
        for dE_mag in (1.0e-3, 1.0e-2):
            rng = np.random.default_rng(0)
            dE = rng.standard_normal(3) * dE_mag
            _, S_actual, _, _, _ = law.evaluate(E + dE)
            S_predicted = S0 + CC @ dE
            rel_err = np.linalg.norm(S_actual - S_predicted) / max(np.linalg.norm(S_actual - S0), 1e-30)
            tag = "ok" if rel_err < 0.15 else "HIGH"
            if rel_err >= 0.15:
                all_ok = False
            print(f"    E={E}, |dE|~{dE_mag:.0e}: rel. first-order error={rel_err:.3e} [{tag}]")

    print(f"\n[verify-tangent] {'PASS' if all_ok else 'CHECK'}: first-order prediction "
          f"{'stayed' if all_ok else 'did not stay'} under 15% relative error at these perturbation sizes.")


if __name__ == "__main__":
    main()
