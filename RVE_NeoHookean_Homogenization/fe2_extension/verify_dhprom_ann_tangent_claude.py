#!/usr/bin/env python3
"""Stage 1 tangent verification: same discipline already used for the
PANN tangents in this project (Cook.gid/pann_constitutive_law_claude.py's
own __main__ self-test) -- check the finite-difference tangent two
independent ways:

1. Step-size stability: does the central-difference tangent stabilize
   (Cauchy-converge) as h shrinks, or is it dominated by float32 noise
   at very small h?
2. Independent first-order check: does S(E + dE), computed by a FRESH
   call to evaluate() (not reused from the tangent's own internal FD
   evaluations), actually match S(E) + CC @ dE for a small dE, at
   several different macro strain states along the held-out path?
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
    print("[verify-tangent] building DHpromAnnDirectLaw ...")
    law = DHpromAnnDirectLaw()
    law_module._DEFAULT_LAW = law  # reuse the same instance for dhprom_ann_pk2_2d_vectorized

    test_states = [
        np.array([0.05, 0.0, 0.0]),
        np.array([0.3, -0.1, 0.05]),
        np.array([0.8, 0.4, -0.05]),
        np.array([1.2, 0.6, 0.06]),
    ]

    print("\n[verify-tangent] (1) step-size stability of the central-difference tangent:")
    for E in test_states:
        ccs = {}
        for h in (1.0e-3, 1.0e-4, 1.0e-5):
            _, CC = dhprom_ann_pk2_2d_vectorized(E.reshape(1, 3), h=h)
            ccs[h] = CC[0]
        d_3_4 = np.linalg.norm(ccs[1.0e-3] - ccs[1.0e-4]) / max(np.linalg.norm(ccs[1.0e-4]), 1e-30)
        d_4_5 = np.linalg.norm(ccs[1.0e-4] - ccs[1.0e-5]) / max(np.linalg.norm(ccs[1.0e-5]), 1e-30)
        print(f"    E={E}: ||CC(1e-3)-CC(1e-4)||/||CC(1e-4)||={d_3_4:.3e}, "
              f"||CC(1e-4)-CC(1e-5)||/||CC(1e-5)||={d_4_5:.3e}")

    print("\n[verify-tangent] (2) independent first-order check, S(E+dE) vs S(E)+CC@dE:")
    all_ok = True
    for E in test_states:
        _, S0 = law.evaluate(E)
        _, CC = dhprom_ann_pk2_2d_vectorized(E.reshape(1, 3), h=1.0e-4)
        CC = CC[0]
        for dE_mag in (1.0e-3, 1.0e-2):
            rng = np.random.default_rng(0)
            dE = rng.standard_normal(3) * dE_mag
            _, S_actual = law.evaluate(E + dE)
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
