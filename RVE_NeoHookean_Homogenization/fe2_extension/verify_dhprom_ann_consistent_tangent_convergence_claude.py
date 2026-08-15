#!/usr/bin/env python3
"""Convergence-order diagnostic for the consistent tangent, disambiguating
"genuine second-order Taylor remainder" from "the derivative itself has a
bug", which a single-magnitude first-order check cannot distinguish.

For a FIXED direction u and shrinking step t, define
    resid(t) = || S(E + t*u) - S(E) - t * CC @ u ||        (absolute, no normalization)
If CC is the true derivative: S(E+t*u)-S(E) = t*CC@u + O(t^2), so
    resid(t) ~ C2 * t^2  ->  resid(t)/t -> 0 as t -> 0.
If CC carries a genuine (t-independent) error Delta = CC_used - CC_true:
    S(E+t*u)-S(E) - t*CC_used@u = -t*Delta@u + O(t^2), so
    resid(t)/t -> ||Delta@u|| = a nonzero constant, as t -> 0.

So resid(t)/t decreasing steadily towards 0 as t shrinks is the signature of
a correct tangent; resid(t)/t flattening onto a nonzero plateau is the
signature of a real bug. Stops shrinking t before evaluate()'s internal
float32 casts (torch tensors) start dominating the difference with rounding
noise rather than signal -- t is not pushed below 1e-6.
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


def main():
    print("[verify-convergence] building DHpromAnnDirectLaw ...")
    law = DHpromAnnDirectLaw()

    test_states = [
        np.array([0.05, 0.0, 0.0]),
        np.array([0.3, -0.1, 0.05]),
        np.array([0.8, 0.4, -0.05]),
        np.array([1.2, 0.6, 0.06]),
    ]
    ts = [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6]

    for E in test_states:
        _, sig0, _, CC = law.evaluate_with_tangent(E)
        rng = np.random.default_rng(42)
        u = rng.standard_normal(3)
        u = u / np.linalg.norm(u)
        pred_deriv = CC @ u
        print(f"\nE={E}, direction u={u}")
        prev_ratio = None
        for t in ts:
            _, sig_t = law.evaluate(E + t * u)
            resid = np.linalg.norm(sig_t - sig0 - t * pred_deriv)
            ratio = resid / t
            trend = ""
            if prev_ratio is not None and prev_ratio > 1e-30:
                trend = f" (x{ratio / prev_ratio:.3f} vs previous t)"
            print(f"    t={t:.0e}: resid={resid:.6e}, resid/t={ratio:.6e}{trend}")
            prev_ratio = ratio


if __name__ == "__main__":
    main()
