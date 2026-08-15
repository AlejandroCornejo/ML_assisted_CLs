#!/usr/bin/env python3
"""Same convergence-order diagnostic as
verify_dhprom_ann_consistent_tangent_convergence_claude.py (Stage 1),
applied to Stage 2's one first-order-check outlier (E=[0.8,0.4,-0.05],
33% at |dE|~1e-2): does resid(t)/t shrink towards 0 as t shrinks (genuine
second-order Taylor remainder -> tangent is correct) or plateau at a
nonzero constant (a real, t-independent tangent error)? See that script's
docstring for the full derivation of why this distinguishes the two.
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

from hprom_ann_iterative_law_claude import HpromAnnIterativeLaw


def main():
    print("[verify-convergence-stage2] building HpromAnnIterativeLaw ...")
    law = HpromAnnIterativeLaw()

    test_states = [
        np.array([0.05, 0.0, 0.0]),
        np.array([0.8, 0.4, -0.05]),
        np.array([1.2, 0.6, 0.06]),
    ]
    ts = [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6]

    for E in test_states:
        _, sig0, _, _, _, _, CC, _, _ = law.evaluate_with_tangent(E)
        rng = np.random.default_rng(42)
        u = rng.standard_normal(3)
        u = u / np.linalg.norm(u)
        pred_deriv = CC @ u
        print(f"\nE={E}, direction u={u}")
        prev_ratio = None
        for t in ts:
            _, sig_t, _, _, conv = law.evaluate(E + t * u)
            resid = np.linalg.norm(sig_t - sig0 - t * pred_deriv)
            ratio = resid / t
            trend = ""
            if prev_ratio is not None and prev_ratio > 1e-30:
                trend = f" (x{ratio / prev_ratio:.3f} vs previous t)"
            print(f"    t={t:.0e}: converged={conv}, resid={resid:.6e}, resid/t={ratio:.6e}{trend}")
            prev_ratio = ratio


if __name__ == "__main__":
    main()
