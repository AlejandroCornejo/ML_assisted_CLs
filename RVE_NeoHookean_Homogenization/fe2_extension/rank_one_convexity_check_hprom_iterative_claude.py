#!/usr/bin/env python3
"""Same rank-one convexity audit as rank_one_convexity_check_claude.py
(already validated there against energy-autograd on 'certified', machine-
precision agreement -- not re-validated here since the formula itself is
law-agnostic), applied to HpromAnnIterativeLaw (Stage 2) instead of
D-HPROM-ANN (Stage 1).

Also tracks the inner Newton loop's own convergence flag per sample,
since at the large, aggressive deformations this audit samples (stretches
up to ~2.7x, far beyond anything Cook's membrane produces), the inner
q_p-correction loop might not always converge cleanly -- reported
explicitly rather than silently ignored, so a high non-convergence rate
would be flagged as a caveat on the audit itself, not hidden in it.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from rank_one_convexity_check_claude import run_audit  # noqa: E402
from hprom_ann_iterative_law_claude import HpromAnnIterativeLaw  # noqa: E402

if __name__ == "__main__":
    print("[main] building HpromAnnIterativeLaw ...")
    law = HpromAnnIterativeLaw()

    n_nonconverged = [0]
    n_calls = [0]

    def eval_hprom_iter(E_voigt):
        n_calls[0] += 1
        _eps, sig, _q_p, _n_it, converged, _dEps, dSig, _eps_check, _sig_check = law.evaluate_with_tangent(E_voigt)
        if not converged:
            n_nonconverged[0] += 1
        return sig, dSig

    result = run_audit(eval_hprom_iter, label="HPROM-ANN-iterative", n_samples=2000,
                        stretch_log_range=1.0, seed=20260828)

    print(f"\n[main] inner Newton loop convergence: {n_calls[0] - n_nonconverged[0]}/{n_calls[0]} "
          f"queries converged cleanly ({n_nonconverged[0]} did not).")
