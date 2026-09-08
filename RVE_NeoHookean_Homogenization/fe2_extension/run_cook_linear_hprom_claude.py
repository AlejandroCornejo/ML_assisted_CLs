#!/usr/bin/env python3
"""Cook's-membrane FE2 driven by the pure-linear (no ANN) HPROM law
(linear_hprom_iterative_law_float64_claude.py). Reuses run_cook_hprom_ann_
claude.py's own run_newton_fe2 unmodified (material-law-agnostic), same as
every other MATERIAL_FUNCS variant this session -- only registers a new
key with a continuation-threading wrapper (mirrors run_cruciform_fe2_
claude.py's HpromIterativeContinuationWrapper, adapted for this simpler,
q_prev/step_index-native law -- no qp_init_mode override needed at all).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from linear_hprom_iterative_law_float64_claude import LinearHpromIterativeLawFloat64  # noqa: E402
import run_cook_hprom_ann_claude as cook_module  # noqa: E402


class LinearHpromContinuationWrapper:
    """Per-macro-Gauss-point q_prev threading (by stable positional index),
    committing only CONVERGED iterates as the next warm start -- same
    discipline as HpromIterativeContinuationWrapper in run_cruciform_fe2_
    claude.py, adapted for LinearHpromIterativeLawFloat64's own (simpler,
    already-native) q_prev/step_index interface."""

    def __init__(self, law):
        self.law = law
        self.q_prev_by_point = {}

    def __call__(self, e_voigt, young=None, poisson=None):
        e_voigt = np.asarray(e_voigt, dtype=float)
        n = e_voigt.shape[0]
        S = np.zeros((n, 3), dtype=float)
        CC = np.zeros((n, 3, 3), dtype=float)
        n_nonconverged = 0
        for i in range(n):
            q_prev = self.q_prev_by_point.get(i)
            step_index = 1 if q_prev is None else 2
            _, S[i], q_p_new, _n_it, converged, _, CC[i], _, _ = self.law.evaluate_with_tangent(
                e_voigt[i], q_prev=q_prev, step_index=step_index,
            )
            if not converged:
                n_nonconverged += 1
            else:
                self.q_prev_by_point[i] = q_p_new
        if n_nonconverged:
            print(f"    [linear-hprom-continuation] {n_nonconverged}/{n} points did not converge internally this call")
        return S, CC


def make_linear_hprom_material_func():
    law = LinearHpromIterativeLawFloat64()
    return LinearHpromContinuationWrapper(law)


def run_newton_fe2_linear_hprom(nx=8, ny=8, **kwargs):
    cook_module.MATERIAL_FUNCS["linear_hprom"] = make_linear_hprom_material_func()
    return cook_module.run_newton_fe2("linear_hprom", nx=nx, ny=ny, **kwargs)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--nx", type=int, default=8)
    p.add_argument("--ny", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=cook_module.N_STEPS)
    a = p.parse_args()
    run_newton_fe2_linear_hprom(nx=a.nx, ny=a.ny, n_steps=a.n_steps)
