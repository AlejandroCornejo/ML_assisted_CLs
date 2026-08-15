#!/usr/bin/env python3
"""Fills a gap left by rank_one_check_at_cook_states_pann_nx8_claude.py
(which covers certified/ickan/free at nx=8 but not regression) and by
rank_one_check_at_cook_states_claude.py (which checks regression but
against the paper's own nx=16 Cook data, not this project's nx=8 run).
Same "Table-8-style" at-real-visited-states protocol, now for regression
at nx=8 specifically, using the confirmed-correct stall-tolerant full-ramp
file (cook_results_pann_regression_claude.npz, u_nodal max|uy|=3.2018,
NOT cook_results_regression_nx8_claude.npz, which is the old-driver
frozen-at-step-1 file with max|uy|=0.17).
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

from rank_one_check_at_cook_states_claude import audit_at_real_states, deformation_gradients_from_strains  # noqa: E402

if __name__ == "__main__":
    import pann_constitutive_law_claude as pann_law

    path = HERE / "cook_results_pann_regression_claude.npz"
    d = np.load(path)
    e_gp = d["e_gp"]
    print(f"[regression nx=8] loaded {path.name}: e_gp.shape={e_gp.shape}, "
          f"|e_gp|_max={np.abs(e_gp).max():.4e}, u_nodal max|uy|={np.abs(d['u_nodal'][:, 1]).max():.4f}")
    F_all = deformation_gradients_from_strains(e_gp)
    law = pann_law.get_law("regression")

    def eval_regression(E_voigt, _law=law):
        S, CC = _law.pk2_and_tangent(E_voigt.reshape(1, 3))
        return S[0], CC[0]

    r = audit_at_real_states(F_all, eval_regression, "regression (nx=8, correct stall-tolerant file)")
    print("\n=== RESULT ===")
    print(f"regression_nx8   n_gp={r['n_gp']:5d}  n_total={r['n_total']:6d}  "
          f"violations={r['n_violations']:5d} ({r['fraction']:.3%})  worst={r['worst_curvature']:.4e}")
